# FILE: merge_env.py

import re
import tempfile
import glob
import os
import json
import time
import subprocess
import traceback
from json import JSONDecoder
from typing import Tuple, Dict, Optional
import datetime
import gymnasium as gym
import shutil
import numpy as np
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter
import networkx as nx

from graph_tracker import GraphTracker
from reward_info_extractor import RewardInfoExtractor, MergeInfo, validate_extracted_info
from reward_function_variants import create_reward_function

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

TB_LOGDIR = "tb_debug"
writer = SummaryWriter(log_dir=TB_LOGDIR)


def _safe_load_list(path: str, retries: int = 60, delay: float = 0.25) -> list:
    """Robustly load the *first* JSON value from `path` with retries."""
    dec = JSONDecoder()
    last_err = None
    for _ in range(retries):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                s = f.read().lstrip()
            if not s:
                time.sleep(delay)
                continue
            obj, _ = dec.raw_decode(s)
            return obj if isinstance(obj, list) else []
        except (OSError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(delay)
    return []


def write_json_atomic(obj, final_path: str):
    """Write `obj` to JSON at `final_path` atomically."""
    dir_ = os.path.dirname(final_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)


def _wait_json_complete(path: str, timeout: float = 180.0, poll: float = 0.25) -> bool:
    """Wait until `path` exists and looks complete."""
    deadline = time.time() + timeout
    last_size = -1
    while time.time() < deadline:
        try:
            if not os.path.exists(path):
                time.sleep(poll)
                continue
            size = os.path.getsize(path)
            if size <= 0:
                time.sleep(poll)
                continue

            with open(path, "rb") as f:
                if size > 8192:
                    f.seek(-8192, os.SEEK_END)
                tail = f.read()

            tail = tail.rstrip(b"\r\n\t ")
            if not tail:
                time.sleep(poll)
                continue

            last_byte = tail[-1]
            looks_closed = last_byte in (ord("]"), ord("}"))

            if looks_closed and size == last_size:
                return True

            last_size = size
            time.sleep(poll)
        except OSError:
            time.sleep(poll)
    return False


class MergeEnv(gym.Env):
    """
    Gym environment wrapping Fast Downward's merge-and-shrink heuristic.
    ✅ FIXED: Now properly stores domain and problem file paths.
    """

    metadata = {"render.modes": []}

    def __init__(
            self,
            domain_file: str,
            problem_file: str,
            max_merges: int = 20,
            debug: bool = False,
            reward_variant: str = 'rich',
            max_states: int = 4000,
            threshold_before_merge: int = 1,
            **reward_kwargs
    ) -> None:
        """
        Initialize the merge-and-shrink learning environment.

        Args:
            domain_file: Absolute or relative path to domain.pddl
            problem_file: Absolute or relative path to problem.pddl
            max_merges: Maximum number of merges allowed per episode
            debug: If True, use in-memory debug mode (no real FD)
            reward_variant: Which reward function to use ('rich', 'astar_search', etc.)
            max_states: Max abstract states for M&S algorithm
            threshold_before_merge: Threshold for triggering shrinking
            **reward_kwargs: Additional kwargs for reward function (w_f_stability, etc.)
        """
        super().__init__()

        # ✅ CRITICAL: Store file paths as ABSOLUTE paths
        self.domain_file = os.path.abspath(domain_file)
        self.problem_file = os.path.abspath(problem_file)

        # ✅ CRITICAL FIX #1: Initialize fd_base_dir (was missing!)
        self.fd_base_dir = os.path.abspath("downward")

        # Verify files exist (fail early)
        if not os.path.exists(self.domain_file):
            raise FileNotFoundError(f"Domain file not found: {self.domain_file}")
        if not os.path.exists(self.problem_file):
            raise FileNotFoundError(f"Problem file not found: {self.problem_file}")

        logger.info(f"Environment initialized with:")
        logger.info(f"  Domain:  {self.domain_file}")
        logger.info(f"  Problem: {self.problem_file}")
        logger.info(f"  FD base: {self.fd_base_dir}")

        # ✅ Store M&S hyperparameters
        self.max_states = max_states
        self.threshold_before_merge = threshold_before_merge
        logger.info(f"  max_states: {self.max_states}")
        logger.info(f"  threshold_before_merge: {self.threshold_before_merge}")

        # ✅ Setup reward function with ALL parameters
        self.max_merges = max(1, max_merges)
        try:
            self.reward_function = create_reward_function(reward_variant, **reward_kwargs)
            logger.info(f"✓ Initialized reward function: {reward_variant}")
        except Exception as e:
            logger.error(f"Failed to initialize reward function '{reward_variant}': {e}")
            raise

        # ✅ Setup signal extraction
        self.reward_info_extractor = RewardInfoExtractor(fd_output_dir=os.path.join(self.fd_base_dir, "fd_output"))
        logger.info("✓ Initialized reward info extractor")

        # ✅ Initialize state variables
        self.current_merge_step = 0
        self.process = None  # FD subprocess
        self.fd_log_file = None
        self.graph_tracker: GraphTracker = None

        # ✅ Setup logging directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("logs", exist_ok=True)
        self.log_path = f"logs/run_{timestamp}.jsonl"

        # ✅ NEW: GNN METADATA COLLECTION
        self.gnn_metadata_dir = os.path.join("downward", "gnn_metadata")
        os.makedirs(self.gnn_metadata_dir, exist_ok=True)
        self.gnn_decisions_log = []  # Collect all GNN decisions in episode

        # ✅ Initialize observation tracking
        self.prev_total_states = 0
        self.max_vars = 1
        self.max_iter = 1
        self.centrality = {}

        # ✅ Define observation and action spaces (for gym.Env)
        self.feat_dim = 19  # Number of node features per node ✅ NEW: Expanded feature set
        # self.observation_space = spaces.Dict({
        #     "x": spaces.Box(0.0, 1.0, shape=(100, self.feat_dim), dtype=np.float32),
        #     "edge_index": spaces.Box(0, 100, shape=(2, 1000), dtype=np.int64),
        #     "num_nodes": spaces.Box(0, 100, shape=(), dtype=np.int32),
        #     "num_edges": spaces.Box(0, 1000, shape=(), dtype=np.int32),
        # })
        # ✅ UPDATED: Add edge features to observation space
        self.observation_space = spaces.Dict({
            "x": spaces.Box(0.0, 1.0, shape=(100, self.feat_dim), dtype=np.float32),
            "edge_index": spaces.Box(0, 100, shape=(2, 1000), dtype=np.int64),
            "edge_features": spaces.Box(  # ✅ NEW
                -1.0, 1.0,
                shape=(1000, 8),
                dtype=np.float32
            ),
            "num_nodes": spaces.Box(0, 100, shape=(), dtype=np.int32),
            "num_edges": spaces.Box(0, 1000, shape=(), dtype=np.int32),
        })
        self.action_space = spaces.Discrete(1000)  # Max 1000 possible merges

        # ✅ Store debug mode
        self.debug = debug

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset the env."""
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5.0)
        except Exception:
            pass
        if self.fd_log_file:
            try:
                self.fd_log_file.close()
            except Exception:
                pass
            self.fd_log_file = None

        super().reset(seed=seed)

        # for folder in ("gnn_output", "fd_output"):
        #     d = os.path.join("downward", folder)
        #     if not os.path.isdir(d):
        #         continue
        #     for fname in os.listdir(d):
        #         if folder == "fd_output" and fname == "log.txt":
        #             continue
        #         try:
        #             os.remove(os.path.join(d, fname))
        #         except Exception:
        #             pass

        # ✅ CRITICAL: Ensure output directories exist BEFORE cleaning
        os.makedirs("downward/gnn_output", exist_ok=True)
        os.makedirs("downward/fd_output", exist_ok=True)

        # ✅ Clean old files
        for folder in ("gnn_output", "fd_output"):
            d = os.path.join("downward", folder)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if folder == "fd_output" and fname == "log.txt":
                    continue
                try:
                    fpath = os.path.join(d, fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        logger.debug(f"Cleaned: {fpath}")
                except Exception as e:
                    logger.warning(f"Could not clean {fpath}: {e}")

        toy_dir = os.environ.get("TOY_TS", None)
        ts_file, cg_file = "merged_transition_systems.json", "causal_graph.json"
        if toy_dir and self.debug:
            ts_path = os.path.join(toy_dir, ts_file)
            cg_path = os.path.join(toy_dir, cg_file)
        else:
            ts_path = cg_path = None
            for attempt in range(1, 4):
                try:
                    ts_path, cg_path = self._handshake_with_fd()
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Handshake attempt {attempt} failed: {e}")
                    time.sleep(1.0)
            if ts_path is None or cg_path is None:
                if self.debug:
                    logger.warning("⚠️ Handshake failed; falling back to toy data")
                    toy_dir = os.environ.get("TOY_TS", "toy")
                    ts_path = os.path.join(toy_dir, ts_file)
                    cg_path = os.path.join(toy_dir, cg_file)
                else:
                    raise RuntimeError("Handshake with FD failed and debug mode is off")

        # ✅ ADD: Export metadata from previous episode
        if self.gnn_decisions_log:
            self._export_episode_metadata()
            self.gnn_decisions_log = []

        self.current_merge_step = 0
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"logs/run_{ts}.jsonl"

        self.graph_tracker = GraphTracker(ts_json_path=ts_path, cg_json_path=cg_path, is_debug=self.debug)
        G = self.graph_tracker.graph
        self.max_vars = max(
            (len(d.get("incorporated_variables", [])) for _, d in G.nodes(data=True)),
            default=1
        ) or 1
        self.max_iter = max(
            (d.get("iteration", 0) for _, d in G.nodes(data=True)),
            default=0
        ) or 1
        self.centrality = nx.closeness_centrality(G)

        obs = self._get_observation()
        self.prev_total_states = self._count_total_states()
        self.state = obs
        return obs, {}

    def _handshake_with_fd(self) -> Tuple[str, str]:
        """Launch FD and wait for initial JSONs."""
        dw_dir = os.path.abspath("downward")
        if not os.path.isdir(dw_dir):
            raise RuntimeError(f"Downward folder not found: {dw_dir}")

        def _tail(path: str, n: int = 120) -> str:
            if not os.path.exists(path):
                return "(no log file yet)"
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                return "".join(lines[-n:])
            except Exception as e:
                return f"(failed to read log tail: {e})"

        def _robust_copy(src: str, dst: str, tries: int = 80, delay: float = 0.1):
            for _ in range(tries):
                try:
                    with open(src, "rb") as fin, open(dst, "wb") as fout:
                        fout.write(fin.read())
                    return
                except OSError:
                    time.sleep(delay)
            raise RuntimeError(f"Could not copy {src} -> {dst}")

        def _wait_stable(path: str, grace: float = 0.15, timeout: float = 30.0) -> bool:
            end = time.time() + timeout
            last = -1
            while time.time() < end:
                if os.path.exists(path):
                    try:
                        sz = os.path.getsize(path)
                    except OSError:
                        sz = -1
                    if sz > 0 and sz == last:  # File exists and size hasn't changed
                        return True
                    last = sz
                time.sleep(grace)
            return False

        ## STEP 1: Clean up old files
        logger.info("[Handshake] Cleaning up old files...")
        for fname in ("causal_graph.json", "merged_transition_systems.json", "output.sas"):
            p = os.path.join(dw_dir, fname)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete {p}: {e}")

        for subdir_name in ["gnn_output", "fd_output"]:
            subdir_path = os.path.join(dw_dir, subdir_name)
            os.makedirs(subdir_path, exist_ok=True)
            for f in os.listdir(subdir_path):
                try:
                    os.remove(os.path.join(subdir_path, f))
                except Exception:
                    pass

        ## STEP 2: Safely terminate any previous FD process
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=3.0)
        except (subprocess.TimeoutExpired, AttributeError):
            if hasattr(self, 'process') and self.process:
                self.process.kill()
        except Exception:
            pass

        ## STEP 3: Build the FD command with GNN merge strategy
        # ✅ NEW: Build FD command from scratch using stored file paths
        fd_translate_bin = os.path.join(dw_dir, "builds/release/bin/translate/translate.py")
        fd_downward_exe = os.path.join(dw_dir, "builds/release/bin/downward.exe")

        # ✅ Use stored paths (absolute)
        domain_path = self.domain_file
        problem_path = self.problem_file

        logger.info(f"[Handshake] Building FD command:")
        logger.info(f"  Domain:  {domain_path}")
        logger.info(f"  Problem: {problem_path}")

        # ✅ Build complete FD command with merge_gnn strategy
        cmd = (
            f'python "{fd_translate_bin}" "{domain_path}" "{problem_path}" '
            f'--sas-file output.sas && '
            f'"{fd_downward_exe}" '
            r'--search "astar(merge_and_shrink('
            r'merge_strategy=merge_gnn(),'  # ✅ Uses GNN for merge decisions
            r'shrink_strategy=shrink_bisimulation(greedy=false,at_limit=return),'
            r'label_reduction=exact(before_shrinking=true,before_merging=false),'
            f'max_states={self.max_states},'  # ✅ Use stored parameter
            f'threshold_before_merge={self.threshold_before_merge}'  # ✅ Use stored parameter
            r'))" < output.sas'
        )

        logger.info(f"[Handshake] FD Command: {cmd[:150]}...")

        ## STEP 4: Launch FD process
        fd_dir = os.path.join(dw_dir, "fd_output")
        log_path = os.path.join(fd_dir, "log.txt")
        self.fd_log_file = open(log_path, "w", buffering=1, encoding="utf-8")

        self.process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=dw_dir,
            stdout=self.fd_log_file,
            stderr=self.fd_log_file,
        )

        ## STEP 5: Wait for translator
        logger.info("[Handshake] Waiting for translator to produce 'output.sas'...")
        sas_path = os.path.join(dw_dir, "output.sas")
        if not _wait_stable(sas_path, timeout=60.0):
            tail = _tail(log_path)
            raise RuntimeError(f"Translator failed to produce output.sas.\nLog tail:\n{tail}")
        logger.info("[Handshake] 'output.sas' is ready.")

        ## STEP 6: Wait for initial JSONs
        logger.info("[Handshake] Waiting for planner to produce initial JSONs...")
        cg_root = os.path.join(dw_dir, "causal_graph.json")
        ts_root = os.path.join(dw_dir, "merged_transition_systems.json")
        start, timeout_s = time.time(), 240.0

        while not (os.path.exists(cg_root) and os.path.exists(ts_root)):
            if self.process and (self.process.poll() is not None):
                tail = _tail(log_path)
                raise RuntimeError(
                    f"FD exited early (code {self.process.returncode}) while waiting for JSONs.\nLog tail:\n{tail}")
            if time.time() - start > timeout_s:
                tail = _tail(log_path)
                raise RuntimeError(f"Timeout waiting for initial JSONs after {timeout_s:.0f}s.\nLog tail:\n{tail}")
            time.sleep(0.1)

        ## STEP 7: Copy and verify
        logger.info("[Handshake] Initial JSONs found. Verifying and copying...")
        if not (_wait_stable(cg_root) and _wait_stable(ts_root)):
            tail = _tail(log_path)
            raise RuntimeError(f"Root JSONs never stabilized.\nLog tail:\n{tail}")

        fd_cg = os.path.join(fd_dir, "causal_graph.json")
        fd_ts = os.path.join(fd_dir, "merged_transition_systems.json")

        dec = JSONDecoder()
        for _ in range(80):
            _robust_copy(cg_root, fd_cg)
            _robust_copy(ts_root, fd_ts)
            try:
                with open(fd_ts, "r", encoding="utf-8", errors="ignore") as f:
                    s = f.read().lstrip()
                if not s:
                    time.sleep(0.1)
                    continue
                obj, _ = dec.raw_decode(s)
                break
            except json.JSONDecodeError:
                time.sleep(0.1)
        else:
            tail = _tail(log_path)
            raise RuntimeError("Copied TS JSON is not parseable.\n" + tail)

        logger.info("[Handshake] Handshake complete.")
        return fd_ts, fd_cg

    def _log_step(self, src: int, tgt: int, info: dict, reward: float, done: bool):
        """Append merge-step metrics to log file."""
        total_states = self._count_total_states()

        entry = {
            "step": int(self.current_merge_step),
            "merge_choice": [int(src), int(tgt)],
            "plan_cost": int(info.get("plan_cost", 0)),
            "num_expansions": int(info.get("num_expansions", 0)),
            "num_significant_f_changes": int(info.get("num_significant_f_changes", 0)),
            "delta_states": int(info.get("delta_states", 0)),
            "total_states": int(total_states),
            "reward": float(reward),
            "done": bool(done),
            "timestamp": time.time(),
        }

        with open(self.log_path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """✅ FIXED: Action handling with complete validation."""
        try:
            # Check if FD process crashed
            if self.process is not None:
                ret = self.process.poll()
                if ret is not None and ret < 10:
                    print(f"[ERROR] FD process exited with code {ret}")
                    return self.state, 0.0, True, False, {"error": "process_crashed"}

            # Get current edges
            edges = list(self.graph_tracker.graph.edges)
            if not edges:
                print("[WARNING] No edges available for merging")
                return self.state, 0.0, True, False, {"error": "no_edges"}

            # ✅ FIX: Action validation and clamping
            action = int(action)
            num_valid_edges = len(edges)

            # Clamp to valid range: [0, num_valid_edges-1]
            action_idx = max(0, min(action % max(num_valid_edges, 1), num_valid_edges - 1))

            print(f"[DEBUG] Action {action} → index {action_idx} (edges: {num_valid_edges})")

            src, tgt = edges[action_idx]

            # ✅ VALIDATION: Check nodes exist and are different
            if src == tgt:
                print(f"[ERROR] Self-merge attempted: {src} == {tgt}")
                return self.state, 0.0, True, False, {"error": "self_merge"}

            if src not in self.graph_tracker.graph or tgt not in self.graph_tracker.graph:
                print(f"[ERROR] Merge nodes invalid: ({src}, {tgt})")
                return self.state, 0.0, True, False, {"error": "invalid_nodes"}

            if self.debug:
                # Need to extract (src, tgt) for debug mode
                edges = list(self.graph_tracker.graph.edges())
                action_idx = max(0, min(int(action) % max(len(edges), 1), len(edges) - 1))
                src, tgt = edges[action_idx]
                return self._step_debug(src, tgt)
            else:
                return self._step_real(action)  # ✅ Pass action, not (src, tgt)

        except Exception as e:
            print(f"[ERROR] Exception in step(): {e}")
            import traceback
            traceback.print_exc()
            return self.state, 0.0, True, False, {"error": str(e)}

    def _step_debug(self, src: int, tgt: int):
        """
        ✅ ENHANCED: In-memory merge (debug mode) with detailed logging.

        Shows every step of the merge process with:
        - State before and after
        - Computation details
        - Graph metrics
        - Reward calculation
        """
        logger.info("\n" + "=" * 90)
        logger.info(f"STEP {self.current_merge_step}: DEBUG MODE MERGE")
        logger.info("=" * 90)

        try:
            # ====================================================================
            # PHASE 1: ENTRY LOGGING & VALIDATION
            # ====================================================================

            logger.info("\n[PHASE 1] ENTRY VALIDATION")
            logger.info(f"  Merge indices: ({src}, {tgt})")
            logger.info(f"  Current merge step: {self.current_merge_step}")
            logger.info(f"  Max merges allowed: {self.max_merges}")

            # ====================================================================
            # PHASE 2: PRE-MERGE STATE SNAPSHOT
            # ====================================================================

            logger.info("\n[PHASE 2] PRE-MERGE STATE SNAPSHOT")

            # Count total states before merge
            old_total = self._count_total_states()
            logger.info(f"  Total states before merge: {old_total}")

            # Log node details
            G = self.graph_tracker.graph
            logger.info(f"  Num nodes before: {len(G.nodes)}")
            logger.info(f"  Num edges before: {len(G.edges)}")

            # Log specific nodes being merged
            if src in G.nodes and tgt in G.nodes:
                src_data = G.nodes[src]
                tgt_data = G.nodes[tgt]
                logger.info(f"\n  Node {src} (source):")
                logger.info(f"    - num_states: {src_data.get('num_states', 'N/A')}")
                logger.info(f"    - iteration: {src_data.get('iteration', 'N/A')}")
                logger.info(f"    - incorporated_variables: {src_data.get('incorporated_variables', 'N/A')}")

                logger.info(f"\n  Node {tgt} (target):")
                logger.info(f"    - num_states: {tgt_data.get('num_states', 'N/A')}")
                logger.info(f"    - iteration: {tgt_data.get('iteration', 'N/A')}")
                logger.info(f"    - incorporated_variables: {tgt_data.get('incorporated_variables', 'N/A')}")

            # ====================================================================
            # PHASE 3: PERFORM MERGE
            # ====================================================================

            logger.info("\n[PHASE 3] PERFORMING MERGE")
            logger.info(f"  Calling graph_tracker.merge_nodes([{src}, {tgt}])...")

            self.graph_tracker.merge_nodes([src, tgt])
            self.current_merge_step += 1

            logger.info(f"  ✓ Merge completed, current_merge_step = {self.current_merge_step}")

            # ====================================================================
            # PHASE 4: POST-MERGE STATE SNAPSHOT
            # ====================================================================

            logger.info("\n[PHASE 4] POST-MERGE STATE SNAPSHOT")

            new_total = self._count_total_states()
            delta_states = new_total - old_total

            logger.info(f"  Total states after merge: {new_total}")
            logger.info(f"  Delta states (new - old): {delta_states}")
            logger.info(f"  State explosion ratio: {delta_states / max(old_total, 1):.4f}")

            # Log updated graph
            G = self.graph_tracker.graph
            logger.info(f"  Num nodes after: {len(G.nodes)}")
            logger.info(f"  Num edges after: {len(G.edges)}")

            # ====================================================================
            # PHASE 5: UPDATE GRAPH METRICS
            # ====================================================================

            logger.info("\n[PHASE 5] UPDATE GRAPH METRICS")

            self.max_vars = max(
                (len(d.get("incorporated_variables", [])) for _, d in G.nodes(data=True)),
                default=1
            ) or 1
            logger.info(f"  max_vars: {self.max_vars}")

            self.max_iter = max(
                (d.get("iteration", 0) for _, d in G.nodes(data=True)),
                default=0
            ) or 1
            logger.info(f"  max_iter: {self.max_iter}")

            self.centrality = nx.closeness_centrality(G)
            logger.info(f"  Centrality computed for {len(self.centrality)} nodes")

            # ====================================================================
            # PHASE 6: COMPUTE REWARD
            # ====================================================================

            logger.info("\n[PHASE 6] REWARD COMPUTATION")

            reward = -max(abs(delta_states), 0.1)
            logger.info(f"  Reward formula: -max(|delta_states|, 0.1)")
            logger.info(f"  Computed reward: {reward:.4f}")

            # ====================================================================
            # PHASE 7: TENSORBOARD LOGGING
            # ====================================================================

            logger.info("\n[PHASE 7] TENSORBOARD LOGGING")

            try:
                writer.add_scalar("env/num_nodes", len(self.graph_tracker.graph.nodes),
                                  self.current_merge_step)
                logger.info(f"  ✓ Logged num_nodes: {len(self.graph_tracker.graph.nodes)}")

                writer.add_scalar("env/num_edges", len(self.graph_tracker.graph.edges),
                                  self.current_merge_step)
                logger.info(f"  ✓ Logged num_edges: {len(self.graph_tracker.graph.edges)}")

                writer.add_scalar("env/total_states", new_total, self.current_merge_step)
                logger.info(f"  ✓ Logged total_states: {new_total}")

                writer.add_scalar("env/delta_states", delta_states, self.current_merge_step)
                logger.info(f"  ✓ Logged delta_states: {delta_states}")

                writer.add_scalar("env/reward", reward, self.current_merge_step)
                logger.info(f"  ✓ Logged reward: {reward:.4f}")
            except Exception as e:
                logger.warning(f"  ⚠️ TensorBoard logging error: {e}")

            # ====================================================================
            # PHASE 8: BUILD OBSERVATION
            # ====================================================================

            logger.info("\n[PHASE 8] BUILD OBSERVATION")

            obs = self._get_observation()
            logger.info(f"  Observation built successfully")
            logger.info(f"  - x shape: {obs['x'].shape}")
            logger.info(f"  - edge_index shape: {obs['edge_index'].shape}")
            logger.info(f"  - num_nodes: {obs['num_nodes']}")
            logger.info(f"  - num_edges: {obs['num_edges']}")

            # ====================================================================
            # PHASE 9: TERMINATION CHECK
            # ====================================================================

            logger.info("\n[PHASE 9] TERMINATION CHECK")

            num_remaining = len(G.nodes)
            done = (num_remaining <= 1) or (self.current_merge_step >= self.max_merges)

            logger.info(f"  Num remaining nodes: {num_remaining}")
            logger.info(f"  Current step: {self.current_merge_step} / {self.max_merges}")
            logger.info(f"  Done: {done}")

            if done:
                if num_remaining <= 1:
                    logger.info(f"  Reason: Only {num_remaining} node(s) left")
                else:
                    logger.info(f"  Reason: Max merges reached")

            # ====================================================================
            # PHASE 10: BUILD INFO DICT
            # ====================================================================

            logger.info("\n[PHASE 10] BUILD INFO DICT")

            info = {
                "num_nodes": len(self.graph_tracker.graph.nodes),
                "num_edges": len(self.graph_tracker.graph.edges),
                "total_states": new_total,
                "delta_states": delta_states,
                "plan_cost": 0,
                "num_expansions": 0,
                "error": None,
            }
            logger.info(f"  Info dict built with {len(info)} keys")

            # ====================================================================
            # PHASE 11: LOG STEP & UPDATE STATE
            # ====================================================================

            logger.info("\n[PHASE 11] LOGGING & STATE UPDATE")

            self._log_step(src, tgt, info, reward, done)
            logger.info(f"  ✓ Step logged to file")

            self.state = obs
            logger.info(f"  ✓ State updated")

            # ====================================================================
            # PHASE 12: RETURN
            # ====================================================================

            logger.info("\n[PHASE 12] RETURN")
            logger.info(f"  Returning: obs, reward={reward:.4f}, done={done}, info")
            logger.info("=" * 90 + "\n")



            return obs, reward, done, False, info

        except Exception as e:
            logger.error(f"\n❌ ERROR in _step_debug: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.info("=" * 90 + "\n")
            return self.state, 0.0, True, False, {"error": str(e)}

    def _step_real(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        ✅ ENHANCED: Execute one merge step in REAL mode with BAD MERGE DETECTION.

        This function now:
        1. Validates action and extracts merge pair
        2. Writes merge decision to FD
        3. Waits for FD acknowledgment
        4. Waits for updated TS files
        5. Updates graph tracker
        6. ✅ DETECTS BAD MERGE SITUATIONS
        7. Extracts signals and computes reward
        8. Returns result with comprehensive logging
        """

        logger.info("\n" + "=" * 90)
        logger.info(f"STEP {self.current_merge_step}: REAL MODE MERGE WITH BAD MERGE DETECTION")
        logger.info("=" * 90)

        try:
            # ====================================================================
            # PHASE 1: VALIDATE INPUT & EXTRACT MERGE PAIR
            # ====================================================================

            logger.info("\n[PHASE 1] INPUT VALIDATION & ACTION EXTRACTION")
            logger.info(f"  Input action: {action} (type: {type(action).__name__})")

            if not hasattr(self, 'graph_tracker') or self.graph_tracker is None:
                raise RuntimeError("graph_tracker not initialized")

            # Get current edges
            edges = list(self.graph_tracker.graph.edges())
            logger.info(f"  Available edges in graph: {len(edges)}")

            if not edges:
                logger.warning("  ⚠️ No edges available for merging")
                return self.state, 0.0, True, False, {"error": "no_edges"}

            # ✅ FIX: Action validation and extraction
            action = int(action)
            num_valid_edges = len(edges)
            action_idx = max(0, min(action % max(num_valid_edges, 1), num_valid_edges - 1))

            logger.info(f"  Action index calculation:")
            logger.info(
                f"    - action % max(num_valid_edges, 1) = {action} % {max(num_valid_edges, 1)} = {action % max(num_valid_edges, 1)}")
            logger.info(f"    - clamped to [0, {num_valid_edges - 1}]: {action_idx}")

            node_a, node_b = edges[action_idx]
            logger.info(f"  ✓ Extracted merge pair from edge {action_idx}: ({node_a}, {node_b})")

            # ✅ VALIDATION: Check nodes exist and are different
            if node_a == node_b:
                logger.error(f"  ❌ ERROR: Self-merge attempted: {node_a} == {node_b}")
                return self.state, 0.0, True, False, {"error": "self_merge"}

            if node_a not in self.graph_tracker.graph or node_b not in self.graph_tracker.graph:
                logger.error(f"  ❌ ERROR: Merge nodes invalid: ({node_a}, {node_b})")
                return self.state, 0.0, True, False, {"error": "invalid_nodes"}

            src = node_a
            tgt = node_b

            logger.info(f"  ✓ Validation passed")

            # ====================================================================
            # PHASE 2: PRE-MERGE DIAGNOSTICS & DECISION LOGGING
            # ====================================================================

            logger.info(f"\n[PHASE 2] PRE-MERGE STATE ANALYSIS")

            pre_merge_total = sum(d.get('num_states', 0) for _, d in self.graph_tracker.graph.nodes(data=True))
            logger.info(f"    - Total states before merge: {pre_merge_total}")
            logger.info(f"    - Num nodes: {len(self.graph_tracker.graph.nodes)}")
            logger.info(f"    - Num edges: {len(self.graph_tracker.graph.edges)}")

            # Get node-specific data
            src_data = self.graph_tracker.graph.nodes[src]
            tgt_data = self.graph_tracker.graph.nodes[tgt]
            src_size = src_data.get('num_states', 1)
            tgt_size = tgt_data.get('num_states', 1)
            expected_product = src_size * tgt_size

            logger.info(f"\n    Nodes being merged:")
            logger.info(f"      - Node {src}: {src_size} states")
            logger.info(f"      - Node {tgt}: {tgt_size} states")
            logger.info(f"      - Expected product: {expected_product} states")

            # ====================================================================
            # PHASE 3: WRITE MERGE DECISION
            # ====================================================================

            logger.info(f"\n[PHASE 3] WRITE MERGE DECISION")

            merge_decision = {
                "iteration": self.current_merge_step,
                "merge_pair": [int(src), int(tgt)],
                "timestamp": time.time()
            }

            gnn_output_dir = os.path.abspath(os.path.join(self.fd_base_dir, "gnn_output"))
            os.makedirs(gnn_output_dir, exist_ok=True)

            merge_decision_path = os.path.join(gnn_output_dir, f"merge_{self.current_merge_step}.json")

            # ✅ Write atomically
            import tempfile
            try:
                fd, temp_path = tempfile.mkstemp(
                    dir=gnn_output_dir,
                    suffix=".json",
                    prefix=f"merge_{self.current_merge_step}_"
                )
                with os.fdopen(fd, 'w') as f:
                    json.dump(merge_decision, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(temp_path, merge_decision_path)
                logger.info(f"  ✓ Wrote merge decision: {merge_decision_path}")

            except Exception as e:
                logger.error(f"  ❌ Failed to write merge decision: {e}")
                raise

            # ====================================================================
            # PHASE 4: WAIT FOR FD ACKNOWLEDGMENT
            # ====================================================================

            logger.info(f"\n[PHASE 4] WAIT FOR FD ACKNOWLEDGMENT")

            ack_path = os.path.join(
                self.fd_base_dir,
                "fd_output",
                f"gnn_ack_{self.current_merge_step}.json"
            )

            logger.info(f"  Expected ACK file: {ack_path}")

            start_time = time.time()
            ACK_TIMEOUT = 30.0

            while time.time() - start_time < ACK_TIMEOUT:
                elapsed = time.time() - start_time

                if os.path.exists(ack_path):
                    try:
                        with open(ack_path) as f:
                            ack_data = json.load(f)
                        elapsed = time.time() - start_time
                        logger.info(f"  ✓ ACK received after {elapsed:.2f}s")
                        break
                    except json.JSONDecodeError:
                        time.sleep(0.1)
                        continue

                if int(elapsed) > 0 and int(elapsed) % 5 == 0:
                    logger.debug(f"  Still waiting... ({elapsed:.0f}s)")

                time.sleep(0.1)
            else:
                elapsed = time.time() - start_time
                raise TimeoutError(
                    f"FD did not acknowledge merge within {ACK_TIMEOUT}s (waited {elapsed:.1f}s)")

            # ====================================================================
            # PHASE 5: WAIT FOR UPDATED TS FILE
            # ====================================================================

            logger.info(f"\n[PHASE 5] WAIT FOR UPDATED TS FILE")

            ts_path = os.path.join(
                self.fd_base_dir,
                "fd_output",
                f"ts_{self.current_merge_step}.json"
            )

            logger.info(f"  Expected TS file: {ts_path}")

            start_time = time.time()
            TS_TIMEOUT = 60.0

            while time.time() - start_time < TS_TIMEOUT:
                elapsed = time.time() - start_time

                if os.path.exists(ts_path):
                    try:
                        with open(ts_path) as f:
                            ts_data = json.load(f)
                        elapsed = time.time() - start_time
                        logger.info(f"  ✓ TS file received after {elapsed:.2f}s")
                        break
                    except (json.JSONDecodeError, IOError):
                        time.sleep(0.2)
                        continue

                if int(elapsed) > 0 and int(elapsed) % 10 == 0:
                    logger.debug(f"  Still waiting... ({elapsed:.0f}s)")

                time.sleep(0.2)
            else:
                elapsed = time.time() - start_time
                raise TimeoutError(
                    f"TS file not produced within {TS_TIMEOUT}s (waited {elapsed:.1f}s)")

            # ====================================================================
            # PHASE 6: UPDATE GRAPH TRACKER
            # ====================================================================

            logger.info(f"\n[PHASE 6] UPDATE GRAPH TRACKER")
            logger.info(f"  Calling graph_tracker.merge_nodes([{src}, {tgt}])...")

            try:
                # ✅ --- WRAP THIS CALL ---
                self.graph_tracker.merge_nodes([src, tgt])
                logger.info(f"  ✓ Merged in graph tracker")
            except ValueError as e_merge:
                # ✅ --- HANDLE THE REJECTED MERGE ---
                logger.error(f"  ❌ Graph merge failed (rejected by tracker): {e_merge}")
                # Treat this as a failed step - return negative reward and terminate episode
                reward = -2.0  # Assign a strong negative reward for invalid merge attempt
                done = True
                info = {"error": "merge_rejected", "message": str(e_merge)}
                obs = self._get_observation()  # Get current observation
                self._log_step(src, tgt, info, reward, done)  # Log the failure
                return obs, reward, done, False, info  # Return immediately
                # ✅ --- END HANDLING ---
            except Exception as e:  # Catch other potential errors during merge
                logger.error(f"  ❌ Graph merge failed: {e}")
                raise  # Re-raise unexpected errors

            # ====================================================================
            # PHASE 7: EXTRACT SIGNALS & COMPUTE REWARD
            # ====================================================================

            logger.info(f"\n[PHASE 7] EXTRACT SIGNALS & COMPUTE REWARD")
            logger.info(f"  Calling reward_info_extractor.extract_merge_info(iteration={self.current_merge_step})...")

            try:
                merge_info = self.reward_info_extractor.extract_merge_info(
                    iteration=self.current_merge_step,
                    timeout=10.0
                )

                if merge_info is None:
                    logger.warning(f"  ⚠️ Failed to extract merge info, using defaults")
                    reward = 0.0
                else:
                    logger.info(f"  ✓ Merge info extracted successfully")

                    # ✅ ENHANCED: Log extracted signals with bad merge context
                    logger.info(f"\n  Extracted signals:")
                    logger.info(f"    - iteration: {merge_info.iteration}")
                    logger.info(f"    - states_before: {merge_info.states_before}")
                    logger.info(f"    - states_after: {merge_info.states_after}")
                    logger.info(f"    - delta_states: {merge_info.delta_states}")
                    logger.info(f"    - f_value_stability: {merge_info.f_value_stability:.4f}")
                    logger.info(f"    - num_significant_f_changes: {merge_info.num_significant_f_changes}")
                    logger.info(f"    - state_explosion_penalty: {merge_info.state_explosion_penalty:.4f}")
                    logger.info(f"    - nodes_expanded: {merge_info.nodes_expanded}")
                    logger.info(f"    - search_depth: {merge_info.search_depth}")
                    logger.info(f"    - solution_cost: {merge_info.solution_cost}")
                    logger.info(f"    - branching_factor: {merge_info.branching_factor:.4f}")
                    logger.info(f"    - solution_found: {merge_info.solution_found}")

                    # ✅ NEW: Pre-reward bad merge diagnostics
                    logger.info(f"\n  [BAD MERGE DIAGNOSTIC CHECKS]:")

                    # Check 1: State explosion
                    if merge_info.states_after > expected_product * 1.5:
                        logger.warning(
                            f"    ⚠️  State explosion detected: {merge_info.states_after} > {expected_product * 1.5:.0f}")

                    # Check 2: Goal reachability
                    goal_reachable = any(f != float('inf') and f < 1_000_000_000 for f in merge_info.f_after)
                    if not goal_reachable:
                        logger.error(f"    ❌ CRITICAL: Goal unreachable after merge!")

                    # Check 3: F-stability
                    if merge_info.f_value_stability < 0.3:
                        logger.warning(f"    ⚠️  Poor F-stability: {merge_info.f_value_stability:.4f}")

                    # Check 4: Unreachable states
                    unreachable_count = sum(1 for f in merge_info.f_after if f == float('inf') or f >= 1_000_000_000)
                    unreachable_ratio = unreachable_count / max(merge_info.states_after, 1)
                    if unreachable_ratio > 0.7:
                        logger.warning(f"    ⚠️  High unreachability: {unreachable_ratio * 100:.1f}%")

                    # Check 5: Branching factor
                    if merge_info.branching_factor > 8.0:
                        logger.warning(f"    ⚠️  High branching factor: {merge_info.branching_factor:.4f}")

                    logger.info(f"\n  Computing reward:")
                    logger.info(f"    - Reward function: {self.reward_function.name}")
                    logger.info(f"    - Calling compute() with merge_info and signals...")

                    reward = self.reward_function.compute(
                        merge_info=merge_info,
                        search_expansions=merge_info.nodes_expanded,
                        plan_cost=merge_info.solution_cost,
                        is_terminal=False
                    )

                    logger.info(f"    ✓ Reward computed: {reward:.4f}")

                    # Log component breakdown
                    components = self.reward_function.get_components_dict()
                    logger.info(f"\n  Reward component breakdown:")
                    for key, val in components.items():
                        logger.info(f"    - {key:<30} {val:.4f}")

            except Exception as e:
                logger.warning(f"  ⚠️ Reward computation failed: {e}")
                logger.warning(f"  Using reward = 0.0")
                reward = 0.0

            # ====================================================================
            # PHASE 8: CHECK EPISODE TERMINATION
            # ====================================================================

            logger.info(f"\n[PHASE 8] TERMINATION CHECK")

            num_remaining = self.graph_tracker.graph.number_of_nodes()
            logger.info(f"  Num remaining nodes: {num_remaining}")
            logger.info(f"  Current merge step: {self.current_merge_step} / {self.max_merges}")

            done = (num_remaining <= 1) or (self.current_merge_step >= self.max_merges - 1)
            logger.info(f"  Done: {done}")

            if done:
                if num_remaining <= 1:
                    logger.info(f"  Reason: Only {num_remaining} node(s) remaining")
                if self.current_merge_step >= self.max_merges - 1:
                    logger.info(f"  Reason: Max merges reached")

            # ====================================================================
            # PHASE 9: BUILD OBSERVATION
            # ====================================================================

            logger.info(f"\n[PHASE 9] BUILD OBSERVATION")

            try:
                obs = self._get_observation()
                logger.info(f"  ✓ Observation built successfully")
                logger.info(f"    - x shape: {obs['x'].shape}")
                logger.info(f"    - edge_index shape: {obs['edge_index'].shape}")
                logger.info(f"    - num_nodes: {obs['num_nodes']}")
                logger.info(f"    - num_edges: {obs['num_edges']}")
            except Exception as e:
                logger.error(f"  ❌ Failed to build observation: {e}")
                obs = self._get_observation()

            # ====================================================================
            # PHASE 10: INCREMENT STEP COUNTER
            # ====================================================================

            logger.info(f"\n[PHASE 10] STEP COUNTER INCREMENT")

            logger.info(f"  Before: current_merge_step = {self.current_merge_step}")
            self.current_merge_step += 1
            logger.info(f"  After: current_merge_step = {self.current_merge_step}")

            # ====================================================================
            # PHASE 11: BUILD INFO DICT & LOG
            # ====================================================================

            logger.info(f"\n[PHASE 11] BUILD INFO DICT & LOG STEP")

            info = {
                "merge_pair": [int(src), int(tgt)],
                "num_nodes": num_remaining,
                "step": self.current_merge_step - 1,
            }

            if merge_info is not None:
                from validate_merge_signals import validate_merge_signals
                is_valid, issues = validate_merge_signals(merge_info)
                if not is_valid:
                    logger.error(f"[SIGNAL VALIDATION] ✗ Merge signals invalid: {issues}")
                    merge_info = None

                info.update(merge_info.to_dict())
                logger.info(f"  ✓ Added merge_info to info dict")

            self._log_step(src, tgt, info, reward, done)
            logger.info(f"  ✓ Step logged")

            # ====================================================================
            # PHASE 12: RETURN
            # ====================================================================

            # In MergeEnv.step(), after computing reward:

            # ✅ ADD THIS BEFORE RETURNING
            self._save_gnn_decision_metadata(
                merge_step=self.current_merge_step,
                action=action,
                src=src,
                tgt=tgt,
                obs=obs,
                reward=reward,
                info=info
            )

            logger.info(f"\n[PHASE 12] RETURN RESULT")
            logger.info(f"  Returning:")
            logger.info(f"    - obs: shape {obs['x'].shape}")
            logger.info(f"    - reward: {reward:.4f}")
            logger.info(f"    - done: {done}")
            logger.info(f"    - truncated: False")
            logger.info(f"    - info: {len(info)} keys")
            logger.info("=" * 90 + "\n")

            return obs, float(reward), done, False, info

        except Exception as e:
            logger.error(f"\n❌ STEP FAILED: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.info("=" * 90 + "\n")
            return self.state, 0.0, True, False, {"error": str(e)}

    def _wait_for_json_file_safe(self, path: str, step: int, timeout_seconds: float = 60.0) -> Optional[Dict]:
        """
        ✅ ROBUST: Wait for JSON file with proper validation and diagnostics.

        NOW WITH PHASE TRACKING to identify exactly where handshake breaks down.

        Returns: Parsed JSON dict, or None if timeout/error occurs
        """
        start_time = time.time()
        last_size = -1
        last_modified = -1
        consecutive_parse_errors = 0
        max_consecutive_errors = 3

        # ✅ Track which phase of handshake we're in
        file_basename = os.path.basename(path)

        if "gnn_ack" in file_basename:
            phase_name = "ACK"
            phase_desc = "FD acknowledging merge decision"
        elif "ts_" in file_basename:
            phase_name = "TS"
            phase_desc = "Updated transition system"
        elif "merge_before" in file_basename:
            phase_name = "BEFORE"
            phase_desc = "Pre-merge metrics"
        elif "merge_after" in file_basename:
            phase_name = "AFTER"
            phase_desc = "Post-merge metrics"
        else:
            phase_name = "DATA"
            phase_desc = "Data file"

        logger.info(f"[Step {step}] [PHASE: {phase_name}] Starting wait...")
        logger.info(f"[Step {step}] [PHASE: {phase_name}] Waiting for: {path}")
        logger.info(f"[Step {step}] [PHASE: {phase_name}] Description: {phase_desc}")
        logger.info(f"[Step {step}] [PHASE: {phase_name}] Timeout: {timeout_seconds}s")

        while time.time() - start_time < timeout_seconds:
            elapsed = time.time() - start_time

            if not os.path.exists(path):
                if int(elapsed) > 0 and int(elapsed) % 15 == 0:  # Every 15 seconds
                    logger.debug(f"[Step {step}] [PHASE: {phase_name}] Still waiting... ({elapsed:.0f}s)")
                time.sleep(0.5)
                continue

            try:
                # ✅ Check file size and modification time
                current_size = os.path.getsize(path)
                current_mtime = os.path.getmtime(path)

                if current_size == 0:
                    logger.debug(f"[Step {step}] [PHASE: {phase_name}] File exists but EMPTY (0 bytes)")
                    consecutive_parse_errors += 1
                    if consecutive_parse_errors >= max_consecutive_errors:
                        logger.error(
                            f"[Step {step}] [PHASE: {phase_name}] File empty for {consecutive_parse_errors} checks - giving up")
                        return None
                    time.sleep(1.0)
                    continue

                # ✅ Wait for file to stabilize (size and mtime haven't changed)
                if current_size == last_size and current_mtime == last_modified:
                    logger.debug(f"[Step {step}] [PHASE: {phase_name}] File stable at {current_size} bytes, parsing...")

                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if not content.strip():
                            logger.warning(f"[Step {step}] [PHASE: {phase_name}] File content is empty/whitespace")
                            consecutive_parse_errors += 1
                            time.sleep(1.0)
                            continue

                        # ✅ Validate JSON structure
                        content_stripped = content.strip()
                        if not (content_stripped.startswith('{') or content_stripped.startswith('[')):
                            logger.error(
                                f"[Step {step}] [PHASE: {phase_name}] Invalid JSON: doesn't start with {{ or [")
                            logger.error(f"[Step {step}] [PHASE: {phase_name}] First 100 chars: {content[:100]}")
                            consecutive_parse_errors += 1
                            time.sleep(1.0)
                            continue

                        # ✅ Try to parse
                        data = json.loads(content)

                        logger.info(
                            f"[Step {step}] [PHASE: {phase_name}] ✅ Successfully parsed JSON ({current_size} bytes)")
                        logger.info(f"[Step {step}] [PHASE: {phase_name}] Elapsed time: {elapsed:.1f}s")
                        consecutive_parse_errors = 0  # Reset on success
                        return data

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"[Step {step}] [PHASE: {phase_name}] JSON parse error (line {e.lineno}, col {e.colno}): {e.msg}")
                        logger.debug(f"[Step {step}] [PHASE: {phase_name}] Content preview: {content[:200]}")
                        consecutive_parse_errors += 1

                        if consecutive_parse_errors >= max_consecutive_errors:
                            logger.error(
                                f"[Step {step}] [PHASE: {phase_name}] JSON parsing failed {max_consecutive_errors} times - giving up")
                            return None

                        time.sleep(2.0)
                        continue

                else:
                    # File size or mtime changed - still writing
                    logger.debug(
                        f"[Step {step}] [PHASE: {phase_name}] File size changing: {last_size} → {current_size} bytes (still writing)")
                    last_size = current_size
                    last_modified = current_mtime
                    consecutive_parse_errors = 0
                    time.sleep(0.5)
                    continue

            except (OSError, IOError) as e:
                logger.debug(f"[Step {step}] [PHASE: {phase_name}] File I/O error: {e}")
                consecutive_parse_errors += 1
                time.sleep(1.0)
                continue

            except Exception as e:
                logger.error(f"[Step {step}] [PHASE: {phase_name}] Unexpected error: {e}", exc_info=True)
                return None

            # Check if FD died
            if self.process and self.process.poll() is not None:
                rc = self.process.returncode
                logger.error(f"[Step {step}] [PHASE: {phase_name}] ❌ FD process DIED with code {rc}")
                logger.error(
                    f"[Step {step}] [PHASE: {phase_name}] FD crashed before producing {os.path.basename(path)}")
                return None

        # TIMEOUT
        logger.error(
            f"[Step {step}] [PHASE: {phase_name}] ❌ TIMEOUT after {timeout_seconds}s waiting for {os.path.basename(path)}")
        logger.error(f"[Step {step}] [PHASE: {phase_name}] This indicates:")
        logger.error(f"[Step {step}] [PHASE: {phase_name}]   1. FD process crashed or hung")
        logger.error(f"[Step {step}] [PHASE: {phase_name}]   2. JSON file not being written by FD")
        logger.error(f"[Step {step}] [PHASE: {phase_name}]   3. File I/O or permission problem")

        # Diagnostic: list what files DO exist
        fd_output_dir = "downward/fd_output"
        if os.path.exists(fd_output_dir):
            logger.error(f"[Step {step}] [PHASE: {phase_name}] Files in fd_output/:")
            try:
                for fname in sorted(os.listdir(fd_output_dir)):
                    fpath = os.path.join(fd_output_dir, fname)
                    if os.path.isfile(fpath):
                        size = os.path.getsize(fpath)
                        logger.error(f"[Step {step}] [PHASE: {phase_name}]   - {fname} ({size} bytes)")
            except:
                pass

        return None

    def _diagnose_fd_output(self, step: int) -> None:
        """✅ NEW: Detailed diagnostics when FD output fails."""
        fd_output_dir = "downward/fd_output"

        logger.info(f"\n[DIAGNOSIS] Checking FD output state for step {step}...")

        # Check directory
        if not os.path.exists(fd_output_dir):
            logger.error(f"[DIAGNOSIS] fd_output directory DOES NOT EXIST")
            return

        logger.info(f"[DIAGNOSIS] Files in {fd_output_dir}:")
        try:
            for fname in sorted(os.listdir(fd_output_dir)):
                fpath = os.path.join(fd_output_dir, fname)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    mtime = os.path.getmtime(fpath)
                    age_sec = time.time() - mtime
                    logger.info(f"  - {fname:<30} {size:>10} bytes (age: {age_sec:.1f}s)")
        except Exception as e:
            logger.error(f"[DIAGNOSIS] Error listing files: {e}")

        # Check for expected files
        expected_ts = os.path.join(fd_output_dir, f"ts_{step}.json")
        expected_before = os.path.join(fd_output_dir, f"merge_before_{step}.json")
        expected_after = os.path.join(fd_output_dir, f"merge_after_{step}.json")

        logger.info(f"[DIAGNOSIS] Expected files for step {step}:")
        for expected_path in [expected_ts, expected_before, expected_after]:
            exists = os.path.exists(expected_path)
            status = "✓ EXISTS" if exists else "✗ MISSING"
            logger.info(f"  {status}: {os.path.basename(expected_path)}")

            if exists:
                try:
                    size = os.path.getsize(expected_path)
                    with open(expected_path, 'r') as f:
                        content = f.read()
                    logger.info(f"    Size: {size} bytes")
                    logger.info(f"    Preview: {content[:100]}...")
                except Exception as e:
                    logger.warning(f"    Error reading: {e}")

        # Check FD process
        if self.process:
            retcode = self.process.poll()
            if retcode is not None:
                logger.error(f"[DIAGNOSIS] FD process has EXITED with code {retcode}")
            else:
                logger.info(f"[DIAGNOSIS] FD process is RUNNING (PID: {self.process.pid})")

    def _diagnose_signals(self, iteration: int):
        """✅ NEW: Verify signal correctness."""
        before_path = os.path.join("downward", "fd_output", f"merge_before_{iteration}.json")
        after_path = os.path.join("downward", "fd_output", f"merge_after_{iteration}.json")

        if not os.path.exists(before_path) or not os.path.exists(after_path):
            logger.error(f"Signal files missing for iteration {iteration}")
            logger.error(f"  Before path: {before_path} (exists: {os.path.exists(before_path)})")
            logger.error(f"  After path: {after_path} (exists: {os.path.exists(after_path)})")
            return False

        try:
            with open(before_path) as f:
                before = json.load(f)
            with open(after_path) as f:
                after = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load signal files: {e}")
            return False

        # ✅ VERIFICATION 1: State count consistency
        ts1_size = before.get("ts1_size", 0)
        ts2_size = before.get("ts2_size", 0)
        expected_merged_size = ts1_size * ts2_size
        actual_merged_size = after.get("num_states", 0)

        logger.info(f"[VERIFY] Iteration {iteration}:")
        logger.info(f"  TS1 size: {ts1_size}, TS2 size: {ts2_size}")
        logger.info(f"  Expected merged size: {expected_merged_size}")
        logger.info(f"  Actual merged size: {actual_merged_size}")

        if actual_merged_size > expected_merged_size * 1.1:
            logger.warning(f"  ⚠️ Merged size larger than expected (not properly shrunk?)")

        # ✅ VERIFICATION 2: F-value list lengths
        f1_len = len(before.get("ts1_f_values", []))
        f2_len = len(before.get("ts2_f_values", []))
        f_after_len = len(after.get("f_values", []))

        logger.info(f"  F-values: |f1|={f1_len}, |f2|={f2_len}, |f_after|={f_after_len}")

        if f1_len != ts1_size or f2_len != ts2_size:
            logger.error(f"  ❌ F-value list size mismatch!")
            return False

        # ✅ VERIFICATION 3: A* signals present and valid
        signals = after.get("search_signals", {})
        logger.info(f"  A* signals: {signals}")

        if not signals:
            logger.warning(f"  ⚠️ No A* signals exported")
        else:
            for key in ["nodes_expanded", "branching_factor", "solution_found"]:
                if key not in signals:
                    logger.error(f"  ❌ Missing signal: {key}")
                    return False

        return True
    def _parse_fd_log(self) -> Tuple[int, int]:
        """Read FD log and extract plan cost and expansions."""
        if self.fd_log_file:
            try:
                self.fd_log_file.flush()
            except Exception:
                pass

        log_path = os.path.join("downward", "fd_output", "log.txt")
        plan_cost, expansions = 0, 0
        if os.path.exists(log_path):
            try:
                text = open(log_path, "r", encoding="utf-8", errors="ignore").read()
            except Exception:
                text = ""
            if text:
                m1 = list(re.finditer(r"Plan length:\s*(\d+)", text))
                if m1:
                    plan_cost = int(m1[-1].group(1))
                m2 = list(re.finditer(r"[Ee]xpanded\s+(\d+)\s+state(?:s)?(?:\(\w*\))?", text))
                if m2:
                    expansions = int(m2[-1].group(1))
        return plan_cost, expansions

    # ============================================================================
    # ✅ NEW: Extract rich edge features for GNN learning
    # ============================================================================

    def _extract_edge_features(self) -> np.ndarray:
        """
        ✅ NEW: Extract rich features about merge candidates.

        For each edge (u, v), compute:
        1. Relative size difference
        2. Expected merge size ratio
        3. Shared variable count
        4. Reachability metrics
        5. Iteration difference
        6. Centrality similarity
        7. F-value consistency
        8. Merge risk indicator

        Returns:
            [E, 8] edge features
        """
        G = self.graph_tracker.graph
        edges = list(G.edges())

        if not edges:
            return np.zeros((0, 8), dtype=np.float32)

        edge_features = []

        for u, v in edges:
            u_data = G.nodes[u]
            v_data = G.nodes[v]

            # Feature 1: Relative size difference (normalized)
            u_size = u_data.get("num_states", 1)
            v_size = v_data.get("num_states", 1)
            max_size_in_graph = max(
                d.get("num_states", 1) for _, d in G.nodes(data=True)
            )
            size_diff = float(abs(u_size - v_size)) / max(max_size_in_graph, 1)

            # Feature 2: Expected merged size ratio (% of reachable states)
            # Heuristic: product of sizes as fraction of typical max
            product_size = (u_size * v_size) / max(max_size_in_graph * max_size_in_graph, 1)
            merge_size_ratio = np.clip(product_size, 0.0, 1.0)

            # Feature 3: Shared variables (normalized)
            u_vars = set(u_data.get("incorporated_variables", []))
            v_vars = set(v_data.get("incorporated_variables", []))
            shared_vars = len(u_vars & v_vars)
            total_vars = len(u_vars | v_vars)
            shared_ratio = shared_vars / max(total_vars, 1)

            # Feature 4: Reachability consistency
            u_f = u_data.get("f_before", [])
            v_f = v_data.get("f_before", [])
            u_reachable = sum(1 for f in u_f if f != float('inf') and f < 1_000_000_000) / max(len(u_f), 1)
            v_reachable = sum(1 for f in v_f if f != float('inf') and f < 1_000_000_000) / max(len(v_f), 1)
            reachability_similarity = 1.0 - abs(u_reachable - v_reachable)

            # Feature 5: Iteration difference (normalized)
            u_iter = u_data.get("iteration", 0)
            v_iter = v_data.get("iteration", 0)
            max_iter = max((d.get("iteration", 0) for _, d in G.nodes(data=True)), default=1)
            iter_diff = float(abs(u_iter - v_iter)) / max(max_iter, 1)

            # Feature 6: Centrality similarity
            u_centrality = self.centrality.get(u, 0.0)
            v_centrality = self.centrality.get(v, 0.0)
            centrality_similarity = 1.0 - abs(u_centrality - v_centrality)

            # Feature 7: F-value consistency (std of combined distributions)
            f_combined = u_f + v_f
            if f_combined:
                # Filter to valid values
                f_valid = [f for f in f_combined if f != float('inf') and f < 1_000_000_000]
                if f_valid:
                    f_std = float(np.std(f_valid)) / (1.0 + float(np.mean(f_valid)))
                    f_consistency = np.clip(1.0 - f_std, 0.0, 1.0)
                else:
                    f_consistency = 0.0
            else:
                f_consistency = 0.5

            # Feature 8: Merge risk indicator
            # Combines: degree, transition density, reachability
            u_degree = G.degree(u) / max(G.number_of_nodes(), 1)
            v_degree = G.degree(v) / max(G.number_of_nodes(), 1)
            u_trans = u_data.get("num_transitions", 0) / max(u_size, 1)
            v_trans = v_data.get("num_transitions", 0) / max(v_size, 1)

            merge_risk = np.clip(
                (u_degree + v_degree) * 0.5 + (u_trans + v_trans) * 0.25,
                0.0, 1.0
            )

            edge_features.append([
                size_diff,
                merge_size_ratio,
                shared_ratio,
                reachability_similarity,
                iter_diff,
                centrality_similarity,
                f_consistency,
                merge_risk
            ])

        return np.array(edge_features, dtype=np.float32)

    def _get_observation(self) -> Dict:
        """✅ ULTRA-OPTIMIZED: Persistent array pre-allocation + vectorization."""
        max_nodes, max_edges = 100, 1000

        G = self.graph_tracker.graph

        # ✅ OPTIMIZATION 1: Pre-allocate observation arrays ONCE (not per call)
        if not hasattr(self, '_obs_cache'):
            self._obs_cache = {
                'x': np.zeros((max_nodes, self.feat_dim), dtype=np.float32),
                'edge_index': np.zeros((2, max_edges), dtype=np.int64),
                'edge_features': np.zeros((max_edges, 8), dtype=np.float32),
            }
            self._last_graph_hash = None

        # ✅ OPTIMIZATION 2: Quick hash check - skip rebuild if graph unchanged
        current_hash = self.graph_tracker._get_graph_hash()
        if current_hash == self._last_graph_hash and self._last_graph_hash is not None:
            # Graph hasn't changed - return cached observation
            return {
                "x": self._obs_cache['x'].copy(),
                "edge_index": self._obs_cache['edge_index'].copy(),
                "edge_features": self._obs_cache['edge_features'].copy(),
                "num_nodes": np.int32(self._last_num_nodes),
                "num_edges": np.int32(self._last_num_edges),
            }
        self._last_graph_hash = current_hash

        # ✅ REUSE arrays (clear instead of reallocate)
        x = self._obs_cache['x']
        x.fill(0)

        ei = self._obs_cache['edge_index']
        ei.fill(0)

        ef = self._obs_cache['edge_features']
        ef.fill(0)

        # ✅ OPTIMIZATION 3: Get cached metrics (don't recompute)
        degs = dict(G.degree()).values()
        max_deg = max(max(degs, default=0), 1)
        max_states_node = max((d.get("num_states", 0) for _, d in G.nodes(data=True)), default=1) or 1

        # ✅ OPTIMIZATION 4: Compute F-scale ONCE (vectorized)
        f_values_all = []
        for _, d in G.nodes(data=True):
            f_vals = d.get("f_before", [])
            if f_vals:
                f_values_all.extend(f_vals)

        f_scale = max(f_values_all) if f_values_all else 1.0
        f_scale = max(f_scale, 1.0)

        # ✅ OPTIMIZATION 5: Use cached centrality (not recomputed every step)
        centrality = self.graph_tracker.get_centrality()
        max_vars = self.graph_tracker.get_max_vars()
        max_iter = self.graph_tracker.get_max_iter()

        # ✅ OPTIMIZATION 6: Vectorized node feature computation
        node_features_list = []
        idx = {}

        for i, (nid, data) in enumerate(G.nodes(data=True)):
            if i >= max_nodes:
                break

            # Batch compute all features for this node at once
            ns_raw = float(data.get("num_states", 0))
            num_states = ns_raw / float(max_states_node)
            is_atomic = 1.0 if data.get("iteration", -1) == -1 else 0.0
            d_norm = G.degree(nid) / max_deg
            od_norm = G.out_degree(nid) / max_deg

            # ✅ CACHED F-stats (memoized in graph_tracker)
            f_min_norm, f_mean_norm, f_max_norm, f_std_norm = self.graph_tracker.f_stats(nid)

            if f_scale > 0:
                f_min_norm = max(0.0, min(f_min_norm / f_scale, 1.0))
                f_mean_norm = max(0.0, min(f_mean_norm / f_scale, 1.0))
                f_max_norm = max(0.0, min(f_max_norm / f_scale, 1.0))
                f_std_norm = max(0.0, min(f_std_norm / f_scale, 1.0))
            else:
                f_min_norm = f_mean_norm = f_max_norm = f_std_norm = 0.0

            # Heuristic quality features
            f_vals = np.array(data.get("f_before", []), dtype=np.float32)
            if len(f_vals) > 0 and f_scale > 0:
                valid_f = f_vals[(f_vals != np.inf) & (f_vals >= 0) & (f_vals < 1e9)]
                if len(valid_f) > 0:
                    avg_f_norm = float(np.mean(valid_f)) / f_scale
                    max_f_norm_heur = float(np.max(valid_f)) / f_scale
                    f_median = np.median(valid_f)
                    heuristic_concentration = float(np.std(valid_f)) / (1.0 + f_median)
                    heuristic_concentration = float(np.clip(heuristic_concentration, 0.0, 1.0))
                else:
                    avg_f_norm = 0.0
                    max_f_norm_heur = 0.0
                    heuristic_concentration = 0.0
            else:
                avg_f_norm = 0.0
                max_f_norm_heur = 0.0
                heuristic_concentration = 0.0

            reachable_ratio = 1.0
            if len(f_vals) > 0:
                unreachable = np.sum((f_vals == np.inf) | (f_vals >= 1e9))
                reachable_ratio = float(1.0 - unreachable / len(f_vals))

            num_vars_norm = len(data.get("incorporated_variables", [])) / float(max_vars)
            iter_idx_norm = data.get("iteration", 0) / float(max_iter)
            centrality_norm = float(centrality.get(nid, 0.0))

            num_neighbors = len(list(G.neighbors(nid)))
            neighbor_risk = float(num_neighbors) / max(len(G.nodes), 1)

            # ✅ SINGLE ASSIGNMENT (faster than individual assignments)
            x[i, :] = [
                num_states, is_atomic, d_norm, od_norm,
                avg_f_norm, max_f_norm_heur, heuristic_concentration, reachable_ratio,
                num_vars_norm, iter_idx_norm, centrality_norm,
                f_min_norm, f_mean_norm, f_max_norm, f_std_norm,
                neighbor_risk, np.clip(ns_raw / 10000.0, 0.0, 1.0),
                0.0, 0.0
            ]

            idx[nid] = i

        num_nodes_feat = len(idx)

        # ✅ OPTIMIZATION 7: Vectorized edge processing
        edges = [
            (idx[u], idx[v])
            for u, v in G.edges()
            if u in idx and v in idx
        ]
        ne = len(edges)

        for j, (u, v) in enumerate(edges[:max_edges]):
            ei[0, j] = u
            ei[1, j] = v

        # ✅ OPTIMIZATION 8: Extract edge features (only if needed)
        if ne > 0:
            edge_features = self._extract_edge_features_cached()
            ef[:ne, :] = edge_features[:ne]

        # Store for next check
        self._last_num_nodes = num_nodes_feat
        self._last_num_edges = ne

        return {
            "x": x,
            "edge_index": ei,
            "edge_features": ef,
            "num_nodes": np.int32(num_nodes_feat),
            "num_edges": np.int32(ne),
        }

    def _extract_edge_features_cached(self) -> np.ndarray:
        """✅ NEW: Cached edge feature extraction."""
        G = self.graph_tracker.graph
        edges = list(G.edges())

        if not edges:
            return np.zeros((0, 8), dtype=np.float32)

        # ✅ Quick check: if graph hasn't changed, return cached
        current_hash = self.graph_tracker._get_graph_hash()
        if (hasattr(self, '_edge_features_cache_hash') and
                self._edge_features_cache_hash == current_hash and
                hasattr(self, '_edge_features_cache')):
            return self._edge_features_cache

        # Compute edge features (this is the expensive part)
        edge_features = []

        # Get graph-level metrics ONCE (not per edge)
        max_size_in_graph = max(
            d.get("num_states", 1) for _, d in G.nodes(data=True)
        ) or 1
        max_iter_global = max((d.get("iteration", 0) for _, d in G.nodes(data=True)), default=1) or 1

        for u, v in edges:
            u_data = G.nodes[u]
            v_data = G.nodes[v]

            u_size = u_data.get("num_states", 1)
            v_size = v_data.get("num_states", 1)
            size_diff = float(abs(u_size - v_size)) / max(max_size_in_graph, 1)

            product_size = (u_size * v_size) / max(max_size_in_graph * max_size_in_graph, 1)
            merge_size_ratio = np.clip(product_size, 0.0, 1.0)

            u_vars = set(u_data.get("incorporated_variables", []))
            v_vars = set(v_data.get("incorporated_variables", []))
            shared_vars = len(u_vars & v_vars)
            total_vars = len(u_vars | v_vars)
            shared_ratio = shared_vars / max(total_vars, 1)

            u_f = u_data.get("f_before", [])
            v_f = v_data.get("f_before", [])
            u_reachable = sum(1 for f in u_f if f != float('inf') and f < 1_000_000_000) / max(len(u_f), 1)
            v_reachable = sum(1 for f in v_f if f != float('inf') and f < 1_000_000_000) / max(len(v_f), 1)
            reachability_similarity = 1.0 - abs(u_reachable - v_reachable)

            u_iter = u_data.get("iteration", 0)
            v_iter = v_data.get("iteration", 0)
            iter_diff = float(abs(u_iter - v_iter)) / max(max_iter_global, 1)

            # Use CACHED centrality
            centrality = self.graph_tracker.get_centrality()
            u_centrality = centrality.get(u, 0.0)
            v_centrality = centrality.get(v, 0.0)
            centrality_similarity = 1.0 - abs(u_centrality - v_centrality)

            f_combined = u_f + v_f
            if f_combined:
                f_valid = [f for f in f_combined if f != float('inf') and f < 1_000_000_000]
                if f_valid:
                    f_std = float(np.std(f_valid)) / (1.0 + float(np.mean(f_valid)))
                    f_consistency = np.clip(1.0 - f_std, 0.0, 1.0)
                else:
                    f_consistency = 0.0
            else:
                f_consistency = 0.5

            u_degree = G.degree(u) / max(G.number_of_nodes(), 1)
            v_degree = G.degree(v) / max(G.number_of_nodes(), 1)
            u_trans = u_data.get("num_transitions", 0) / max(u_size, 1)
            v_trans = v_data.get("num_transitions", 0) / max(v_size, 1)

            merge_risk = np.clip(
                (u_degree + v_degree) * 0.5 + (u_trans + v_trans) * 0.25,
                0.0, 1.0
            )

            edge_features.append([
                size_diff,
                merge_size_ratio,
                shared_ratio,
                reachability_similarity,
                iter_diff,
                centrality_similarity,
                f_consistency,
                merge_risk
            ])

        result = np.array(edge_features, dtype=np.float32)

        # Cache the result
        self._edge_features_cache = result
        self._edge_features_cache_hash = current_hash

        return result

    def _save_gnn_decision_metadata(self, merge_step: int, action: int,
                                    src: int, tgt: int, obs: Dict,
                                    reward: float, info: Dict) -> None:
        """✅ NEW: Save GNN decision metadata for analysis."""

        try:
            import json
            from datetime import datetime

            metadata = {
                'episode_step': self.current_merge_step,
                'merge_step': merge_step,
                'action_index': int(action),
                'chosen_edge': [int(src), int(tgt)],
                'observation_shape': {
                    'num_nodes': int(obs.get('num_nodes', 0)),
                    'num_edges': int(obs.get('num_edges', 0)),
                    'node_features_dim': int(obs['x'].shape[-1]) if obs['x'].ndim > 1 else 0,
                },
                'reward_received': float(reward),
                'merge_info': {
                    'plan_cost': info.get('plan_cost', 0),
                    'num_expansions': info.get('num_expansions', 0),
                    'delta_states': info.get('delta_states', 0),
                },
                'timestamp': datetime.now().isoformat(),
                'problem': os.path.basename(self.problem_file),
            }

            self.gnn_decisions_log.append(metadata)

        except Exception as e:
            logger.debug(f"Could not save GNN metadata: {e}")

    def _export_episode_metadata(self) -> None:
        """✅ NEW: Export all GNN decisions from this episode."""

        if not self.gnn_decisions_log:
            return

        try:
            import json
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_metadata = {
                'problem': os.path.basename(self.problem_file),
                'num_decisions': len(self.gnn_decisions_log),
                'decisions': self.gnn_decisions_log,
                'export_timestamp': datetime.now().isoformat(),
            }

            metadata_file = os.path.join(
                self.gnn_metadata_dir,
                f"episode_{timestamp}_{len(self.gnn_decisions_log)}_decisions.json"
            )

            with open(metadata_file, 'w') as f:
                json.dump(episode_metadata, f, indent=2, default=str)

            logger.info(f"✓ Exported GNN episode metadata: {metadata_file}")

        except Exception as e:
            logger.warning(f"Failed to export episode metadata: {e}")

    def _count_total_states(self) -> int:
        return sum(d["num_states"] for _, d in self.graph_tracker.graph.nodes(data=True))

    def close(self):
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        except Exception:
            pass
        finally:
            self.process = None

        try:
            if self.fd_log_file:
                self.fd_log_file.flush()
                self.fd_log_file.close()
        except Exception:
            pass
        finally:
            self.fd_log_file = None