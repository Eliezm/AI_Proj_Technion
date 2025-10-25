# -*- coding: utf-8 -*-
"""
This module provides the GraphTracker class, a data structure for managing the
state of the merge-and-shrink heuristic construction process.

It represents the set of transition systems (TS) as nodes in a directed graph,
where edges represent causal dependencies from the Fast Downward planner. The class
is responsible for loading the initial state from planner output, performing
merge operations on nodes, and updating the graph based on new information from
the planner. It serves as the core state management component for the MergeEnv.
"""

# ------------------------------------------------------------------------------
#  Imports
# ------------------------------------------------------------------------------
import json
import logging
import time
from json import JSONDecoder
from typing import List, Union, Dict, Tuple, Any, FrozenSet, Optional

import networkx as nx
import numpy as np

# matplotlib is an optional dependency for visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
#  Configuration and Constants
# ------------------------------------------------------------------------------
# --- Setup basic logging ---
# Consistent logging configuration with merge_env.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# --- Constants for file I/O ---
FILE_RETRY_COUNT = 60
FILE_RETRY_DELAY_S = 0.2


# ------------------------------------------------------------------------------
#  Helper Functions
# ------------------------------------------------------------------------------

def _load_json_robustly(path: str, retries: int = FILE_RETRY_COUNT, delay: float = FILE_RETRY_DELAY_S) -> Any:
    """
    Parses the first complete JSON object from a file path with retries.

    This function is designed to handle cases where a file might be read while
    another process is writing to it. It ensures the file is not empty and that
    the content appears to be a complete JSON object (ends with '}' or ']')
    before attempting to parse it.

    Method of Action:
    1. Loop for a specified number of `retries`.
    2. Read the file content, ignoring UTF-8 errors.
    3. If the file is empty or doesn't end with a closing brace/bracket,
       it's considered incomplete. Wait and retry.
    4. Use `JSONDecoder.raw_decode` to parse only the *first* valid JSON
       object, which avoids errors from trailing, partially-written data.
    5. If any error occurs (`OSError`, `JSONDecodeError`), wait and retry.
    6. If all retries fail, raise a `RuntimeError` with the last known error.
    """
    decoder = JSONDecoder()
    last_error = None
    for _ in range(retries):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().lstrip()

            if not content:
                # File is empty, wait for content to be written.
                raise json.JSONDecodeError("File is empty", content, 0)

            # Heuristic check for completeness to avoid parsing mid-write.
            tail = content.rstrip()
            if not tail or tail[-1] not in ("]", "}"):
                raise json.JSONDecodeError("JSON appears incomplete (no closing bracket/brace)", content, len(content))

            # Decode the first object, ignoring any trailing garbage.
            obj, _ = decoder.raw_decode(content)
            return obj

        except (OSError, json.JSONDecodeError) as e:
            last_error = e
            time.sleep(delay)

    raise RuntimeError(f"Failed to load valid JSON from '{path}' after {retries} retries. Last error: {last_error}")


def product_state_index(s1: int, s2: int, n2: int) -> int:
    """
    Maps a pair of local states to a single index in their Cartesian product.

    This is a standard row-major order mapping. Given two state spaces of sizes
    `n1` and `n2`, a state `s1` from the first space and `s2` from the second
    are mapped to a unique index in the combined space of size `n1 * n2`.

    Args:
        s1 (int): The index of the state in the first transition system.
        s2 (int): The index of the state in the second transition system.
        n2 (int): The total number of states in the second transition system.

    Returns:
        int: The unique index in the product state space.
    """
    return s1 * n2 + s2


def merge_transition_systems(ts1: Dict[str, Any], ts2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new transition system (TS) via Cartesian product of two TSs.

    This is the core "merge" operation in the merge-and-shrink algorithm. The
    resulting TS has a state space that is the product of the two input state
    spaces.

    Method of Action:
    1.  The number of states in the new TS is `n1 * n2`.
    2.  The new initial state is the product of the individual initial states.
    3.  The new goal states are the product of the individual goal state sets.
    4.  The list of "incorporated variables" is the concatenation of the inputs.
    5.  The "iteration" number is incremented to mark this as a composite,
        non-atomic system.

    Args:
        ts1 (Dict): The first transition system dictionary.
        ts2 (Dict): The second transition system dictionary.

    Returns:
        Dict: The new, merged transition system dictionary.
    """
    n1, n2 = ts1["num_states"], ts2["num_states"]

    merged_ts = {
        "num_states": n1 * n2,
        "init_state": product_state_index(ts1["init_state"], ts2["init_state"], n2),
        "goal_states": [
            product_state_index(g1, g2, n2)
            for g1 in ts1["goal_states"]
            for g2 in ts2["goal_states"]
        ],
        "incorporated_variables": ts1["incorporated_variables"] + ts2["incorporated_variables"],
        # Incrementing the iteration level signifies a merge operation.
        # Atomic systems have iteration = -1.
        "iteration": max(ts1.get("iteration", -1), ts2.get("iteration", -1)) + 1,
    }
    return merged_ts


# ------------------------------------------------------------------------------
#  GraphTracker Class
# ------------------------------------------------------------------------------

class GraphTracker:
    """
    Manages the graph of transition systems for the merge-and-shrink process.

    This class holds a `networkx.DiGraph` where each node represents a transition
    system (TS). Initially, nodes correspond to atomic TSs for individual problem
    variables. The class provides methods to merge nodes (creating a new composite
    TS node) and update node properties based on new data from the planner.

    Attributes:
        graph (nx.DiGraph): The graph of transition systems.
        varset_to_node (Dict): A mapping from a frozenset of variable IDs to the
                                corresponding node ID in the graph. This allows for
                                efficient lookups.
        next_node_id (int): A counter for allocating unique IDs to new merged nodes.
    """

    # def __init__(self, ts_json_path: str, cg_json_path: str, is_debug: bool = False):
    #     """
    #     Initializes the GraphTracker by loading the initial graph structure.
    #
    #     Args:
    #         ts_json_path (str): Path to the JSON file containing the list of
    #                             initial transition systems.
    #         cg_json_path (str): Path to the JSON file defining the causal graph
    #                             edges between variables.
    #         is_debug (bool): If True, runs in debug mode which may alter behavior,
    #                          e.g., by creating a dummy graph if files are missing.
    #     """
    #     self.graph = nx.DiGraph()
    #     self.varset_to_node: Dict[FrozenSet, Union[int, str]] = {}
    #     self.next_node_id: int = 0
    #     self.is_debug = is_debug
    #
    #     # ✅ ADD: Caching for expensive computations
    #     self._centrality_cache: Optional[Dict] = None
    #     self._centrality_cache_valid = False
    #     self._max_vars_cache: Optional[int] = None
    #     self._max_iter_cache: Optional[int] = None
    #     self._graph_hash_last = None  # Track if graph changed
    #
    #     logging.info("Initializing GraphTracker...")
    #     try:
    #         self._load_atomic_systems(ts_json_path)
    #         self._load_causal_edges(cg_json_path)
    #     except Exception as e:
    #         logging.error(f"Failed during initial graph loading: {e}")
    #         if not self.is_debug:
    #             # In non-debug mode, this is a fatal error.
    #             raise
    #         else:
    #             # In debug mode, we can proceed with an empty graph.
    #             logging.warning("Proceeding with an empty graph in debug mode.")

    # --- REPLACE THE EXISTING __init__ METHOD WITH THIS ---
    def __init__(self, ts_json_path: str, cg_json_path: str, is_debug: bool = False):
        """Initialize with caching infrastructure."""
        self.graph = nx.DiGraph()
        self.varset_to_node: Dict[FrozenSet, Union[int, str]] = {}
        self.next_node_id: int = 0
        self.is_debug = is_debug
        # ✅ NEW: Persistent caches (survive across observations)
        self._centrality_cache: Optional[Dict] = None
        self._centrality_cache_valid = False
        self._max_vars_cache: Optional[int] = None
        self._max_iter_cache: Optional[int] = None
        self._f_stats_cache: Dict[int, Tuple[float, float, float, float]] = {}  # Cache for f_stats
        self._graph_hash_last = None
        # Note: _edge_features_cache and _node_features_cache were mentioned
        # in the prompt but not used in the provided methods, so omitting them for now.
        # Add them here if needed later:
        # self._edge_features_cache: Optional[np.ndarray] = None
        # self._node_features_cache: Dict[int, np.ndarray] = {}

        logging.info("Initializing GraphTracker...")
        try:
            self._load_atomic_systems(ts_json_path)
            self._load_causal_edges(cg_json_path)
        except Exception as e:
            logging.error(f"Failed during initial graph loading: {e}")
            if not self.is_debug:
                raise
            else:
                logging.warning("Proceeding with an empty graph in debug mode.")

    # --- END OF REPLACEMENT FOR __init__ ---

    # --- ADD THESE NEW METHODS INSIDE THE GraphTracker CLASS ---

    def _get_graph_hash(self) -> str:
        """✅ Quick hash to detect graph changes."""
        # Hash based on edges (nodes are implicit)
        # Using sorted edges ensures hash consistency regardless of internal order
        edges_tuple = tuple(sorted(self.graph.edges()))
        return str(hash(edges_tuple))

    def _invalidate_caches(self):
        """Call ONLY after actual graph modification."""
        logging.debug("Invalidating GraphTracker caches...")  # Added log
        self._centrality_cache_valid = False
        self._f_stats_cache.clear()
        # Invalidate others if added later
        # self._edge_features_cache = None
        # self._node_features_cache.clear()
        self._graph_hash_last = None  # Reset graph hash tracking

    def get_centrality(self, force_recompute: bool = False) -> Dict:
        """✅ CACHED: Return cached centrality or compute once."""
        if force_recompute:
            self._centrality_cache_valid = False

        # Only recompute if cache is marked invalid
        if not self._centrality_cache_valid:
            logging.debug("Computing centrality (cached)...")
            try:
                # Handle potentially empty or disconnected graphs
                if self.graph.number_of_nodes() > 0:
                    self._centrality_cache = nx.closeness_centrality(self.graph)
                else:
                    self._centrality_cache = {}
            except nx.NetworkXError:  # Handles disconnected graph case
                logging.warning(
                    "Graph is disconnected, centrality might be misleading. Computing per component (using closeness_centrality).")
                self._centrality_cache = nx.closeness_centrality(
                    self.graph)  # Or handle components separately if needed
            self._centrality_cache_valid = True  # Mark as valid even if empty/disconnected

        return self._centrality_cache if self._centrality_cache is not None else {}

    def get_max_vars(self) -> int:
        """✅ CACHED: Return max incorporated variables."""
        # Compute only if cache is empty
        if self._max_vars_cache is None:
            logging.debug("Computing max_vars (cached)...")
            self._max_vars_cache = max(
                (len(d.get("incorporated_variables", [])) for _, d in self.graph.nodes(data=True)),
                default=1  # Default if graph is empty
            ) or 1  # Ensure it's at least 1 if max returns 0
        return self._max_vars_cache

    def get_max_iter(self) -> int:
        """✅ CACHED: Return max iteration level."""
        # Compute only if cache is empty
        if self._max_iter_cache is None:
            logging.debug("Computing max_iter (cached)...")
            self._max_iter_cache = max(
                (d.get("iteration", 0) for _, d in self.graph.nodes(data=True)),
                default=0  # Default if graph is empty
            ) or 1  # Ensure it's at least 1 if max returns 0
        return self._max_iter_cache

    # --- END OF NEW METHODS TO ADD ---

    # def _invalidate_caches(self):
    #     """Call after any graph modification."""
    #     self._centrality_cache_valid = False
    #     self._graph_hash_last = None
    #
    # def get_centrality(self) -> Dict:
    #     """Return cached centrality or compute once."""
    #     if not self._centrality_cache_valid:
    #         self._centrality_cache = nx.closeness_centrality(self.graph)
    #         self._centrality_cache_valid = True
    #     return self._centrality_cache
    #
    # def get_max_vars(self) -> int:
    #     """Return cached max_vars."""
    #     if self._max_vars_cache is None:
    #         self._max_vars_cache = max(
    #             (len(d.get("incorporated_variables", [])) for _, d in self.graph.nodes(data=True)),
    #             default=1
    #         ) or 1
    #     return self._max_vars_cache

    # def get_max_iter(self) -> int:
    #     """Return cached max_iter."""
    #     if self._max_iter_cache is None:
    #         self._max_iter_cache = max(
    #             (d.get("iteration", 0) for _, d in self.graph.nodes(data=True)),
    #             default=0
    #         ) or 1
    #     return self._max_iter_cache

    def update_graph(self, ts_json_path: str) -> None:
        """
        Updates the graph with new transition system data from a JSON file.

        This method reads a TS list from the given path and updates the properties
        of existing nodes. This is typically used after a merge operation, where
        the planner provides an updated TS file for the newly created node.

        Args:
            ts_json_path (str): The path to the JSON file with TS data.
        """
        logging.info(f"Updating graph from '{ts_json_path}'...")
        try:
            data = _load_json_robustly(ts_json_path)
            ts_list = data if isinstance(data, list) else [data]

            for ts in ts_list:
                if not isinstance(ts, dict):
                    continue
                self._add_or_update_node(ts)

        except Exception as e:
            logging.warning(f"Could not parse or process TS JSON from '{ts_json_path}': {e}")

    # def merge_nodes(self, node_ids: List[Union[int, str]]) -> None:
    #     """
    #     Merges two nodes in the graph to create a new, composite node.
    #
    #     This is the primary state-changing operation driven by the RL agent.
    #
    #     Method of Action:
    #     1.  Retrieves the TS data for the two nodes to be merged (`A` and `B`).
    #     2.  Computes the new merged TS using `merge_transition_systems`.
    #     3.  Assigns a new, unique ID (`C`) to the merged TS.
    #     4.  Adds the new node `C` to the graph.
    #     5.  Rewires all incoming/outgoing edges from `A` and `B` to point to `C`.
    #     6.  Removes the original nodes `A` and `B` from the graph.
    #
    #     Text Diagram of Edge Rewiring:
    #     BEFORE MERGE:
    #     [P1] -> [A] -> [S1]
    #     [P2] -> [B] -> [S2]
    #
    #     AFTER MERGING A and B into C:
    #     [P1] -> [C] -> [S1]
    #     [P2] -> [C] -> [S2]
    #     """
    #     """
    #     ✅ FIXED: Merges two nodes with validation.
    #     """
    #     if len(node_ids) != 2:
    #         raise ValueError(f"merge_nodes requires exactly two node IDs, got {len(node_ids)}")
    #
    #     a, b = node_ids
    #
    #     # ✅ NEW: Comprehensive validation
    #     if a not in self.graph:
    #         raise KeyError(f"Node {a} not in graph. Available: {list(self.graph.nodes())}")
    #     if b not in self.graph:
    #         raise KeyError(f"Node {b} not in graph. Available: {list(self.graph.nodes())}")
    #
    #     # ✅ NEW: Prevent self-merge
    #     if a == b:
    #         raise ValueError(f"Cannot merge node with itself: {a}")
    #
    #     # ✅ NEW: Verify nodes are connected (optional, but good validation)
    #     if not (self.graph.has_edge(a, b) or self.graph.has_edge(b, a)):
    #         print(f"[WARNING] Merging disconnected nodes {a}, {b}")
    #
    #     logging.info(f"Merging nodes {a} and {b}...")
    #     ts1 = self.graph.nodes[a]
    #     ts2 = self.graph.nodes[b]
    #
    #     # 1. Compute the merged transition system.
    #     merged_ts = merge_transition_systems(ts1, ts2)
    #     new_id = self.next_node_id
    #     self.next_node_id += 1
    #
    #     # 2. Add the new merged node to the graph.
    #     self.graph.add_node(new_id, **merged_ts)
    #     var_key = frozenset(merged_ts["incorporated_variables"])
    #     self.varset_to_node[var_key] = new_id
    #
    #     # 3. Rewire edges from the original nodes to the new node.
    #     self._rewire_edges(a, new_id)
    #     self._rewire_edges(b, new_id)
    #
    #     # 4. Remove the original nodes.
    #     self.graph.remove_nodes_from([a, b])
    #
    #     self._invalidate_caches()  # ✅ ADD THIS LINE
    #     logging.info(f"Successfully merged nodes into new node {new_id} with {merged_ts['num_states']} states.")

    # --- REPLACE THE EXISTING merge_nodes METHOD WITH THIS ---
    def merge_nodes(self, node_ids: List[Union[int, str]]) -> None:
        """✅ FIXED: Merges nodes and invalidates caches."""
        if len(node_ids) != 2:
            raise ValueError(f"merge_nodes requires exactly two node IDs, got {len(node_ids)}")

        a, b = node_ids

        # Validation (kept from your existing code)
        if a not in self.graph:
            raise KeyError(f"Node {a} not in graph. Available: {list(self.graph.nodes())}")
        if b not in self.graph:
            raise KeyError(f"Node {b} not in graph. Available: {list(self.graph.nodes())}")
        if a == b:
            raise ValueError(f"Cannot merge node with itself: {a}")
        if not (self.graph.has_edge(a, b) or self.graph.has_edge(b, a)):
            # Keep this warning or remove if merging disconnected is intended
            logger.warning(f"Merging potentially disconnected nodes {a}, {b}")

        logging.info(f"Merging nodes {a} and {b}...")
        ts1 = self.graph.nodes[a]
        ts2 = self.graph.nodes[b]

        # Compute the merged transition system.
        merged_ts = merge_transition_systems(ts1, ts2)
        new_id = self.next_node_id
        self.next_node_id += 1

        # Add the new merged node to the graph.
        self.graph.add_node(new_id, **merged_ts)
        var_key = frozenset(merged_ts["incorporated_variables"])
        self.varset_to_node[var_key] = new_id

        # Rewire edges from the original nodes to the new node.
        self._rewire_edges(a, new_id)
        self._rewire_edges(b, new_id)

        # Remove the original nodes.
        self.graph.remove_nodes_from([a, b])

        # ✅ KEY CHANGE: Invalidate caches AFTER modification
        self._invalidate_caches()

        # ✅ Reset max_vars and max_iter caches as they might change
        self._max_vars_cache = None
        self._max_iter_cache = None

        logging.info(f"Successfully merged nodes into new node {new_id} with {merged_ts['num_states']} states.")

    # --- END OF REPLACEMENT FOR merge_nodes ---

    # def f_stats(self, node_id: Union[int, str]) -> Tuple[float, float, float, float]:
    #     """
    #     Calculates statistics for the 'f_before' values of a given node.
    #
    #     The 'f_before' list contains heuristic values for each abstract state
    #     within the node's transition system. These stats are used as features
    #     for the GNN policy.
    #
    #     Args:
    #         node_id: The ID of the node to analyze.
    #
    #     Returns:
    #         A tuple of (min, mean, max, std_dev) of the f-values. Returns
    #         (0.0, 0.0, 0.0, 0.0) if no f-values are present.
    #     """
    #     if node_id not in self.graph.nodes:
    #         return 0.0, 0.0, 0.0, 0.0
    #
    #     f_values = self.graph.nodes[node_id].get("f_before", [])
    #
    #     if not f_values:
    #         return 0.0, 0.0, 0.0, 0.0
    #
    #     arr = np.array(f_values, dtype=np.float32)
    #     return float(arr.min()), float(arr.mean()), float(arr.max()), float(arr.std())

    # --- REPLACE THE EXISTING f_stats METHOD WITH THIS ---
    def f_stats(self, node_id: Union[int, str]) -> Tuple[float, float, float, float]:
        """✅ CACHED: Memoized F-statistics computation."""
        # Check cache first
        if node_id in self._f_stats_cache:
            return self._f_stats_cache[node_id]

        # Compute only if not cached
        logging.debug(f"Computing f_stats for node {node_id} (cached)...")  # Added log
        if node_id not in self.graph.nodes:
            result = (0.0, 0.0, 0.0, 0.0)
        else:
            # Filter f_values ONCE
            f_values_raw = self.graph.nodes[node_id].get("f_before", [])
            # Filter out inf and large values more robustly
            f_values = [f for f in f_values_raw if f != float('inf') and f < 1_000_000_000]

            if not f_values:  # Check if list is empty AFTER filtering
                result = (0.0, 0.0, 0.0, 0.0)
            else:
                arr = np.array(f_values, dtype=np.float32)
                # Handle potential empty array after filtering again (safety)
                if arr.size == 0:
                    result = (0.0, 0.0, 0.0, 0.0)
                else:
                    # Use np.nanmin, np.nanmean etc. if NaNs are possible, otherwise regular functions are fine
                    result = (
                        float(np.min(arr)),
                        float(np.mean(arr)),
                        float(np.max(arr)),
                        float(np.std(arr))
                    )

        # Cache for next time
        self._f_stats_cache[node_id] = result
        return result

    # --- END OF REPLACEMENT FOR f_stats ---

    def _load_atomic_systems(self, ts_json_path: str) -> None:
        """
        Loads the initial set of atomic transition systems from a JSON file.

        These form the initial nodes of the graph. Atomic systems are identified
        by having `iteration == -1`.
        """
        logging.info(f"Loading atomic systems from '{ts_json_path}'...")
        data = _load_json_robustly(ts_json_path)
        ts_list = data if isinstance(data, list) else [data]

        num_loaded = 0
        for ts in ts_list:
            if isinstance(ts, dict) and ts.get("iteration", -1) == -1:
                self._add_or_update_node(ts)
                num_loaded += 1

        # Set the next node ID to be higher than any existing integer ID to avoid collisions.
        int_ids = [n for n in self.graph.nodes if isinstance(n, int)]
        self.next_node_id = max(int_ids, default=-1) + 1
        logging.info(f"Loaded {num_loaded} atomic systems. Next node ID set to {self.next_node_id}.")

    # REPLACE THE OLD METHOD WITH THIS NEW ONE
    def _load_causal_edges(self, cg_json_path: str) -> None:
        """Loads the causal graph edges from a JSON file into the graph."""
        logging.info(f"Loading causal edges from '{cg_json_path}'...")
        try:
            with open(cg_json_path, "r") as f:
                data = json.load(f)

            edges = data.get("edges", [])
            if not edges:
                logging.warning("Causal graph file contains no 'edges' key or the list is empty.")
                return

            # Debug: log what nodes exist before trying to add edges
            logging.info(f"Current graph nodes before adding edges: {list(self.graph.nodes())}")

            num_added = 0
            for edge in edges:
                src = edge.get("from")
                tgt = edge.get("to")

                # This check is crucial and now more explicit
                if src is not None and tgt is not None:
                    if self.graph.has_node(src) and self.graph.has_node(tgt):
                        self.graph.add_edge(src, tgt)
                        num_added += 1
                        logging.info(f"Added edge ({src}, {tgt})")
                    else:
                        logging.warning(
                            f"Skipping edge ({src}, {tgt}) because one or both nodes do not exist in the graph. "
                            f"Current nodes: {list(self.graph.nodes())}"
                        )
                else:
                    logging.warning(f"Edge has None values: from={src}, to={tgt}")

            logging.info(f"Loaded {num_added} causal edges. Final edge count: {len(list(self.graph.edges()))}")

        except FileNotFoundError:
            logging.warning(f"Causal graph file '{cg_json_path}' not found. No edges loaded.")
            if not self.is_debug:
                raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading causal edges: {e}", exc_info=True)
            if not self.is_debug:
                raise

    def _add_or_update_node(self, ts: Dict[str, Any]) -> None:
        """
        Adds a new node or updates an existing node's data based on a TS dict.

        The identity of a node is determined by its set of "incorporated_variables".
        If a node representing a given set of variables already exists, its
        attributes are updated. Otherwise, a new node is created.

        This method also removes the large 'transitions' list from the node data
        to save memory.

        ✅ NEW: Validates that TS has meaningful data before adding.
        """
        # Validate input
        if not ts or not isinstance(ts, dict):
            logging.warning("Skipping invalid TS: not a dict or empty")
            return

        ts_data = ts.copy()

        # ✅ MEMORY: Don't store full transitions - just count them
        transitions = ts_data.pop("transitions", [])
        ts_data["num_transitions"] = len(transitions)

        ts_data.pop("transitions", None)

        # ✅ NEW: Comprehensive validation
        inc_vars = ts_data.get("incorporated_variables", [])

        if not inc_vars:
            logging.warning("❌ Skipping TS with NO INCORPORATED VARIABLES")
            logging.warning(f"   TS keys: {list(ts.keys())}")
            logging.warning(f"   TS content: {ts}")
            return

        # ✅ NEW: Validate other critical fields
        num_states = ts_data.get("num_states", 0)
        if num_states <= 0:
            logging.warning(f"⚠️  Skipping TS with invalid num_states: {num_states}")
            return

        # ✅ NEW: Check if TS looks complete
        required_fields = ["num_states", "init_state", "goal_states"]
        missing = [k for k in required_fields if k not in ts_data]
        if missing:
            logging.warning(f"⚠️  TS missing fields: {missing}")
            logging.warning(f"   Available fields: {list(ts_data.keys())}")
            # Don't skip - continue with defaults
            for field in missing:
                if field == "init_state":
                    ts_data[field] = 0
                elif field == "goal_states":
                    ts_data[field] = [0]

        # Now proceed with normal logic
        var_key = frozenset(inc_vars)
        existing_node_id = self.varset_to_node.get(var_key)

        if existing_node_id is not None and existing_node_id in self.graph:
            logging.info(f"✓ Updating existing node {existing_node_id} with {num_states} states")
            self.graph.nodes[existing_node_id].update(ts_data)
        else:
            # New node
            is_atomic = ts_data.get("iteration", -1) == -1
            if is_atomic:
                node_id = inc_vars[0]
                logging.info(f"✓ Adding atomic node {node_id} for variable {inc_vars[0]}")
            else:
                node_id = self.next_node_id
                self.next_node_id += 1
                logging.info(f"✓ Adding merged node {node_id} for variables {inc_vars}")

            self.graph.add_node(node_id, **ts_data)
            self.varset_to_node[var_key] = node_id

            # Update counter
            if isinstance(node_id, int):
                self.next_node_id = max(self.next_node_id, node_id + 1)

            logging.info(f"   → Node {node_id} has {num_states} states")


    def _rewire_edges(self, old_id: Union[int, str], new_id: Union[int, str]) -> None:
        """
        Moves all incoming and outgoing edges from an old node to a new node.
        """
        # Rewire incoming edges: for every predecessor `p` of `old_id`, add edge `(p, new_id)`.
        if old_id in self.graph:
            for predecessor in list(self.graph.predecessors(old_id)):
                if predecessor != new_id:  # Avoid self-loops with the other merged node
                    self.graph.add_edge(predecessor, new_id)
            # Rewire outgoing edges: for every successor `s` of `old_id`, add edge `(new_id, s)`.
            for successor in list(self.graph.successors(old_id)):
                if successor != new_id:
                    self.graph.add_edge(new_id, successor)

    def display(self) -> None:
        """
        Renders and displays the current state of the graph using matplotlib.

        Note: This is intended for interactive debugging and requires the
        `matplotlib` library to be installed.
        """
        if plt is None:
            logging.warning("matplotlib is not installed. Cannot display graph.")
            return

        plt.figure(figsize=(12, 9))
        pos = nx.spring_layout(self.graph, seed=42)

        labels = {
            n: f"ID: {n}\n|S|={d.get('num_states', '?')}\nIter: {d.get('iteration', '?')}"
            for n, d in self.graph.nodes(data=True)
        }

        nx.draw_networkx(
            self.graph,
            pos,
            labels=labels,
            node_size=1500,
            node_color="lightblue",
            font_size=8,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
        )
        plt.title("Transition System Causal Graph", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # This block serves as a "smoke test" to verify basic functionality.
    # It requires dummy JSON files to be present in the same directory.
    logging.info("--- Running GraphTracker Smoke Test ---")

    # Create dummy files for testing purposes
    DUMMY_CG_FILE = "causal_graph_test.json"
    DUMMY_TS_FILE = "ts_test.json"

    cg_data = {"edges": [{"from": 0, "to": 1}, {"from": 1, "to": 2}]}
    ts_data = [
        {"num_states": 2, "init_state": 0, "goal_states": [1], "incorporated_variables": [0], "iteration": -1},
        {"num_states": 3, "init_state": 1, "goal_states": [2], "incorporated_variables": [1], "iteration": -1},
        {"num_states": 4, "init_state": 2, "goal_states": [0], "incorporated_variables": [2], "iteration": -1},
    ]

    with open(DUMMY_CG_FILE, "w") as f:
        json.dump(cg_data, f)
    with open(DUMMY_TS_FILE, "w") as f:
        json.dump(ts_data, f)

    try:
        # 1. Test initialization
        tracker = GraphTracker(ts_json_path=DUMMY_TS_FILE, cg_json_path=DUMMY_CG_FILE, is_debug=True)
        print("\nInitial Graph Nodes:", list(tracker.graph.nodes()))
        print("Initial Graph Edges:", list(tracker.graph.edges()))
        # tracker.display() # Uncomment for visual inspection

        # 2. Test merging
        if len(tracker.graph.nodes) >= 2:
            nodes_to_merge = [0, 1]
            tracker.merge_nodes(nodes_to_merge)
            print(f"\nGraph Nodes after merging {nodes_to_merge}:", list(tracker.graph.nodes()))
            print("Graph Edges after merging:", list(tracker.graph.edges()))
            # tracker.display()

            # 3. Test f_stats on the new node
            new_node_id = list(tracker.graph.nodes)[-1]
            # Add some dummy f-values to test f_stats
            tracker.graph.nodes[new_node_id]['f_before'] = [10, 20, 30, 40]
            stats = tracker.f_stats(new_node_id)
            print(
                f"\nF-stats for new node {new_node_id}: min={stats[0]}, mean={stats[1]}, max={stats[2]}, std={stats[3]}")

        else:
            print("\nNot enough nodes to test merge.")

        logging.info("--- Smoke Test Completed Successfully ---")

    except Exception as e:
        logging.error(f"--- Smoke Test FAILED: {e} ---")

    finally:
        # Clean up dummy files
        import os

        if os.path.exists(DUMMY_CG_FILE): os.remove(DUMMY_CG_FILE)
        if os.path.exists(DUMMY_TS_FILE): os.remove(DUMMY_TS_FILE)