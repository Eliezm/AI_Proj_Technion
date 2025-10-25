import numpy as np
from merge_env import MergeEnv


class DefaultMergePolicy:
    """
    A greedy baseline policy that chooses the merge action that results in
    the smallest number of total abstract states.
    """

    def __init__(self, env: MergeEnv):
        """Initializes the policy with a reference to the environment."""
        self.env = env

    def _get_total_states(self, tracker) -> int:
        """Calculates the sum of states across all nodes."""
        return sum(d.get("num_states", 0) for _, d in tracker.graph.nodes(data=True))

    def predict(self, observation: dict, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """✅ OPTIMIZED: Lightweight feature extraction for baseline."""
        graph_tracker = self.env.graph_tracker
        if not graph_tracker or not graph_tracker.graph.edges:
            return np.array([0], dtype=np.int64), None

        edges = list(graph_tracker.graph.edges)
        if not edges:
            return np.array([0], dtype=np.int64), None

        best_action_index = 0
        min_resulting_states = float('inf')

        # ✅ SIMPLIFIED: Only use state counts (not full feature extraction)
        for i, (u, v) in enumerate(edges):
            try:
                # Simple heuristic: prefer merging smaller systems
                u_states = graph_tracker.graph.nodes[u].get("num_states", 0)
                v_states = graph_tracker.graph.nodes[v].get("num_states", 0)

                # Quick estimate: product size
                resulting_states = u_states * v_states

                if resulting_states < min_resulting_states:
                    min_resulting_states = resulting_states
                    best_action_index = i
            except Exception as e:
                continue

        return np.array([best_action_index], dtype=np.int64), None

    def run_episode(self) -> tuple[float, int, int]:
        """
        Runs a full episode using this greedy policy.

        Returns:
            A tuple containing:
            - The total reward accumulated during the episode.
            - The final plan cost from the planner log.
            - The final number of expansions from the planner log.
        """
        obs, info = self.env.reset()
        done = False
        total_reward = 0.0
        final_plan_cost = 0
        final_expansions = 0
        steps = 0
        max_steps = 1000

        while not done and steps < max_steps:
            try:
                action, _ = self.predict(obs)
                scalar_action = int(action[0])

                obs, reward, terminated, truncated, info = self.env.step(scalar_action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

                # ✅ FIX: Extract from info dict correctly
                if isinstance(info, dict):
                    if "plan_cost" in info:
                        final_plan_cost = int(info["plan_cost"])
                    if "num_expansions" in info:
                        final_expansions = int(info["num_expansions"])

            except Exception as e:
                import traceback
                traceback.print_exc()
                break

        return total_reward, final_plan_cost, final_expansions