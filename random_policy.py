import numpy as np
import time
from merge_env import MergeEnv


class RandomMergePolicy:
    """A simple baseline policy that always selects a valid merge action uniformly at random."""

    def __init__(self, env: MergeEnv):
        """Initializes the policy with a reference to the environment."""
        self.env = env

    def predict(self, observation: dict, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """Predicts a random action based on the current observation."""
        num_edges_arr = np.asarray(observation.get("num_edges", 0)).reshape(-1)
        num_edges = int(num_edges_arr[0]) if num_edges_arr.size > 0 else 0

        action = 0 if num_edges <= 0 else np.random.randint(num_edges)
        return np.array([action], dtype=np.int64), None

    def run_episode(self) -> tuple[float, int, int]:
        """
        Runs a full episode in the environment using this random policy.

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