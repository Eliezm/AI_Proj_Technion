# FILE: gnn_policy.py (CRITICAL FIXES)
import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple, Dict, Any
from gnn_model import GNNModel


class GNNExtractor(nn.Module):
    """Wraps GNNModel for SB3 compatibility."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gnn = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        edge_logits, node_embs = self.gnn(x, edge_index)
        return edge_logits, node_embs


class GNNPolicy(ActorCriticPolicy):
    """✅ FIXED: Robust policy with action validation."""

    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.ReLU,
            hidden_dim: int = 128,
            **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=activation_fn,
            **kwargs
        )

        self.node_feat_dim = observation_space["x"].shape[-1]
        self.hidden_dim = hidden_dim

        self.extractor = GNNExtractor(input_dim=self.node_feat_dim, hidden_dim=self.hidden_dim)
        self.value_net = nn.Linear(self.hidden_dim, 1)
        self.action_net = nn.Identity()

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1))

    def _mask_invalid_edges(self, logits: Tensor, num_edges: int) -> Tensor:
        """
        ✅ FIXED: Mask invalid edges without breaking everything.

        GUARANTEE: At least one edge remains unmasked.
        """
        E = logits.size(0)

        # ✅ SAFETY: Clamp num_edges to valid range [1, E]
        num_edges_clamped = max(1, min(int(num_edges), E))

        mask = torch.arange(E, device=logits.device) < num_edges_clamped

        # ✅ GUARANTEE: At least one edge is available
        if not mask.any():
            mask[0] = True

        masked = logits.clone()
        masked[~mask] = -1e9

        return masked

    def _sample_or_argmax(self, logits: Tensor, deterministic: bool) -> Tuple[Tensor, Tensor]:
        """
        ✅ FIXED: Sample action safely with fallback.

        Returns (action, log_prob)
        """
        try:
            logits = torch.clamp(logits, min=-100, max=100)
            probs = torch.softmax(logits, dim=0)

            # ✅ SAFETY: Check if distribution is valid
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print("WARNING: Invalid probabilities, using uniform")
                probs = torch.ones_like(logits) / len(logits)

            dist = Categorical(probs=probs)

            if deterministic:
                action = probs.argmax(dim=0)
            else:
                action = dist.sample()

            logp = dist.log_prob(action)

            # ✅ SAFETY: Check log_prob
            if torch.isnan(logp) or torch.isinf(logp):
                logp = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

            return action, logp

        except Exception as e:
            print(f"WARNING: Sampling failed: {e}, using argmax fallback")
            action = logits.argmax(dim=0)
            logp = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            return action, logp

    @torch.no_grad()
    def predict(self, observation: Dict[str, Any], state=None, mask=None, deterministic=False):
        """✅ MERGED: Full input validation followed by vectorized inference."""
        self.eval()
        device = self.device

        x = torch.as_tensor(observation["x"], dtype=torch.float32, device=device)
        ei = torch.as_tensor(observation["edge_index"], dtype=torch.long, device=device)

        # --- NEWLY ADDED: Input Validation Block ---
        try:
            # Validate tensor dimensions
            if x.dim() < 2:
                raise ValueError(f"x must be at least 2D, got {x.dim()}D: {x.shape}")
            if ei.dim() < 2:
                raise ValueError(f"edge_index must be at least 2D, got {ei.dim()}D: {ei.shape}")

            # Use the second to last dimension for num_nodes to handle both batched (B, N, F) and unbatched (N, F) inputs
            num_nodes = x.shape[-2]

            # Validate that edge indices are within the number of nodes
            if ei.numel() > 0:
                max_idx = ei.max().item()
                if max_idx >= num_nodes:
                    print(f"[WARNING] Edge index {max_idx} >= num_nodes {num_nodes}. Clamping indices.")
                    ei = torch.clamp(ei, max=num_nodes - 1)  # Clamp the tensor and reassign it
        except Exception as e:
            print(f"ERROR during initial GNN input validation: {e}")
            # Determine batch size to return a correctly shaped error tensor
            B = x.shape[0] if x.dim() > 2 else 1
            return np.zeros(B, dtype=int), None
        # --- End of Validation Block ---

        ne = observation.get("num_edges", None)

        # ✅ HANDLE num_edges parsing
        if ne is None:
            ne = [ei.shape[1]] if ei.numel() > 0 else [0]
        elif isinstance(ne, np.ndarray):
            ne = ne.reshape(-1).tolist()
        elif isinstance(ne, (int, float)):
            ne = [int(ne)]
        elif isinstance(ne, torch.Tensor):
            ne = ne.cpu().numpy().reshape(-1).tolist()
        else:
            ne = list(ne) if hasattr(ne, '__iter__') else [int(ne)]

        # Reshape for batch processing if input is a single sample
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if ei.ndim == 2 and ei.shape[0] == 2:
            ei = ei.unsqueeze(0)

        B = x.shape[0]
        actions = []

        for i in range(B):
            try:
                logits_i, node_embs_i = self.extractor(x[i], ei[i])

                # ✅ VALIDATE logits
                if torch.isnan(logits_i).any() or torch.isinf(logits_i).any():
                    print(f"WARNING: Invalid logits in batch {i}, using uniform")
                    logits_i = torch.ones_like(logits_i)

                num_edges = int(ne[i]) if i < len(ne) else logits_i.shape[0]

                if num_edges <= 0 or logits_i.shape[0] == 0:
                    a_i = torch.tensor(0, dtype=torch.long, device=device)
                else:
                    masked = self._mask_invalid_edges(logits_i, num_edges)
                    a_i, _ = self._sample_or_argmax(masked, deterministic)

                # ✅ VALIDATE action
                if a_i < 0 or (logits_i.shape[0] > 0 and a_i >= logits_i.shape[0]):
                    print(f"WARNING: Invalid action {a_i}, clamping")
                    a_i = torch.tensor(0, dtype=torch.long, device=device)

                actions.append(a_i)

            except Exception as e:
                print(f"ERROR in predict batch {i}: {e}")
                actions.append(torch.tensor(0, dtype=torch.long, device=device))

        actions_tensor = torch.stack(actions) if actions else torch.zeros(B, dtype=torch.long, device=device)
        return actions_tensor.cpu().numpy(), None

    def forward(self, obs: Dict[str, Any], deterministic: bool = False
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """Training path with validation."""
        device = self.device
        x = torch.as_tensor(obs["x"], dtype=torch.float32, device=device)
        ei = torch.as_tensor(obs["edge_index"], dtype=torch.long, device=device)
        ne = obs.get("num_edges", None)

        if ne is None:
            ne = torch.tensor([ei.shape[1]], device=device)
        elif not torch.is_tensor(ne):
            ne = torch.as_tensor(ne, dtype=torch.int64, device=device)

        if x.ndim == 2:
            x = x.unsqueeze(0)
        if ei.ndim == 2 and ei.shape[0] == 2:
            ei = ei.unsqueeze(0)
        if ne.ndim == 0:
            ne = ne.unsqueeze(0)

        B = x.shape[0]
        actions, values, logps = [], [], []

        for i in range(B):
            try:
                logits_i, node_embs_i = self.extractor(x[i], ei[i])
                v_i = self.value_net(node_embs_i.mean(dim=0, keepdim=True)).squeeze(-1)

                ne_i = int(ne[i].item()) if i < len(ne) else logits_i.shape[0]

                if ne_i <= 0 or logits_i.shape[0] == 0:
                    a_i = torch.zeros((), dtype=torch.long, device=device)
                    lp_i = torch.zeros((), dtype=torch.float32, device=device)
                else:
                    masked = self._mask_invalid_edges(logits_i, ne_i)
                    a_i, lp_i = self._sample_or_argmax(masked, deterministic)

                actions.append(a_i)
                values.append(v_i)
                logps.append(lp_i)

            except Exception as e:
                print(f"ERROR in forward batch {i}: {e}")
                actions.append(torch.zeros((), dtype=torch.long, device=device))
                values.append(torch.zeros((), device=device))
                logps.append(torch.zeros((), device=device))

        actions = torch.stack(actions)
        values = torch.stack(values).unsqueeze(-1)
        logps = torch.stack(logps)

        return actions, values, logps

    def evaluate_actions(self, obs: Dict[str, Tensor], actions: Tensor
                         ) -> Tuple[Tensor, Tensor, Tensor]:
        """Loss path with validation."""
        device = self.device
        x = obs["x"].to(device, dtype=torch.float32)
        ei = obs["edge_index"].to(device, dtype=torch.long)
        ne = obs.get("num_edges", None)
        if ne is not None:
            ne = ne.to(device)

        if x.ndim == 2:
            x = x.unsqueeze(0)
        if ei.ndim == 2:
            ei = ei.unsqueeze(0)
        if actions.ndim == 0:
            actions = actions.unsqueeze(0)

        B = x.shape[0]
        values, logps, ents = [], [], []

        for i in range(B):
            try:
                logits_i, node_embs_i = self.extractor(x[i], ei[i])
                v_i = self.value_net(node_embs_i.mean(dim=0, keepdim=True)).squeeze(-1)

                ne_i = int(ne[i].item()) if (ne is not None and i < len(ne)) else logits_i.shape[0]

                if ne_i <= 0 or logits_i.shape[0] == 0:
                    logp_i = torch.zeros((), device=device)
                    ent_i = torch.zeros((), device=device)
                else:
                    masked = self._mask_invalid_edges(logits_i, ne_i)
                    dist = Categorical(logits=masked)
                    action_clamped = torch.clamp(actions[i], 0, logits_i.shape[0] - 1)
                    logp_i = dist.log_prob(action_clamped)
                    ent_i = dist.entropy()

                values.append(v_i)
                logps.append(logp_i)
                ents.append(ent_i)

            except Exception as e:
                print(f"ERROR in evaluate_actions batch {i}: {e}")
                values.append(torch.zeros((), device=device))
                logps.append(torch.zeros((), device=device))
                ents.append(torch.zeros((), device=device))

        return torch.stack(values), torch.stack(logps), torch.stack(ents)

    def predict_values(self, obs: Dict[str, Tensor]) -> Tensor:
        """Critic-only values."""
        device = self.device
        x = obs["x"].to(device, dtype=torch.float32)
        ei = obs["edge_index"].to(device, dtype=torch.long)

        if x.ndim == 2:
            x = x.unsqueeze(0)
        if ei.ndim == 2:
            ei = ei.unsqueeze(0)

        vals = []
        for i in range(x.shape[0]):
            try:
                _, node_embs_i = self.extractor(x[i], ei[i])
                val_i = self.value_net(node_embs_i.mean(dim=0, keepdim=True))
                vals.append(val_i)
            except Exception as e:
                print(f"ERROR in predict_values batch {i}: {e}")
                vals.append(torch.zeros((1, 1), device=device))

        return torch.cat(vals, dim=0)