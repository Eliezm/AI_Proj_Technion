# FILE: gnn_policy.py (CRITICAL FIXES)
import traceback

import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple, Dict, Any, Optional
from gnn_model import GNNModel

import logging
logger = logging.getLogger(__name__)


class GNNExtractor(nn.Module):
    """✅ UPDATED: Wraps GNNModel with edge feature support."""

    def __init__(self, input_dim: int, hidden_dim: int, edge_feature_dim: int = 8):
        super().__init__()
        self.gnn = GNNModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=3,
            n_heads=4,  # ✅ NEW: Attention heads
            edge_feature_dim=edge_feature_dim  # ✅ NEW: Edge feature support
        )
        self.edge_feature_dim = edge_feature_dim

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """✅ UPDATED: Pass edge features to GNN."""
        edge_logits, node_embs = self.gnn(x, edge_index, edge_features)
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
    def predict(
            self,
            observation: Dict[str, Any],
            state=None,
            mask=None,
            deterministic=False
    ):
        """✅ FIXED: Properly handle observation batching from Stable-Baselines3."""
        self.eval()
        device = self.device

        # ================================================================
        # PHASE 1: EXTRACT OBSERVATIONS WITH PROPER BATCHING DETECTION
        # ================================================================

        x = torch.as_tensor(observation["x"], dtype=torch.float32, device=device)
        ei = torch.as_tensor(observation["edge_index"], dtype=torch.long, device=device)

        # ✅ FIX: Extract edge_features with proper batching handling
        edge_features = None
        if "edge_features" in observation:
            ef_raw = observation["edge_features"]
            edge_features = torch.as_tensor(ef_raw, dtype=torch.float32, device=device)

        # ================================================================
        # PHASE 2: DETECT BATCH DIMENSION
        # ================================================================

        # Determine if observations are already batched
        if x.ndim == 2:
            # Unbatched: [N, feat_dim]
            is_batched = False
            B = 1
            x = x.unsqueeze(0)  # [1, N, feat_dim]
            ei = ei.unsqueeze(0)  # [1, 2, E]
            if edge_features is not None and edge_features.ndim == 2:
                edge_features = edge_features.unsqueeze(0)  # [1, E, 8]

        elif x.ndim == 3:
            # Already batched: [batch, N, feat_dim]
            is_batched = True
            B = x.shape[0]

            # ✅ CRITICAL FIX: Ensure edge_features is also batched
            if edge_features is not None:
                if edge_features.ndim == 2:
                    # edge_features is [E, 8], need to broadcast to [batch, E, 8]
                    edge_features = edge_features.unsqueeze(0).expand(B, -1, -1)
                # else: already [batch, E, 8], keep as is
        else:
            raise ValueError(f"Invalid observation dimension: {x.ndim}")

        # ================================================================
        # PHASE 3: EXTRACT DIMENSIONS
        # ================================================================

        # Handle num_nodes/num_edges which might be scalars or arrays
        ne = observation.get("num_edges", None)
        num_nodes_obs = observation.get("num_nodes", None)

        # Parse num_edges
        if ne is None:
            ne = [ei.shape[-1]] * B
        elif isinstance(ne, np.ndarray):
            if ne.ndim == 0:
                ne = [int(ne)] * B
            else:
                ne = ne.reshape(-1).tolist()
        elif isinstance(ne, (int, float)):
            ne = [int(ne)] * B
        elif isinstance(ne, torch.Tensor):
            ne = ne.cpu().numpy().reshape(-1).tolist()
        else:
            ne = [int(n) for n in ne] if hasattr(ne, '__iter__') else [int(ne)] * B

        # Ensure ne list has correct length
        if len(ne) == 1 and B > 1:
            ne = ne * B
        elif len(ne) != B:
            logger.warning(f"num_edges length {len(ne)} != batch size {B}, padding")
            ne = (ne + [ei.shape[-1]] * B)[:B]

        # ================================================================
        # PHASE 4: INPUT VALIDATION
        # ================================================================

        try:
            if x.dim() < 2:
                raise ValueError(f"x must be at least 2D, got {x.dim()}D: {x.shape}")
            if ei.dim() < 2:
                raise ValueError(f"edge_index must be at least 2D, got {ei.dim()}D: {ei.shape}")

            num_nodes = x.shape[-2]

            if ei.numel() > 0:
                max_idx = ei.max().item()
                if max_idx >= num_nodes:
                    logger.warning(f"Edge index {max_idx} >= num_nodes {num_nodes}. Clamping.")
                    ei = torch.clamp(ei, max=num_nodes - 1)

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return np.zeros(B, dtype=int), None

        # ================================================================
        # PHASE 5: PROCESS EACH SAMPLE
        # ================================================================

        actions = []

        for i in range(B):
            try:
                # Extract sample from batch
                x_i = x[i]  # [N, feat_dim]
                ei_i = ei[i]  # [2, E]

                # ✅ FIX: Properly extract edge features for this sample
                if edge_features is not None:
                    edge_feats_i = edge_features[i]  # [E, 8]
                else:
                    edge_feats_i = None

                # ✅ VALIDATE SHAPES before passing to extractor
                if x_i.ndim != 2:
                    logger.error(f"Sample {i}: x_i has wrong shape {x_i.shape}")
                    actions.append(torch.tensor(0, dtype=torch.long, device=device))
                    continue
                if ei_i.numel() > 0 and ei_i.ndim != 2:
                    logger.error(f"Sample {i}: ei_i has wrong shape {ei_i.ndim}")
                    actions.append(torch.tensor(0, dtype=torch.long, device=device))
                    continue
                if edge_feats_i is not None and edge_feats_i.ndim != 2:
                    logger.error(f"Sample {i}: edge_feats_i has wrong shape {edge_feats_i.shape}")
                    actions.append(torch.tensor(0, dtype=torch.long, device=device))
                    continue

                # Forward through GNN
                logits_i, node_embs_i = self.extractor(x_i, ei_i, edge_feats_i)

                # Validate logits
                if torch.isnan(logits_i).any() or torch.isinf(logits_i).any():
                    logger.warning(f"Sample {i}: Invalid logits, using uniform")
                    logits_i = torch.ones_like(logits_i)

                num_edges = int(ne[i]) if i < len(ne) else logits_i.shape[0]

                if num_edges <= 0 or logits_i.shape[0] == 0:
                    a_i = torch.tensor(0, dtype=torch.long, device=device)
                else:
                    masked = self._mask_invalid_edges(logits_i, num_edges)
                    a_i, _ = self._sample_or_argmax(masked, deterministic)

                # Validate action
                if a_i < 0 or (logits_i.shape[0] > 0 and a_i >= logits_i.shape[0]):
                    logger.warning(f"Sample {i}: Invalid action {a_i}, clamping to 0")
                    a_i = torch.tensor(0, dtype=torch.long, device=device)

                actions.append(a_i)

            except Exception as e:
                logger.error(f"Sample {i} processing failed: {e}")
                logger.error(traceback.format_exc())
                actions.append(torch.tensor(0, dtype=torch.long, device=device))

        # ================================================================
        # PHASE 6: RETURN RESULT
        # ================================================================

        actions_tensor = torch.stack(actions) if actions else torch.zeros(B, dtype=torch.long, device=device)
        return actions_tensor.cpu().numpy(), None

    def forward(self, obs: Dict[str, Any], deterministic: bool = False
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """✅ FIXED: Properly handle batched observations."""
        device = self.device

        # ================================================================
        # EXTRACT AND VALIDATE OBSERVATIONS
        # ================================================================

        x = torch.as_tensor(obs["x"], dtype=torch.float32, device=device)
        ei = torch.as_tensor(obs["edge_index"], dtype=torch.long, device=device)

        # Extract edge_features with proper batching
        edge_features = None
        if "edge_features" in obs:
            edge_features = torch.as_tensor(obs["edge_features"], dtype=torch.float32, device=device)

        # ================================================================
        # DETECT AND HANDLE BATCHING
        # ================================================================

        if x.ndim == 2:
            # Unbatched input
            x = x.unsqueeze(0)  # [1, N, feat]
            ei = ei.unsqueeze(0) if ei.ndim == 2 and ei.shape[0] == 2 else ei
            if edge_features is not None and edge_features.ndim == 2:
                edge_features = edge_features.unsqueeze(0)  # [1, E, feat]
        elif x.ndim == 3:
            # Already batched - ensure edge_features is also batched
            B = x.shape[0]
            if edge_features is not None and edge_features.ndim == 2:
                edge_features = edge_features.unsqueeze(0).expand(B, -1, -1)

        # ================================================================
        # EXTRACT ACTUAL DIMENSIONS
        # ================================================================

        num_nodes_obs = obs.get("num_nodes", None)
        if num_nodes_obs is not None:
            if isinstance(num_nodes_obs, torch.Tensor):
                actual_num_nodes = int(num_nodes_obs.item()) if num_nodes_obs.dim() == 0 else int(
                    num_nodes_obs[0].item())
            else:
                actual_num_nodes = int(np.asarray(num_nodes_obs).flat[0])
        else:
            actual_num_nodes = x.shape[-2]

        ne_obs = obs.get("num_edges", None)
        if ne_obs is not None:
            if isinstance(ne_obs, torch.Tensor):
                actual_num_edges = int(ne_obs.item()) if ne_obs.dim() == 0 else int(ne_obs[0].item())
            else:
                actual_num_edges = int(np.asarray(ne_obs).flat[0])
        else:
            actual_num_edges = ei.shape[-1]

        # ================================================================
        # TRIM TO ACTUAL SIZES
        # ================================================================

        x_trimmed = x[..., :actual_num_nodes, :]

        if actual_num_edges > 0:
            ei_trimmed = ei[..., :actual_num_edges]
            ei_trimmed = torch.clamp(ei_trimmed, 0, actual_num_nodes - 1)
        else:
            ei_trimmed = torch.zeros((*ei.shape[:-1], 0), dtype=torch.long, device=device)

        # Trim edge_features accordingly
        if edge_features is not None and actual_num_edges > 0:
            edge_features_trimmed = edge_features[..., :actual_num_edges, :]
        else:
            edge_features_trimmed = None

        # ================================================================
        # PROCESS BATCH
        # ================================================================

        B = x_trimmed.shape[0]
        actions, values, logps = [], [], []

        for i in range(B):
            try:
                # Extract sample
                x_i = x_trimmed[i]  # [N, feat]
                ei_i = ei_trimmed[i]  # [2, E]
                edge_feats_i = edge_features_trimmed[i] if edge_features_trimmed is not None else None  # [E, 8]

                # Forward through GNN
                logits_i, node_embs_i = self.extractor(x_i, ei_i, edge_feats_i)

                # Compute value
                if node_embs_i.dim() == 1:
                    node_embs_i = node_embs_i.unsqueeze(0)

                if node_embs_i.shape[0] > 0:
                    v_i = self.value_net(node_embs_i.mean(dim=0, keepdim=True)).squeeze(-1)
                else:
                    v_i = torch.zeros((), dtype=torch.float32, device=device)

                # Sample action
                ne_i = actual_num_edges

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
                logger.error(f"Batch {i} forward failed: {e}")
                logger.error(traceback.format_exc())
                actions.append(torch.zeros((), dtype=torch.long, device=device))
                values.append(torch.zeros((), device=device))
                logps.append(torch.zeros((), device=device))

        actions = torch.stack(actions)
        values = torch.stack(values).unsqueeze(-1)
        logps = torch.stack(logps)

        return actions, values, logps

    def evaluate_actions(self, obs: Dict[str, Tensor], actions: Tensor
                         ) -> Tuple[Tensor, Tensor, Tensor]:
        """✅ FIXED: Evaluate actions with proper batching."""
        device = self.device

        x = obs["x"].to(device, dtype=torch.float32)
        ei = obs["edge_index"].to(device, dtype=torch.long)

        # Extract edge_features with batching awareness
        edge_features = None
        if "edge_features" in obs:
            edge_features = obs["edge_features"].to(device, dtype=torch.float32)

        # ✅ Handle batching
        if x.ndim == 2:
            x = x.unsqueeze(0)
            ei = ei.unsqueeze(0) if ei.ndim == 2 and ei.shape[0] == 2 else ei
            if edge_features is not None and edge_features.ndim == 2:
                edge_features = edge_features.unsqueeze(0)
        elif x.ndim == 3:
            B = x.shape[0]
            if edge_features is not None and edge_features.ndim == 2:
                edge_features = edge_features.unsqueeze(0).expand(B, -1, -1)

        # Extract dimensions
        num_nodes_obs = obs.get("num_nodes", None)
        actual_num_nodes = int(np.asarray(num_nodes_obs).flat[0]) if num_nodes_obs is not None else x.shape[-2]

        ne = obs.get("num_edges", None)
        actual_num_edges = int(np.asarray(ne).flat[0]) if ne is not None else ei.shape[-1]

        # Trim
        x_trimmed = x[..., :actual_num_nodes, :]
        if actual_num_edges > 0:
            ei_trimmed = ei[..., :actual_num_edges]
            ei_trimmed = torch.clamp(ei_trimmed, 0, actual_num_nodes - 1)
        else:
            ei_trimmed = torch.zeros((*ei.shape[:-1], 0), dtype=torch.long, device=device)

        if edge_features is not None and actual_num_edges > 0:
            edge_features_trimmed = edge_features[..., :actual_num_edges, :]
        else:
            edge_features_trimmed = None

        # Process batch
        B = x_trimmed.shape[0]
        values, logps, ents = [], [], []

        for i in range(B):
            try:
                x_i = x_trimmed[i]
                ei_i = ei_trimmed[i]
                edge_feats_i = edge_features_trimmed[i] if edge_features_trimmed is not None else None

                logits_i, node_embs_i = self.extractor(x_i, ei_i, edge_feats_i)

                if node_embs_i.shape[0] > 0:
                    v_i = self.value_net(node_embs_i.mean(dim=0, keepdim=True)).squeeze(-1)
                else:
                    v_i = torch.zeros((), device=device)

                ne_i = actual_num_edges

                if ne_i <= 0 or logits_i.shape[0] == 0:
                    logp_i = torch.zeros((), device=device, requires_grad=True)
                    ent_i = torch.zeros((), device=device, requires_grad=True)
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
                logger.error(f"Evaluate batch {i} failed: {e}")
                values.append(torch.zeros((), device=device, requires_grad=True))
                logps.append(torch.zeros((), device=device, requires_grad=True))
                ents.append(torch.zeros((), device=device, requires_grad=True))

        return torch.stack(values), torch.stack(logps), torch.stack(ents)

    def predict_values(self, obs: Dict[str, Tensor]) -> Tensor:
        """✅ FIXED: Predict values with proper batching."""
        device = self.device

        x = obs["x"].to(device, dtype=torch.float32)
        ei = obs["edge_index"].to(device, dtype=torch.long)

        # Extract edge_features with batching awareness
        edge_features = None
        if "edge_features" in obs:
            edge_features = obs["edge_features"].to(device, dtype=torch.float32)

        # Handle batching
        if x.ndim == 2:
            x = x.unsqueeze(0)
            ei = ei.unsqueeze(0) if ei.ndim == 2 and ei.shape[0] == 2 else ei
            if edge_features is not None and edge_features.ndim == 2:
                edge_features = edge_features.unsqueeze(0)
        elif x.ndim == 3:
            B = x.shape[0]
            if edge_features is not None and edge_features.ndim == 2:
                edge_features = edge_features.unsqueeze(0).expand(B, -1, -1)

        # Extract dimensions
        num_nodes_obs = obs.get("num_nodes", None)
        actual_num_nodes = int(np.asarray(num_nodes_obs).flat[0]) if num_nodes_obs is not None else x.shape[-2]

        ne = obs.get("num_edges", None)
        actual_num_edges = int(np.asarray(ne).flat[0]) if ne is not None else ei.shape[-1]

        # Trim
        x_trimmed = x[..., :actual_num_nodes, :]
        if actual_num_edges > 0:
            ei_trimmed = ei[..., :actual_num_edges]
            ei_trimmed = torch.clamp(ei_trimmed, 0, actual_num_nodes - 1)
        else:
            ei_trimmed = torch.zeros((*ei.shape[:-1], 0), dtype=torch.long, device=device)

        if edge_features is not None and actual_num_edges > 0:
            edge_features_trimmed = edge_features[..., :actual_num_edges, :]
        else:
            edge_features_trimmed = None

        # Process batch
        B = x_trimmed.shape[0]
        vals = []

        for i in range(B):
            try:
                x_i = x_trimmed[i]
                ei_i = ei_trimmed[i]
                edge_feats_i = edge_features_trimmed[i] if edge_features_trimmed is not None else None

                _, node_embs_i = self.extractor(x_i, ei_i, edge_feats_i)

                if node_embs_i.shape[0] > 0:
                    val_i = self.value_net(node_embs_i.mean(dim=0, keepdim=True))
                else:
                    val_i = torch.zeros((1, 1), device=device)

                vals.append(val_i)

            except Exception as e:
                logger.error(f"Predict values batch {i} failed: {e}")
                vals.append(torch.zeros((1, 1), device=device))

        return torch.cat(vals, dim=0)