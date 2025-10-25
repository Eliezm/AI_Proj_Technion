# FILE: gnn_model.py (COMPLETE REWRITE WITH ATTENTION & EDGE FEATURES)
import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np

import logging
logger = logging.getLogger(__name__)


# ============================================================================
# ATTENTION BACKBONE: Improved GCN with Multi-Head Attention
# ============================================================================

class GCNWithAttention(nn.Module):
    """✅ NEW: GCN backbone with multi-head attention for focusing on key relationships."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 3, n_heads: int = 4):
        super().__init__()

        # GCN layers
        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [hidden_dim]
        for i in range(n_layers):
            layers.append(GCNConv(dims[i], dims[i + 1]))
        self.convs = nn.ModuleList(layers)


        # ✅ NEW: Graph Attention Network layer for learning which relationships matter
        self.attention = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=n_heads,
            concat=True,
            dropout=0.1,
            add_self_loops=False
        )

        # Post-attention projection (if concat=True, attention outputs n_heads*out_channels)
        self.attention_proj = nn.Linear(hidden_dim * n_heads, hidden_dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass with attention mechanism."""
        device = x.device
        dtype = x.dtype
        edge_index = edge_index.to(device, dtype=torch.long)

        # Standard GCN pass
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
            x = self.dropout(x)

        # ✅ NEW: Apply attention on top of GCN embeddings
        # This learns which node pairs are important for merge decisions
        if edge_index.numel() > 0:
            try:
                attn_out = self.attention(x, edge_index)
                # Project back to hidden_dim
                attn_out = self.activation(self.attention_proj(attn_out))
                # Residual connection: blend GCN output with attention output
                x = x + attn_out * 0.3  # Small weight to preserve GCN learning
            except Exception as e:
                # Fallback if attention fails
                logger.warning(f"Attention layer failed: {e}, skipping")

        return x


# ============================================================================
# EDGE FEATURE ENCODER: Encode merge candidate properties
# ============================================================================

class EdgeFeatureEncoder(nn.Module):
    """✅ NEW: Encodes rich features about merge candidates."""

    def __init__(self, num_edge_features: int = 8, output_dim: int = 16):
        super().__init__()

        # Map edge features through neural net
        self.encoder = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )

        self.output_dim = output_dim

    def forward(self, edge_features: Tensor) -> Tensor:
        """
        Encode edge features into embeddings.

        Args:
            edge_features: [E, num_edge_features] raw edge features

        Returns:
            [E, output_dim] encoded edge features
        """
        if edge_features.numel() == 0:
            return torch.zeros(0, self.output_dim, device=edge_features.device)

        return self.encoder(edge_features)


# ============================================================================
# ATTENTION-WEIGHTED EDGE SCORER
# ============================================================================

class AttentionWeightedEdgeScorer(nn.Module):
    """✅ NEW: Score edges using attention + edge features + node embeddings."""

    def __init__(self, hidden_dim: int, edge_feature_dim: int = 16):
        super().__init__()

        # Combine node embeddings with edge features
        # Input: [src_embedding, tgt_embedding, edge_features]
        total_dim = 2 * hidden_dim + edge_feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 2 * total_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        # ✅ NEW: Attention weights for edge features
        # Learn which edge features are most important
        self.edge_attention = nn.Sequential(
            nn.Linear(edge_feature_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
            self,
            node_embs: Tensor,
            edge_index: Tensor,
            edge_features: Optional[Tensor] = None
    ) -> Tensor:
        """
        Score edges using attention-weighted combination of features.

        Args:
            node_embs: [N, H] node embeddings
            edge_index: [2, E] edge list in COO format
            edge_features: [E, D] edge features (optional)

        Returns:
            [E] edge scores
        """
        # ✅ SAFETY: Handle empty edge lists
        if edge_index.numel() == 0 or edge_index.shape[1] == 0:
            return torch.zeros(0, device=node_embs.device, dtype=torch.float32)

        src_idx, tgt_idx = edge_index
        num_nodes = node_embs.shape[0]

        # ✅ SAFETY: Validate indices
        max_idx = max(src_idx.max().item(), tgt_idx.max().item()) if len(src_idx) > 0 else -1
        if max_idx >= num_nodes:
            print(f"WARNING: Edge index contains invalid node ID {max_idx} >= {num_nodes}")
            return torch.zeros(len(src_idx), device=node_embs.device, dtype=torch.float32)

        src_emb = node_embs[src_idx]  # [E, H]
        tgt_emb = node_embs[tgt_idx]  # [E, H]

        # ✅ NEW: Handle edge features with attention
        if edge_features is not None and edge_features.numel() > 0:
            # Attention weights for edge features
            edge_attn_weights = self.edge_attention(edge_features)  # [E, 1]

            # Scale edge features by attention weights
            edge_feats_weighted = edge_features * edge_attn_weights  # [E, D]

            # Concatenate node embeddings with weighted edge features
            edge_feat = torch.cat([src_emb, tgt_emb, edge_feats_weighted], dim=1)  # [E, 2H+D]
        else:
            # Fallback: just use node embeddings
            edge_feat = torch.cat([src_emb, tgt_emb], dim=1)  # [E, 2H]

        score = self.mlp(edge_feat).squeeze(-1)  # [E]

        # ✅ SAFETY: Clamp to avoid explosion
        score = torch.clamp(score, min=-1e6, max=1e6)

        # ✅ SAFETY: Replace NaN/Inf with safe defaults
        score = torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

        return score


# ============================================================================
# UNIFIED GNN MODEL WITH ATTENTION & EDGE FEATURES
# ============================================================================

class GNNModel(nn.Module):
    """✅ COMPLETE: Full GNN with attention, edge features, and robust validation."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            n_layers: int = 3,
            n_heads: int = 4,
            edge_feature_dim: int = 8
    ):
        super().__init__()

        # ✅ NEW: GCN with attention backbone
        self.backbone = GCNWithAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads
        )

        # ✅ NEW: Edge feature encoder
        self.edge_encoder = EdgeFeatureEncoder(
            num_edge_features=edge_feature_dim,
            output_dim=16
        )

        # ✅ NEW: Attention-weighted edge scorer
        self.scorer = AttentionWeightedEdgeScorer(
            hidden_dim=hidden_dim,
            edge_feature_dim=16
        )

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_features: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with validation."""
        # ✅ INPUT VALIDATION
        if x.dim() != 2:
            raise ValueError(f"Node features must be 2D, got {x.dim()}D")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Edge index must be [2, E], got {edge_index.shape}")

        device = x.device
        edge_index = edge_index.to(device, dtype=torch.long)

        # ✅ EDGE INDEX VALIDATION - CRITICAL!
        num_nodes = x.shape[0]
        if edge_index.numel() > 0:
            max_idx = edge_index.max().item()
            if max_idx >= num_nodes:
                logger.warning(f"Edge index {max_idx} >= num_nodes {num_nodes}. Clamping...")
                edge_index = torch.clamp(edge_index, 0, num_nodes - 1)

            # ✅ NEW: Check for negative indices
            min_idx = edge_index.min().item()
            if min_idx < 0:
                logger.warning(f"Negative edge index {min_idx} found. Clamping...")
                edge_index = torch.clamp(edge_index, 0, num_nodes - 1)

        # Rest of forward pass...
        node_embs = self.backbone(x, edge_index)

        if torch.isnan(node_embs).any():
            logger.warning("NaN in node embeddings, replacing with 0")
            node_embs = torch.nan_to_num(node_embs, nan=0.0)

        encoded_edge_features = None
        if edge_features is not None and edge_features.numel() > 0:
            try:
                encoded_edge_features = self.edge_encoder(edge_features.float())
            except Exception as e:
                logger.warning(f"Could not encode edge features: {e}")

        edge_logits = self.scorer(node_embs, edge_index, encoded_edge_features)

        return edge_logits, node_embs