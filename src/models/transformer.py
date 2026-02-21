"""
InsiderTransformerAE — Full Reconstruction Autoencoder
====================================================
Encoder-only Transformer with full MSE loss (no masking, no [CLS]).

Architecture:
  Input: (batch, seq_len, n_continuous) + (batch, seq_len, n_categorical)
  → Continuous: Linear(n_continuous → d_model)
  → Categorical: Embed → Linear(total_embed → d_model)
  → Fuse: Linear(2*d_model → d_model) + LayerNorm + Dropout
  → + Learned Positional Encoding
  → Transformer Encoder (4 layers, 8 heads, dropout=0.2)
  → Reconstruction Head: LayerNorm → Linear(d_model → d_ff) → GELU → Linear(d_ff → n_continuous)
  → Output: (batch, seq_len, n_continuous)

Loss: MSE(input, output) over behavioral features only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class CategoricalEmbedding(nn.Module):
    """Embed each categorical feature separately, then project to d_model."""

    def __init__(self, cat_cardinalities: Dict[str, int],
                 cat_embed_dims: Dict[str, int], d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0

        for name, cardinality in cat_cardinalities.items():
            dim = cat_embed_dims.get(name, 8)
            self.embeddings[name] = nn.Embedding(cardinality + 1, dim)
            total_embed_dim += dim

        self.proj = nn.Linear(total_embed_dim, d_model)
        self.cat_names = list(cat_cardinalities.keys())

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: (batch, seq_len, n_cat_features) integer tensor
        Returns:
            (batch, seq_len, d_model)
        """
        embeds = []
        for i, name in enumerate(self.cat_names):
            embeds.append(self.embeddings[name](x_cat[:, :, i]))
        cat_embedded = torch.cat(embeds, dim=-1)
        return self.proj(cat_embedded)


class InsiderTransformerAE(nn.Module):
    """
    Full reconstruction Transformer autoencoder for insider threat detection.

    Dual-input: continuous features + categorical embeddings.
    No masking, no [CLS] token. Full MSE loss on behavioral features.

    Args:
        n_continuous: Number of continuous input features
        cat_cardinalities: {feature_name: num_categories}
        cat_embed_dims: {feature_name: embedding_dim}
        d_model: Transformer hidden dimension (default: 128)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of encoder layers (default: 4)
        d_ff: Feedforward dimension (default: 512)
        max_seq_len: Maximum sequence length (default: 60)
        dropout: Dropout rate (default: 0.2)
        behavioral_indices: Optional list of feature indices used for loss/scoring.
            If None, all n_continuous features are used.
    """

    def __init__(
        self,
        n_continuous: int,
        cat_cardinalities: Dict[str, int],
        cat_embed_dims: Dict[str, int],
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 60,
        dropout: float = 0.2,
        behavioral_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        if behavioral_indices is not None:
            self.register_buffer(
                'behavioral_indices',
                torch.tensor(behavioral_indices, dtype=torch.long),
            )
        else:
            self.behavioral_indices = None

        # --- Input projections ---
        self.continuous_proj = nn.Linear(n_continuous, d_model)
        self.categorical_embed = CategoricalEmbedding(
            cat_cardinalities, cat_embed_dims, d_model
        )
        self.input_fuse = nn.Linear(2 * d_model, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # --- Positional encoding (learned) ---
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        # --- Transformer Encoder (Pre-LayerNorm, GELU) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False
        )

        # --- Reconstruction head (2-layer MLP) ---
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, n_continuous),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for Linear, normal(0, 0.02) for Embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # -----------------------------------------------------------------
    # Input pipeline: project -> fuse -> positional encoding
    # -----------------------------------------------------------------

    def _project_inputs(
        self, x_cont: torch.Tensor, x_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project continuous and categorical inputs separately.

        Returns:
            h_cont: (batch, seq_len, d_model)
            h_cat:  (batch, seq_len, d_model)
        """
        return self.continuous_proj(x_cont), self.categorical_embed(x_cat)

    def _fuse_inputs(
        self, h_cont: torch.Tensor, h_cat: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate continuous + categorical projections, project, norm, dropout.

        Args:
            h_cont: (batch, seq_len, d_model)
            h_cat:  (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        h = self.input_fuse(torch.cat([h_cont, h_cat], dim=-1))
        return self.input_dropout(self.input_norm(h))

    def _encode(self, h: torch.Tensor) -> torch.Tensor:
        """Add positional encoding and run encoder.

        Args:
            h: (batch, seq_len, d_model) — fused input
        Returns:
            (batch, seq_len, d_model) — encoder output
        """
        batch, seq_len, _ = h.shape
        positions = torch.arange(seq_len, device=h.device).unsqueeze(0)
        h = h + self.pos_encoding(positions)
        return self.encoder(h)

    # -----------------------------------------------------------------
    # Forward / scoring
    # -----------------------------------------------------------------

    def _get_score_features(self) -> Optional[torch.Tensor]:
        """Return behavioral_indices buffer if set, else None (use all)."""
        return self.behavioral_indices

    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass. Full reconstruction (no masking).

        Args:
            x_cont: (batch, seq_len, n_continuous) float
            x_cat: (batch, seq_len, n_cat) long

        Returns:
            predictions: (batch, seq_len, n_continuous)
        """
        h_cont, h_cat = self._project_inputs(x_cont, x_cat)
        h = self._fuse_inputs(h_cont, h_cat)
        h = self._encode(h)
        predictions = self.reconstruction_head(h)
        return predictions

    def get_reconstruction_error(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample reconstruction error.

        Uses only behavioral features if behavioral_indices is set.

        Returns:
            error: (batch,) — mean MSE per sample over all positions
        """
        predictions = self.forward(x_cont, x_cat)
        feat_idx = self._get_score_features()
        if feat_idx is not None:
            diff = (predictions[:, :, feat_idx] - x_cont[:, :, feat_idx]) ** 2
        else:
            diff = (predictions - x_cont) ** 2
        return diff.mean(dim=(1, 2))

    @torch.no_grad()
    def get_embeddings(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Extract fixed-size sequence embeddings from the encoder.

        Useful for downstream tasks like FAISS-based user baselining.

        Args:
            x_cont: (batch, seq_len, n_continuous)
            x_cat: (batch, seq_len, n_cat)
            pooling: "mean" (default) or "last"
                - "mean": average encoder output across all time steps
                - "last": use the last time step's hidden state

        Returns:
            embeddings: (batch, d_model)
        """
        self.eval()
        h_cont, h_cat = self._project_inputs(x_cont, x_cat)
        h = self._fuse_inputs(h_cont, h_cat)
        h = self._encode(h)  # (batch, seq_len, d_model)

        if pooling == "mean":
            return h.mean(dim=1)
        elif pooling == "last":
            return h[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling method: {pooling!r}. Use 'mean' or 'last'.")

    @torch.no_grad()
    def get_anomaly_scores(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> torch.Tensor:
        """Anomaly score = 0.5 * mean_day_error + 0.5 * max_day_error.

        Uses only behavioral features if behavioral_indices is set.

        Args:
            x_cont: (batch, seq_len, n_continuous)
            x_cat: (batch, seq_len, n_cat)
        Returns:
            scores: (batch,) — anomaly score per sequence
        """
        self.eval()
        predictions = self.forward(x_cont, x_cat)
        feat_idx = self._get_score_features()
        if feat_idx is not None:
            errors = ((predictions[:, :, feat_idx] - x_cont[:, :, feat_idx]) ** 2).mean(dim=-1)
        else:
            errors = ((predictions - x_cont) ** 2).mean(dim=-1)

        # 0.5 * mean + 0.5 * max
        return 0.5 * errors.mean(dim=1) + 0.5 * errors.max(dim=1)[0]


# =========================================================================
# Factory
# =========================================================================

def create_model(
    config: dict,
    n_continuous: int,
    cat_cardinalities: Dict[str, int],
    behavioral_indices: Optional[List[int]] = None,
) -> InsiderTransformerAE:
    """Create InsiderTransformerAE from config."""
    model_cfg = config['model']
    embed_cfg = config['embeddings']

    return InsiderTransformerAE(
        n_continuous=n_continuous,
        cat_cardinalities=cat_cardinalities,
        cat_embed_dims=embed_cfg,
        d_model=model_cfg['d_model'],
        n_heads=model_cfg['n_heads'],
        n_layers=model_cfg['n_layers'],
        d_ff=model_cfg['d_ff'],
        max_seq_len=model_cfg['lookback'],
        dropout=model_cfg['dropout'],
        behavioral_indices=behavioral_indices,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    cat_cards = {'user_id': 1000, 'pc': 1100, 'role': 42,
                 'department': 7, 'functional_unit': 6}
    cat_dims = {'user_id': 16, 'pc': 16, 'role': 8,
                'department': 4, 'functional_unit': 4}

    behavioral = list(range(40))  # first 40 of 45 are "behavioral"
    model = InsiderTransformerAE(
        n_continuous=45,
        cat_cardinalities=cat_cards,
        cat_embed_dims=cat_dims,
        d_model=128, n_heads=8, n_layers=4, d_ff=512,
        max_seq_len=60, dropout=0.2,
        behavioral_indices=behavioral,
    )

    print(f"Parameters: {count_parameters(model):,}")
    print(f"Behavioral features: {len(behavioral)}/{model.n_continuous}")
    print(f"Model:\n{model}\n")

    batch = 4
    x_cont = torch.randn(batch, 60, 45)
    max_ids = [c + 1 for c in cat_cards.values()]
    x_cat = torch.stack([torch.randint(0, m, (batch, 60)) for m in max_ids], dim=-1)

    pred = model(x_cont, x_cat)
    print(f"Predictions: {pred.shape}")

    error = model.get_reconstruction_error(x_cont, x_cat)
    print(f"Recon error: {error.shape}, mean={error.mean():.4f}")

    scores = model.get_anomaly_scores(x_cont, x_cat)
    print(f"Anomaly scores: {scores.shape}, mean={scores.mean():.4f}")

    emb_mean = model.get_embeddings(x_cont, x_cat, pooling="mean")
    emb_last = model.get_embeddings(x_cont, x_cat, pooling="last")
    print(f"Embeddings (mean): {emb_mean.shape}")
    print(f"Embeddings (last): {emb_last.shape}")
