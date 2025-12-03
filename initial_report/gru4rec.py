import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    """
    Minimal GRU4Rec-style model for session-based recommendation.

    Accepts padded item histories and optional per-event features. Uses the
    last non-padding hidden state for prediction.
    """

    def __init__(
        self,
        n_items: int,
        feature_dim: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.emb = nn.Embedding(n_items, emb_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        if feature_dim > 0:
            self.feature_proj = nn.Linear(feature_dim, emb_size)
            rnn_input_dim = emb_size * 2
        else:
            self.feature_proj = None
            rnn_input_dim = emb_size

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            rnn_input_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, n_items)

    def forward(self, item_history: torch.Tensor, feature_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_history: [batch, seq_len] padded item ids.
            feature_history: [batch, seq_len, feature_dim] float features (can be empty).
        """
        item_emb = self.dropout(self.emb(item_history))
        if self.feature_proj is not None:
            feat_emb = self.feature_proj(feature_history)
            rnn_input = torch.cat([item_emb, feat_emb], dim=-1)
        else:
            rnn_input = item_emb

        outputs, _ = self.gru(rnn_input)

        lengths = item_history.ne(0).sum(dim=1)
        last_indices = torch.clamp(lengths - 1, min=0)
        batch_indices = torch.arange(item_history.size(0), device=item_history.device)
        last_hidden = outputs[batch_indices, last_indices]
        logits = self.output(self.dropout(last_hidden))
        return logits
