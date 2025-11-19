import torch
import torch.nn as nn

class SBRNN(nn.Module):
    def __init__(self, n_videos, feature_dim, emb_size, hidden_size, dropout=0.4):
        super().__init__()
        self.feature_dim = feature_dim
        self.emb = nn.Embedding(n_videos, emb_size, padding_idx=0)
        rnn_input_dim = emb_size + feature_dim
        self.rnn = nn.GRU(
            rnn_input_dim,
            hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_videos)
    
    def forward(self, video_id, features):
        """
        video_id: LongTensor [batch, seq_len] with left-padded histories.
        features: FloatTensor [batch, seq_len, feature_dim] (possibly empty last dim).
        """
        item_emb = self.emb(video_id)
        if self.feature_dim:
            rnn_input = torch.cat([item_emb, features], dim=-1)
        else:
            rnn_input = item_emb
        outputs, _ = self.rnn(rnn_input)

        lengths = video_id.ne(0).sum(dim=1)
        last_indices = torch.clamp(lengths - 1, min=0)
        batch_indices = torch.arange(video_id.size(0), device=video_id.device)
        last_hidden = outputs[batch_indices, last_indices]
        logits = self.output_layer(last_hidden)
        return logits
