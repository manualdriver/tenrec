import torch
import numpy as np
from torch.utils.data import Dataset
import polars as pl

"""
Session-Based Datast
Holds out the last video interaction for testing, and the second to last for validation
Throws out sessions with less than 10 examples, and takes the most recent max_len interactions
Test set includes all interactions, validation holds out one, and train holds out two
"""
class sbrDataset(Dataset):
    def __init__(self, path, feature_cols, target_col, min_len=10, max_len=30, split="train"):
        self.split = split
        seqs = []
        df = pl.scan_csv(path, null_values=["\\N"]).collect()
        df = df.with_columns((pl.col("user_id").cum_count().over("user_id")+ 1).alias("timestamp"))
        for user_id, user_df, in df.sort(["user_id", "timestamp"]).group_by("user_id"):
            seq = np.array(user_df[target_col].to_numpy(), dtype=np.int64, copy=True)
            if feature_cols:
                feature_mat = (
                    user_df.select(feature_cols)
                    .to_numpy()
                    .astype(np.float32, copy=True)
                )
            else:
                feature_mat = np.empty((seq.size, 0), dtype=np.float32)
            if seq.size < min_len: #ignore sessions less than minimum
                continue
            if max_len and seq.size > max_len: #take most recent sequence
                seq = seq[-max_len:]
                feature_mat = feature_mat[-max_len:]
            if seq.size < max_len: #pad to max_len with zeroes
                pad_width = max_len - seq.size
                seq = np.pad(seq, (pad_width, 0), constant_values=0)
                feature_mat = np.pad(feature_mat, ((pad_width, 0), (0, 0)), constant_values=0.0)
            seqs.append((seq, feature_mat))
        
        self.split_indices = []
        for seq, feature_mat in seqs:
            if split == "train":
                self.split_indices.append((seq[:-3], feature_mat[:-3], seq[-3]))
            elif split == "val":
                self.split_indices.append((seq[:-2], feature_mat[:-2], seq[-2]))
            elif split == "test":
                self.split_indices.append((seq[:-1], feature_mat[:-1], seq[-1]))
    
    def __len__(self): return len(self.split_indices)

    def __getitem__(self, idx):
            history, history_features, target = self.split_indices[idx]
            item_history = torch.from_numpy(history).long()
            feature_history = torch.from_numpy(history_features).float()
            return item_history, feature_history, torch.tensor(int(target), dtype=torch.long)
