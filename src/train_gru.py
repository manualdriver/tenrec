from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import sbrDataset
from rnn import SBRNN

##THIS IS WRONG: GET UNIQUE NUMBER OF VIDEO IDS INSTEAD OR NORMALIZE!!!
def get_vocab_size(path: str, target_col: str) -> int:
    """Return vocab size including padding index."""
    max_id = (
        pl.scan_csv(path, null_values=["\\N"])
        .select(pl.col(target_col).max())
        .collect()
        .item()
    )
    return int(max_id) + 1  # +1 to account for padding id 0


def run_epoch(loader, model, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for item_hist, feat_hist, targets in loader:
        item_hist = item_hist.to(device)
        feat_hist = feat_hist.to(device)
        targets = targets.to(device)

        logits = model(item_hist, feat_hist)
        loss = criterion(logits, targets)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_examples += targets.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_examples if total_examples else 0.0
    return avg_loss, accuracy


def main():
    data_path = "./data/QB-video.csv"
    feature_cols = ["click", "follow", "like", "share"]
    target_col = "item_id"
    batch_size = 64
    num_workers = 4
    min_len, max_len = 10, 30
    emb_dim, hidden_dim = 256, 512
    dropout = 0.4
    lr = 1e-3
    epochs = 3

    feature_dim = len(feature_cols)
    vocab_size = get_vocab_size(data_path, target_col)

    train_set = sbrDataset(
        path=data_path,
        feature_cols=feature_cols,
        target_col=target_col,
        min_len=min_len,
        max_len=max_len,
        split="train",
    )
    val_set = sbrDataset(
        path=data_path,
        feature_cols=feature_cols,
        target_col=target_col,
        min_len=min_len,
        max_len=max_len,
        split="val",
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SBRNN(
        n_videos=vocab_size,
        feature_dim=feature_dim,
        emb_size=emb_dim,
        hidden_size=hidden_dim,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            train_loader, model, criterion, optimizer, device, train=True
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                val_loader, model, criterion, optimizer=None, device=device, train=False
            )
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f}, train acc {train_acc*100:.2f}% | "
            f"val loss {val_loss:.4f}, val acc {val_acc*100:.2f}%"
        )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "gru_last.pt"
    config = {
        "data_path": data_path,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "min_len": min_len,
        "max_len": max_len,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "vocab_size": vocab_size,
        "feature_dim": feature_dim,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
