import argparse
import ast

import torch
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader

from datasets import sbrDataset
from gru4rec import GRU4Rec
from popular_baseline import PopularBaseline
from random_baseline import RandomBaseline
from rnn import SBRNN
from session_knn import SessionKNN
from train_gru import get_vocab_size


def parse_feature_cols(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(col) for col in parsed]
    except (ValueError, SyntaxError):
        pass
    return [col.strip() for col in raw.split(",") if col.strip()]


def compute_batch_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    k: int,
    scores: torch.Tensor | None = None,
) -> tuple[int, float, int]:
    """Return hit count, ndcg contribution, and batch size."""
    preds = preds.cpu()
    targets = targets.cpu()
    batch_size, pred_k = preds.shape
    hits = preds.eq(targets.unsqueeze(1))
    hit_any = hits.any(dim=1)
    batch_hits = hit_any.sum().item()

    y_true = torch.zeros(batch_size, pred_k, dtype=torch.float32)
    hit_rows, hit_cols = torch.nonzero(hits, as_tuple=True)
    if hit_rows.numel():
        y_true[hit_rows, hit_cols] = 1.0

    if scores is None:
        y_score = torch.arange(pred_k, 0, -1, dtype=torch.float32).unsqueeze(0)
        y_score = y_score.repeat(batch_size, 1)
    else:
        y_score = scores.cpu()

    used_k = min(k, pred_k)
    ndcg_val = ndcg_score(y_true.numpy(), y_score.numpy(), k=used_k)
    return batch_hits, ndcg_val * batch_size, batch_size


def evaluate_random(
    loader: DataLoader, vocab_size: int, device: torch.device, k: int
) -> tuple[float, float]:
    model = RandomBaseline(vocab_size)
    total_examples = 0
    total_hits = 0
    cumulative_ndcg = 0.0

    for _, _, targets in loader:
        batch_size = targets.size(0)
        targets = targets.to(device)
        preds = model.predict_topk(batch_size, k, device=device)

        hits, ndcg_val, count = compute_batch_metrics(preds, targets, k)
        total_hits += hits
        cumulative_ndcg += ndcg_val
        total_examples += count

    hr = total_hits / total_examples if total_examples else 0.0
    ndcg = cumulative_ndcg / total_examples if total_examples else 0.0
    return hr, ndcg


def evaluate_popular(
    loader: DataLoader, data_path: str, target_col: str, device: torch.device, k: int
) -> tuple[float, float]:
    model = PopularBaseline.from_csv(data_path, target_col)
    total_examples = 0
    total_hits = 0
    cumulative_ndcg = 0.0

    for _, _, targets in loader:
        batch_size = targets.size(0)
        targets = targets.to(device)
        preds = model.predict_topk(batch_size, k, device=device)

        hits, ndcg_val, count = compute_batch_metrics(preds, targets, k)
        total_hits += hits
        cumulative_ndcg += ndcg_val
        total_examples += count

    hr = total_hits / total_examples if total_examples else 0.0
    ndcg = cumulative_ndcg / total_examples if total_examples else 0.0
    return hr, ndcg


def evaluate_gru(
    loader: DataLoader, checkpoint_path: str, device: torch.device, k: int
) -> tuple[float, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = SBRNN(
        n_videos=config["vocab_size"],
        feature_dim=config["feature_dim"],
        emb_size=config["emb_dim"],
        hidden_size=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    total_examples = 0
    total_hits = 0
    cumulative_ndcg = 0.0

    with torch.no_grad():
        for item_hist, feat_hist, targets in loader:
            item_hist = item_hist.to(device)
            feat_hist = feat_hist.to(device)
            targets = targets.to(device)
            logits = model(item_hist, feat_hist)
            pred_k = min(k, logits.size(1))
            topk_scores, topk_indices = torch.topk(logits, k=pred_k, dim=1)

            hits, ndcg_val, count = compute_batch_metrics(
                topk_indices, targets, k, scores=topk_scores
            )
            total_hits += hits
            cumulative_ndcg += ndcg_val
            total_examples += count

    hr = total_hits / total_examples if total_examples else 0.0
    ndcg = cumulative_ndcg / total_examples if total_examples else 0.0
    return hr, ndcg


def evaluate_gru4rec(
    loader: DataLoader, checkpoint_path: str, device: torch.device, k: int
) -> tuple[float, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = GRU4Rec(
        n_items=config["vocab_size"],
        feature_dim=config["feature_dim"],
        emb_size=config["emb_dim"],
        hidden_size=config["hidden_dim"],
        num_layers=config.get("num_layers", 2),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    total_examples = 0
    total_hits = 0
    cumulative_ndcg = 0.0

    with torch.no_grad():
        for item_hist, feat_hist, targets in loader:
            item_hist = item_hist.to(device)
            feat_hist = feat_hist.to(device)
            targets = targets.to(device)
            logits = model(item_hist, feat_hist)
            pred_k = min(k, logits.size(1))
            topk_scores, topk_indices = torch.topk(logits, k=pred_k, dim=1)

            hits, ndcg_val, count = compute_batch_metrics(
                topk_indices, targets, k, scores=topk_scores
            )
            total_hits += hits
            cumulative_ndcg += ndcg_val
            total_examples += count

    hr = total_hits / total_examples if total_examples else 0.0
    ndcg = cumulative_ndcg / total_examples if total_examples else 0.0
    return hr, ndcg


def evaluate_knn(loader: DataLoader, model: SessionKNN, k: int) -> tuple[float, float]:
    total_examples = 0
    total_hits = 0
    cumulative_ndcg = 0.0

    for item_hist, _, targets in loader:
        # Compute predictions per example; SessionKNN works on CPU.
        preds = torch.stack(
            [model.predict_topk(hist.cpu(), topk=k) for hist in item_hist], dim=0
        )
        hits, ndcg_val, count = compute_batch_metrics(preds, targets, k)
        total_hits += hits
        cumulative_ndcg += ndcg_val
        total_examples += count

    hr = total_hits / total_examples if total_examples else 0.0
    ndcg = cumulative_ndcg / total_examples if total_examples else 0.0
    return hr, ndcg


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tenrec recommenders.")
    parser.add_argument(
        "--model", choices=("random", "popular", "gru", "gru4rec", "knn"), default="random"
    )
    parser.add_argument("--data-path", default="./data/QB-video.csv")
    parser.add_argument("--feature-cols", default="click,follow,like,share")
    parser.add_argument("--target-col", default="item_id")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-len", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--k", type=int, default=20, help="Top-K cutoff for metrics.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Deprecated alias for --gru-checkpoint (kept for backward compatibility).",
    )
    parser.add_argument(
        "--gru-checkpoint",
        default="./checkpoints/gru_last.pt",
        help="Checkpoint path for GRU evaluations.",
    )
    parser.add_argument(
        "--gru4rec-checkpoint",
        default="./checkpoints/gru4rec_qb.pt",
        help="Checkpoint path for GRU4Rec evaluations.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=50,
        help="Number of nearest sessions to vote for k-NN recommender.",
    )
    parser.add_argument(
        "--knn-min-common",
        type=int,
        default=1,
        help="Minimum overlapping items to consider a neighbor in k-NN recommender.",
    )
    parser.add_argument(
        "--knn-index-batch-size",
        type=int,
        default=512,
        help="Batch size when building the k-NN index from the train split.",
    )
    parser.add_argument(
        "--knn-index-workers",
        type=int,
        default=0,
        help="Number of workers when building the k-NN index from the train split.",
    )
    args = parser.parse_args()

    feature_cols = parse_feature_cols(args.feature_cols)
    dataset = sbrDataset(
        path=args.data_path,
        feature_cols=feature_cols,
        target_col=args.target_col,
        min_len=args.min_len,
        max_len=args.max_len,
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gru_checkpoint = args.gru_checkpoint
    if args.checkpoint:
        gru_checkpoint = args.checkpoint

    if args.model == "random":
        vocab_size = get_vocab_size(args.data_path, args.target_col)
        hr, ndcg = evaluate_random(loader, vocab_size, device, args.k)
    elif args.model == "popular":
        hr, ndcg = evaluate_popular(loader, args.data_path, args.target_col, device, args.k)
    elif args.model == "gru":
        hr, ndcg = evaluate_gru(loader, gru_checkpoint, device, args.k)
    elif args.model == "gru4rec":
        hr, ndcg = evaluate_gru4rec(loader, args.gru4rec_checkpoint, device, args.k)
    elif args.model == "knn":
        train_dataset = sbrDataset(
            path=args.data_path,
            feature_cols=feature_cols,
            target_col=args.target_col,
            min_len=args.min_len,
            max_len=args.max_len,
            split="train",
        )
        knn_model = SessionKNN(
            train_dataset,
            neighbors=args.knn_neighbors,
            min_common=args.knn_min_common,
            loader_batch_size=args.knn_index_batch_size,
            num_workers=args.knn_index_workers,
        )
        hr, ndcg = evaluate_knn(loader, knn_model, args.k)
    else:
        raise ValueError(f"Unsupported model '{args.model}'")

    print(f"{args.model} HR@{args.k}: {hr*100:.2f}%")
    print(f"{args.model} NDCG@{args.k}: {ndcg:.4f}")


if __name__ == "__main__":
    main()
