from __future__ import annotations

import heapq
from collections import Counter
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset


def _extract_history(item_tensor: torch.Tensor) -> list[int]:
    """Return list of non-padding item ids from a padded history tensor."""
    return [int(x) for x in item_tensor.tolist() if int(x) != 0]


class SessionKNN:
    """
    Simple session-based k-NN recommender.

    Uses Jaccard similarity between the item sets of two sessions and votes for
    the neighbor targets, weighted by similarity.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        neighbors: int = 50,
        min_common: int = 1,
        loader_batch_size: int = 512,
        num_workers: int = 0,
    ):
        self.neighbors = neighbors
        self.min_common = min_common

        histories: list[set[int]] = []
        targets: list[int] = []
        target_counts: Counter[int] = Counter()

        loader = DataLoader(
            train_dataset,
            batch_size=loader_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        for item_hist, _, batch_targets in loader:
            for hist, tgt in zip(item_hist, batch_targets):
                hist_items = set(_extract_history(hist))
                histories.append(hist_items)
                targets.append(int(tgt))
                target_counts[int(tgt)] += 1

        self.histories = histories
        self.targets = targets
        # Popular targets used as a fallback when no neighbors share items.
        self.popular_targets = [item for item, _ in target_counts.most_common()]

    @staticmethod
    def _jaccard(a: set[int], b: set[int]) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        if inter == 0:
            return 0.0
        union = len(a | b)
        return inter / union if union else 0.0

    def predict_topk(self, query_history: torch.Tensor, topk: int) -> torch.Tensor:
        """
        Return top-k item predictions for a single padded history tensor.
        """
        query_items = set(_extract_history(query_history))
        if not self.histories:
            return torch.zeros(topk, dtype=torch.long)

        sims: list[tuple[float, int]] = []
        for hist_set, target in zip(self.histories, self.targets):
            sim = self._jaccard(query_items, hist_set)
            if sim <= 0.0:
                continue
            if self.min_common and len(query_items & hist_set) < self.min_common:
                continue
            sims.append((sim, target))

        # Take the top-N most similar sessions.
        top_neighbors = heapq.nlargest(self.neighbors, sims, key=lambda x: x[0])
        scored: dict[int, float] = {}
        for sim, target in top_neighbors:
            scored[target] = scored.get(target, 0.0) + sim

        if not scored:
            fallback = self.popular_targets[:topk]
            # Pad with zeros if we run out of popular targets.
            padded = fallback + [0] * max(0, topk - len(fallback))
            return torch.tensor(padded[:topk], dtype=torch.long)

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        preds: list[int] = [item for item, _ in ranked[:topk]]

        if len(preds) < topk:
            # Fill remaining slots with popular items not already present.
            used = set(preds)
            for item in self.popular_targets:
                if item in used:
                    continue
                preds.append(item)
                if len(preds) == topk:
                    break
        return torch.tensor(preds[:topk], dtype=torch.long)
