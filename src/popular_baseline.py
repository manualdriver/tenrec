import polars as pl
import torch


class PopularBaseline:
    """Deterministic recommender that returns the single most popular item."""

    def __init__(self, item_id: int):
        self.item_id = int(item_id)

    @classmethod
    def from_csv(cls, path: str, target_col: str) -> "PopularBaseline":
        """
        Build the baseline by scanning the dataset once and picking the top item.
        """
        top_item = (
            pl.scan_csv(path, null_values=["\\N"])
            .group_by(target_col)
            .count()
            .sort("count", descending=True)
            .select(target_col)
            .limit(1)
            .collect()
        )
        if top_item.height == 0:
            raise ValueError("Could not determine a popular item from an empty dataset.")
        return cls(top_item[target_col][0])

    def predict_topk(
        self, batch_size: int, k: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """Return a (batch_size, k) tensor filled with the most popular item id."""
        if device is None:
            device = torch.device("cpu")
        return torch.full(
            (batch_size, k),
            fill_value=self.item_id,
            dtype=torch.long,
            device=device,
        )
