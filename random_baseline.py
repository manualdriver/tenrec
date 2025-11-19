import torch


class RandomBaseline:
    """Predicts random item ids for session-based recommendation benchmarks."""

    def __init__(self, vocab_size: int, seed: int | None = None):
        self.vocab_size = vocab_size
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def predict_topk(
        self, batch_size: int, k: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Return top-k random predictions for a batch.

        The items are sampled uniformly with replacement from the catalog.
        """
        if device is None:
            device = torch.device("cpu")
        samples = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, k),
            generator=self.generator,
            device=torch.device("cpu"),
        )
        return samples.to(device)
