## Setup

- Install [uv](https://docs.astral.sh/uv/).
- From the project root, install the environment once: `uv sync`
- You can run commands with `uv run ...` without activating the virtualenv, or activate it with `source .venv/bin/activate`.

## Data

Place `QB-video.csv` (or your dataset) under `data/`. The code expects a target column `item_id` and click/engagement feature columns `click,follow,like,share` by default.

## Running baselines with uv

- Random baseline: `uv run python src/eval.py --model random --data-path data/QB-video.csv --target-col item_id --feature-cols click,follow,like,share --split val --k 20`
- Popularity recommender (most frequent item): `uv run python src/eval.py --model popular --data-path data/QB-video.csv --target-col item_id --feature-cols click,follow,like,share --split val --k 20`

## Training and evaluating the LSTM model

1) Train the LSTM session model (saves `checkpoints/gru_last.pt`):  
`uv run python src/train_gru.py`
2) Evaluate it:  
`uv run python src/eval.py --model gru --gru-checkpoint checkpoints/gru_last.pt --data-path data/QB-video.csv --target-col item_id --feature-cols click,follow,like,share --split val --k 20`
