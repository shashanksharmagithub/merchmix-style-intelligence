from __future__ import annotations

import argparse

from data_science.eda import run_eda
from data_science.features.build_features import build_training_dataset
from data_science.generate_concepts import make_concept_board
from data_science.seasonal import build_seasonal_styles
from data_science.train import train_and_predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full style intelligence pipeline.")
    parser.add_argument(
        "--max-transactions",
        type=int,
        default=None,
        help="Optional row cap for quick local iteration before the full run.",
    )
    args = parser.parse_args()

    print("1/5 Running EDA...")
    run_eda(max_transactions=args.max_transactions)
    print("2/5 Building features...")
    build_training_dataset(max_transactions=args.max_transactions)
    print("3/5 Training model and writing predictions...")
    train_and_predict()
    print("4/5 Creating concept briefs and deterministic concept board...")
    make_concept_board()
    print("5/5 Building seasonal view...")
    build_seasonal_styles()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
