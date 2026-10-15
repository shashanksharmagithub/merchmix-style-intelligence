from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from data_science.config import CONFIG, RAW_DIR, ensure_project_dirs


DATASET_HINT = "h-and-m-personalized-fashion-recommendations"


def check_required_files(raw_dir: Path = RAW_DIR) -> dict[str, bool]:
    return {
        "transactions_train.csv": (raw_dir / "transactions_train.csv").exists(),
        "articles.csv": (raw_dir / "articles.csv").exists(),
        "images/": (raw_dir / "images").exists(),
    }


def print_status(raw_dir: Path = RAW_DIR) -> None:
    status = check_required_files(raw_dir)
    print("Data status:")
    for name, exists in status.items():
        print(f"  {'OK' if exists else 'MISSING'} {name}")


def try_kaggle_download(raw_dir: Path = RAW_DIR) -> None:
    """Download Kaggle competition data when kaggle CLI credentials are configured."""
    ensure_project_dirs()
    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise RuntimeError(
            "Kaggle CLI is not installed. Install requirements.txt or manually place "
            "transactions_train.csv, articles.csv, and images/ under data/raw/."
        )

    cmd = [
        kaggle,
        "competitions",
        "download",
        "-c",
        DATASET_HINT,
        "-p",
        str(raw_dir),
        "--force",
    ]
    subprocess.run(cmd, check=True)
    print("Downloaded Kaggle archive. Unzip it inside data/raw/ before training.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or download H&M assignment data.")
    parser.add_argument("--download", action="store_true", help="Attempt Kaggle CLI download.")
    args = parser.parse_args()

    print_status(CONFIG.transactions_path.parent)
    if args.download:
        try_kaggle_download(CONFIG.transactions_path.parent)
        print_status(CONFIG.transactions_path.parent)
    else:
        print(
            "\nPlace the official H&M files in data/raw/:\n"
            "  data/raw/transactions_train.csv\n"
            "  data/raw/articles.csv\n"
            "  data/raw/images/<article-id-prefix>/<article-id>.jpg\n"
        )


if __name__ == "__main__":
    main()
