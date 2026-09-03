import argparse
from pathlib import Path

import pandas as pd

from .pipeline import train_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label", default="Label")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    print(train_baseline(pd.read_csv(args.input), args.label, args.output))


if __name__ == "__main__":
    main()
