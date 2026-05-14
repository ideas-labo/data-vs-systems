import os
import math
import glob
import argparse
import pandas as pd


def process_single_csv(
    csv_path: str,
    small_ratio: float,
    medium_ratio: float,
    large_ratio: float,
    seed: int,
    replace: bool = False,
):
    print(f"\nhandled file: {csv_path}")
    df = pd.read_csv(csv_path)
    n = len(df)

    n_small = max(1, math.floor(n * small_ratio))
    n_medium = max(1, math.floor(n * medium_ratio))
    n_large = max(1, math.floor(n * large_ratio))

    if not replace:
        # 无放回时，样本数不能超过 n
        n_small = min(n_small, n)
        n_medium = min(n_medium, n)
        n_large = min(n_large, n)

    print(f"original rows: {n} -> small: {n_small}, medium: {n_medium}, large: {n_large} (replace={replace})")

    # 关键：每个规模都“从全集独立采样”，因此 random_state 用不同种子即可
    small_df = df.sample(n=n_small, replace=replace, random_state=seed)
    medium_df = df.sample(n=n_medium, replace=replace, random_state=seed + 1)
    large_df = df.sample(n=n_large, replace=replace, random_state=seed + 2)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.dirname(csv_path)

    out_small = os.path.join(out_dir, f"{stem}_small.csv")
    out_medium = os.path.join(out_dir, f"{stem}_medium.csv")
    out_large = os.path.join(out_dir, f"{stem}_large.csv")

    small_df.to_csv(out_small, index=False)
    medium_df.to_csv(out_medium, index=False)
    large_df.to_csv(out_large, index=False)

    print(f"saved: {out_small}")
    print(f"saved: {out_medium}")
    print(f"saved: {out_large}")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample small/medium/large subsets from each CSV (each subset sampled independently from the full dataset)."
    )
    parser.add_argument("--dir", type=str, default="../data_processing", help="directory containing CSV files")
    parser.add_argument("--small", type=float, default=0.25, help="small ratio")
    parser.add_argument("--medium", type=float, default=0.5, help="medium ratio")
    parser.add_argument("--large", type=float, default=1.0, help="large ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--replace", action="store_true",
                        help="sample with replacement within each subset (default: False)")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"can't find .csv file in {args.dir}")

    for f in csv_files:
        process_single_csv(
            f,
            small_ratio=args.small,
            medium_ratio=args.medium,
            large_ratio=args.large,
            seed=args.seed,
            replace=args.replace,
        )


if __name__ == "__main__":
    main()
