#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import BallTree

SCRIPT_DIR = Path(__file__).resolve().parent
RQ3_DIR = SCRIPT_DIR.parent
FALLBACK_ANALYSIS_DIR = RQ3_DIR.parent
PROJECT_ROOT = FALLBACK_ANALYSIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Adjust this import according to your project structure
from utils.config_manager import ConfigManager


# ============================================================
# User settings
# ============================================================

DATASET_ROOT = PROJECT_ROOT / "datasets"
PREDICTION_OUTPUT_DIR = RQ3_DIR / "results" / "predictions"

SYSTEMS = [
    "mysql",
    "postgresql",
    "httpd",
    "tomcat",
    "gcc",
    "clang",
    "x264",
    "x265",
]

SCALES = [
    "large",
]

# Current fallback strategies
FALLBACKS = [
    "worst",
    "1NN",
    "kNN",
    "kNNw",
    "loess",
    "rf",
    "rf_uncertainty",
    "dal_rf",
    # "dt",
    # "hgb",
    # "gp",
    # "tabpfn",
]

# Sparse ratios: percentage of the full tabular data used as observed data
TRAIN_FRACS = [0.05]

REPEATS = 10

# One repeat is an independent split. Use up to REPEATS workers by default.
MAX_WORKERS: Optional[int] = REPEATS

# Avoid nested parallelism when multiple repeat workers run at the same time.
SURROGATE_N_JOBS = 1

# k for kNN / kNNw
KNN_K = 20

MAXIMIZE_SYSTEMS = {"mysql", "postgresql", "httpd", "tomcat"}

ID_COLS = ["system", "scale", "train_frac", "repeat", "fallback"]
PREDICTION_ID_COLS = ID_COLS + ["true_perf", "pred_perf"]


# ============================================================
# Basic helpers
# ============================================================

def infer_optimization_goal(system: str) -> str:
    """
    Return the objective direction used by ConfigManager.
    """
    if system in MAXIMIZE_SYSTEMS:
        return "maximum"
    return "minimum"


# ============================================================
# Sparse ConfigManager construction
# ============================================================

def build_sparse_config_manager(
    system: str,
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> ConfigManager:
    """
    Build a ConfigManager whose observed table is train_df.

    Important:
    - train_df is used for fallback training and exact lookup.
    - full_df is used only to define the configuration domain / encoder categories.
      This does not leak performance values into the fallback model.
    """
    goal = infer_optimization_goal(system)
    cm = ConfigManager(system=system, optimization_goal=goal)

    train_df = train_df.copy().reset_index(drop=True)
    full_df = full_df.copy().reset_index(drop=True)

    # data_df must be train_df because several models, especially DaL, use self.data_df
    cm.data_df = train_df

    config_train = train_df.iloc[:, :-1]
    perf_train = train_df.iloc[:, -1].astype(float)

    config_full = full_df.iloc[:, :-1]

    cm.config_dim = config_full.shape[1]
    cm.config_space = config_full.drop_duplicates().values.tolist()

    cm.value_domains = []
    for col in config_full.columns:
        cm.value_domains.append(config_full[col].drop_duplicates().tolist())

    cm.config_perf_dict = {
        tuple(row[:-1]): float(row[-1])
        for row in train_df.values
    }

    cm.config_idx_dict = {}
    for idx, row in train_df.iterrows():
        key = tuple(row.iloc[:-1])
        cm.config_idx_dict.setdefault(key, []).append(idx)

    # Worst value should be computed only from observed sparse data
    if goal == "minimum":
        cm.worst_perf = float(perf_train.max())
    else:
        cm.worst_perf = float(perf_train.min())

    # Detect feature types and normalization factors using the full domain.
    # This uses only configuration values, not performance values.
    detect_feature_types_from_domain(cm, config_full)

    # Fit encoder using the full configuration domain.
    # This avoids shape mismatch when holdout contains a category absent in train_df.
    fit_encoder_from_domain(cm, config_full)

    cm.reset_surrogates()
    cm.reset_dal()
    cm.init_fallback_registry()

    return cm


def detect_feature_types_from_domain(cm: ConfigManager, config_df: pd.DataFrame) -> None:
    """
    Detect numerical/categorical types and normalization ranges from the full config domain.
    """
    cm.feature_types = []
    cm.norm_factors = []

    for col in config_df.columns:
        values = config_df[col].dropna().tolist()

        try:
            float_vals = list(map(float, values))
            unique_vals = set(float_vals)
        except Exception:
            cm.feature_types.append("categorical")
            cm.norm_factors.append(1.0)
            continue

        if unique_vals <= {0.0, 1.0} and len(unique_vals) <= 2:
            cm.feature_types.append("categorical")
            cm.norm_factors.append(1.0)
        else:
            cm.feature_types.append("numerical")
            max_val = max(float_vals)
            min_val = min(float_vals)
            norm = max_val - min_val
            cm.norm_factors.append(float(norm if norm != 0 else 1.0))


def fit_encoder_from_domain(cm: ConfigManager, config_df: pd.DataFrame) -> None:
    """
    Fit OrdinalEncoder on the full config domain, then build BallTree only for train configs.
    """
    full_configs = config_df.values.tolist()
    full_str_configs = [
        [cm.to_str_clean(v) for v in cfg]
        for cfg in full_configs
    ]

    cm.encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    cm.encoder.fit(full_str_configs)

    train_configs = list(cm.config_perf_dict.keys())
    train_str_configs = [
        [cm.to_str_clean(v) for v in cfg]
        for cfg in train_configs
    ]

    if len(train_str_configs) > 0:
        cm.encoded_configs = cm.encoder.transform(train_str_configs)
        cm.balltree = BallTree(cm.encoded_configs, metric="hamming")
    else:
        cm.encoded_configs = None
        cm.balltree = None


def prefit_fallback_if_needed(cm: ConfigManager, fallback: str, seed: int) -> None:
    """
    Pre-fit expensive models once per split, rather than inside every prediction.
    """
    try:
        if fallback in {"rf", "rf_uncertainty"}:
            cm.fit_random_forest(
                n_estimators=300,
                random_state=seed,
                n_jobs=SURROGATE_N_JOBS,
            )

        elif fallback == "dal_rf":
            cm.fit_dal_rf(
                depth=5,
                min_samples=10,
                seed=seed,
                use_mi_weight=True,
            )

        elif fallback == "dt":
            cm.fit_decision_tree(random_state=seed)

        elif fallback == "hgb":
            cm.fit_hgb(random_state=seed)

        elif fallback == "gp":
            cm.fit_gaussian_process(random_state=seed, max_train=5000)

    except Exception as e:
        print(f"[WARN] Failed to prefit {fallback}: {e}")


# ============================================================
# Split and evaluation
# ============================================================

def sample_sparse_split(
    df: pd.DataFrame,
    train_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly split full tabular data into sparse train and holdout.

    Assumption:
    each row corresponds to one configuration-performance record.
    """
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac should be in (0, 1), got {train_frac}")

    n = len(df)
    train_size = max(5, int(round(n * train_frac)))
    train_size = min(train_size, n - 1)

    rng = np.random.RandomState(seed)
    train_idx = rng.choice(n, size=train_size, replace=False)

    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True

    train_df = df.iloc[train_mask].copy().reset_index(drop=True)
    holdout_df = df.iloc[~train_mask].copy().reset_index(drop=True)

    return train_df, holdout_df


def predict_holdout(
    cm: ConfigManager,
    holdout_df: pd.DataFrame,
    fallback: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict holdout configurations using a given fallback.
    """
    cfgs = holdout_df.iloc[:, :-1].values.tolist()
    y_true = holdout_df.iloc[:, -1].astype(float).values

    y_pred = []

    for cfg in cfgs:
        try:
            # Directly use evaluate() to preserve your original logic:
            # exact lookup if observed, otherwise fallback.
            results, _ = cm.evaluate(
                [cfg],
                fallback=fallback,
                knn_k=KNN_K,
                lam=1.0,
            )
            pred = float(results[0][1])
        except Exception as e:
            print(f"[WARN] fallback={fallback} failed on cfg={cfg}: {e}")
            pred = np.nan

        y_pred.append(pred)

    return y_true, np.asarray(y_pred, dtype=float)


def run_one_dataset(
    system: str,
    scale: str,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run fallback prediction collection for one system-scale dataset.
    """
    prediction_rows = []

    for train_frac in TRAIN_FRACS:
        for repeat in range(REPEATS):
            seed = 1000 + repeat

            train_df, holdout_df = sample_sparse_split(
                full_df,
                train_frac=train_frac,
                seed=seed,
            )

            for fallback in FALLBACKS:
                print(
                    f"[RUN] system={system}, scale={scale}, "
                    f"train_frac={train_frac}, repeat={repeat}, fallback={fallback}"
                )

                cm = build_sparse_config_manager(
                    system=system,
                    train_df=train_df,
                    full_df=full_df,
                )

                prefit_fallback_if_needed(cm, fallback, seed)

                y_true, y_pred = predict_holdout(
                    cm=cm,
                    holdout_df=holdout_df,
                    fallback=fallback,
                )

                pred_df = holdout_df.copy()
                pred_df = pred_df.rename(columns={holdout_df.columns[-1]: "true_perf"})
                pred_df["pred_perf"] = y_pred
                pred_df.insert(0, "fallback", fallback)
                pred_df.insert(0, "repeat", repeat)
                pred_df.insert(0, "train_frac", train_frac)
                pred_df.insert(0, "scale", scale)
                pred_df.insert(0, "system", system)
                prediction_rows.append(pred_df)

    return (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )


def prediction_frame_from_arrays(
    system: str,
    scale: str,
    train_frac: float,
    repeat: int,
    fallback: str,
    holdout_df: pd.DataFrame,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    pred_df = holdout_df.copy()
    pred_df = pred_df.rename(columns={holdout_df.columns[-1]: "true_perf"})
    pred_df["pred_perf"] = y_pred
    pred_df.insert(0, "fallback", fallback)
    pred_df.insert(0, "repeat", repeat)
    pred_df.insert(0, "train_frac", train_frac)
    pred_df.insert(0, "scale", scale)
    pred_df.insert(0, "system", system)
    return pred_df


def run_one_repeat(
    system: str,
    scale: str,
    full_df: pd.DataFrame,
    train_frac: float,
    repeat: int,
) -> pd.DataFrame:
    seed = 1000 + repeat
    train_df, holdout_df = sample_sparse_split(
        full_df,
        train_frac=train_frac,
        seed=seed,
    )

    prediction_rows = []
    for fallback in FALLBACKS:
        print(
            f"[RUN] system={system}, scale={scale}, "
            f"train_frac={train_frac}, repeat={repeat}, fallback={fallback}"
        )

        cm = build_sparse_config_manager(
            system=system,
            train_df=train_df,
            full_df=full_df,
        )

        prefit_fallback_if_needed(cm, fallback, seed)

        _, y_pred = predict_holdout(
            cm=cm,
            holdout_df=holdout_df,
            fallback=fallback,
        )

        prediction_rows.append(
            prediction_frame_from_arrays(
                system=system,
                scale=scale,
                train_frac=train_frac,
                repeat=repeat,
                fallback=fallback,
                holdout_df=holdout_df,
                y_pred=y_pred,
            )
        )

    return (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )


def run_one_dataset_parallel(
    system: str,
    scale: str,
    full_df: pd.DataFrame,
    max_workers: Optional[int] = MAX_WORKERS,
) -> pd.DataFrame:
    jobs = [
        (train_frac, repeat)
        for train_frac in TRAIN_FRACS
        for repeat in range(REPEATS)
    ]
    if not jobs:
        return pd.DataFrame()

    worker_count = max_workers or min(REPEATS, len(jobs))
    worker_count = max(1, min(worker_count, len(jobs)))

    print(f"[INFO] Running {len(jobs)} repeat jobs with {worker_count} workers.")
    prediction_rows = []

    if worker_count == 1:
        for train_frac, repeat in jobs:
            repeat_df = run_one_repeat(system, scale, full_df, train_frac, repeat)
            print(
                f"[DONE] system={system}, scale={scale}, "
                f"train_frac={train_frac}, repeat={repeat}, rows={len(repeat_df)}"
            )
            if not repeat_df.empty:
                prediction_rows.append(repeat_df)
    else:
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_job = {
                    executor.submit(
                        run_one_repeat,
                        system,
                        scale,
                        full_df,
                        train_frac,
                        repeat,
                    ): (train_frac, repeat)
                    for train_frac, repeat in jobs
                }

                for future in as_completed(future_to_job):
                    train_frac, repeat = future_to_job[future]
                    try:
                        repeat_df = future.result()
                    except Exception as exc:
                        print(
                            f"[ERROR] repeat job failed: system={system}, scale={scale}, "
                            f"train_frac={train_frac}, repeat={repeat}: {exc}"
                        )
                        raise

                    print(
                        f"[DONE] system={system}, scale={scale}, "
                        f"train_frac={train_frac}, repeat={repeat}, rows={len(repeat_df)}"
                    )
                    if not repeat_df.empty:
                        prediction_rows.append(repeat_df)
        except (OSError, PermissionError) as exc:
            print(f"[WARN] Process pool unavailable ({exc}); falling back to sequential execution.")
            for train_frac, repeat in jobs:
                repeat_df = run_one_repeat(system, scale, full_df, train_frac, repeat)
                print(
                    f"[DONE] system={system}, scale={scale}, "
                    f"train_frac={train_frac}, repeat={repeat}, rows={len(repeat_df)}"
                )
                if not repeat_df.empty:
                    prediction_rows.append(repeat_df)

    return (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def save_prediction_outputs(prediction_df: pd.DataFrame, system: str, scale: str) -> None:
    """
    Save holdout configs, true performance, and fallback predictions.
    """
    if prediction_df.empty:
        return

    for fallback, fb_df in prediction_df.groupby("fallback", sort=False):
        pred_dir = PREDICTION_OUTPUT_DIR / fallback
        pred_dir.mkdir(parents=True, exist_ok=True)

        config_cols = [
            col for col in fb_df.columns
            if col not in set(PREDICTION_ID_COLS)
        ]
        out_cols = ID_COLS + config_cols + ["true_perf", "pred_perf"]
        out_cols = select_existing_columns(fb_df, out_cols)

        pred_path = pred_dir / f"{system}_{scale}_predictions.csv"
        fb_df[out_cols].to_csv(pred_path, index=False)
        print(f"[SAVE] {pred_path}")


def main():
    random.seed(0)
    np.random.seed(0)

    PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for system in SYSTEMS:
        for scale in SCALES:
            csv_path = DATASET_ROOT / system / f"{system}_{scale}.csv"

            if not csv_path.exists():
                print(f"[SKIP] Missing dataset: {csv_path}")
                continue

            print(f"\n==============================")
            print(f"Dataset: {csv_path}")
            print(f"==============================")

            full_df = pd.read_csv(csv_path)

            prediction_df = run_one_dataset_parallel(
                system=system,
                scale=scale,
                full_df=full_df,
            )

            if prediction_df.empty:
                continue

            save_prediction_outputs(prediction_df, system, scale)


if __name__ == "__main__":
    main()
