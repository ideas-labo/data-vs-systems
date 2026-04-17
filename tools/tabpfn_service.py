import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd
import json
from tabpfn import TabPFNRegressor


def to_str_clean(val):
    # 与你 ConfigManager 的 to_str_clean 对齐：保证 categorical 稳定
    if isinstance(val, float) and float(val).is_integer():
        return str(int(val))
    return str(val)


def build_design_matrix(df: pd.DataFrame):
    """
    这里采用最稳、可复现的编码策略：
    - numerical: float
    - categorical: one-hot (drop_first=True)
    和你主环境的 _design_matrix_from_configs 一致精神，但这里独立实现，避免跨环境依赖。
    """
    X_cols = df.columns[:-1]
    y_col = df.columns[-1]

    X_raw = df[X_cols].copy()

    # 自动判定每列是 numerical 还是 categorical（与你 detect_feature_types 的逻辑相似）
    feature_types = []
    for c in X_cols:
        vals = X_raw[c].values
        ok = True
        float_vals = []
        for v in vals:
            try:
                float_vals.append(float(v))
            except Exception:
                ok = False
                break
        if not ok:
            feature_types.append("categorical")
        else:
            uniq = set(float_vals)
            if uniq <= {0.0, 1.0} and len(uniq) <= 2:
                feature_types.append("categorical")
            else:
                feature_types.append("numerical")

    # one-hot（drop_first=True）
    mats = []
    meta = {"X_cols": list(X_cols), "feature_types": feature_types, "cat_levels": {}}

    for j, c in enumerate(X_cols):
        if feature_types[j] == "numerical":
            mats.append(X_raw[c].astype(float).to_numpy().reshape(-1, 1))
        else:
            s = X_raw[c].map(to_str_clean).astype(str)
            cats = sorted(s.unique().tolist())
            meta["cat_levels"][c] = cats
            # drop_first
            for cat in cats[1:]:
                mats.append((s == cat).astype(float).to_numpy().reshape(-1, 1))

    X = np.hstack(mats).astype(float)
    y = df[y_col].astype(float).to_numpy()

    return X, y, meta


def design_vector_from_cfg(cfg_list, meta):
    X_cols = meta["X_cols"]
    feature_types = meta["feature_types"]
    cat_levels = meta["cat_levels"]

    # cfg_list 长度 = len(X_cols)
    row = []
    for j, val in enumerate(cfg_list):
        if feature_types[j] == "numerical":
            row.append(float(val))
        else:
            c = X_cols[j]
            sval = to_str_clean(val)
            cats = cat_levels[c]
            for cat in cats[1:]:
                row.append(1.0 if sval == cat else 0.0)

    return np.asarray(row, dtype=float).reshape(1, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to scale csv")
    parser.add_argument("--cfg", required=True, help="comma-separated cfg values (raw)")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--max_train", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore_limits", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # cache key: based on csv path + max_train + seed (你也可以加更多参数)
    cache_key = f"{os.path.basename(args.csv)}__n{args.max_train}__s{args.seed}.pkl"
    cache_path = os.path.join(args.cache_dir, cache_key)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        model = obj["model"]
        meta = obj["meta"]
    else:
        df = pd.read_csv(args.csv)
        X, y, meta = build_design_matrix(df)

        # subsample
        n = X.shape[0]
        if n > args.max_train:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(n, size=args.max_train, replace=False)
            X_train, y_train = X[idx], y[idx]
        else:
            X_train, y_train = X, y

        model = TabPFNRegressor(
            device=args.device,
            ignore_pretraining_limits=args.ignore_limits,
        )
        model.fit(X_train, y_train)

        with open(cache_path, "wb") as f:
            pickle.dump({"model": model, "meta": meta}, f)

    # parse cfg
    cfg_vals = json.loads(args.cfg)

    if len(cfg_vals) != len(meta["X_cols"]):
        raise ValueError(
            f"Config length mismatch: got {len(cfg_vals)}, "
            f"expected {len(meta['X_cols'])}"
        )

    # 尝试把可转 float 的转 float（保持与你主框架一致）
    parsed = []
    for v in cfg_vals:
        try:
            parsed.append(float(v))
        except Exception:
            parsed.append(v)

    x = design_vector_from_cfg(parsed, meta)
    yhat = float(model.predict(x)[0])

    # stdout 输出一个纯数字，方便主进程读取
    sys.stdout.write(str(yhat))


if __name__ == "__main__":
    main()
