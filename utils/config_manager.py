import os
import random
import pandas as pd
from typing import List, Tuple
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import subprocess
import json

from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression


class ConfigManager:
    def __init__(self, system: str, optimization_goal: str = "minimum"):
        self.balltree = None
        self.encoded_configs = None
        self.encoder = None
        self.config_perf_dict = None

        self.config_idx_dict = None
        self.system = system
        self.optimization_goal = optimization_goal
        self.config_dim = None
        self.config_space = []
        self.value_domains = []
        self.init_config_space()

        self.data_df = None
        self.worst_perf = None

        self.feature_types = []
        self.norm_factors = []

        # --- surrogate models ---
        self.rf_model = None
        self.gp_model = None
        self.gp_x_scaler = None
        self.gp_y_scaler = None
        self.dt_model = None
        self.hgb_model = None

        self.tabpfn_python = os.path.join(".venv_tabpfn", "bin", "python")
        self.tabpfn_cache_dir = os.path.join("cache", "tabpfn")

        # --- DaL components ---
        self.dal_enabled = False
        self.dal_depth = None
        self.dal_min_samples = None

        self.dal_cart = None  # DecisionTreeRegressor
        self.dal_division_ids = None  # np.array shape (n_samples,)
        self.dal_router = None  # RandomForestClassifier
        self.dal_local_models = {}  # dict: division_id -> regressor
        self.dal_feature_weights = None  # optional MI weights for routing
        self.dal_scaler_maxX = None  # optional scaling like DaL
        self.dal_scaler_maxY = None

    def init_config_space(self):
        system_path = os.path.join("datasets", self.system)
        csv_files = [f for f in os.listdir(system_path) if f.endswith(".csv")]

        if not csv_files:
            raise FileNotFoundError(f"[ConfigManager] No workload CSV found in {system_path}")

        sample_df = pd.read_csv(os.path.join(system_path, csv_files[0]))
        config_df = sample_df.iloc[:, :-1]
        self.config_dim = config_df.shape[1]

        self.config_space = config_df.drop_duplicates().values.tolist()

        self.value_domains = []
        for col in config_df.columns:
            unique_vals = config_df[col].unique().tolist()
            self.value_domains.append(unique_vals)

    def set_system_scale(self, scales):
        path = os.path.join("datasets", f"{self.system}", f"{scales}.csv")
        self.current_csv_path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ConfigManager] Scale CSV not found: {path}")

        self.data_df = pd.read_csv(path)
        self.precompute_worst_perf()
        self.config_perf_dict = {
            tuple(row[:-1]): row[-1]
            for row in self.data_df.values
        }

        # config -> index (this is important for checking our code correctness)
        self.config_idx_dict = {}
        for idx, row in self.data_df.iterrows():
            key = tuple(row.iloc[:-1])
            self.config_idx_dict.setdefault(key, []).append(idx)

        self.build_balltree()
        self.detect_feature_types()
        self.reset_surrogates()
        self.init_fallback_registry()
        self.reset_dal()
        self.fit_dal_rf(depth=5, min_samples=10, seed=0, use_mi_weight=True)

    def precompute_worst_perf(self):
        perf_values = self.data_df.iloc[:, -1]
        if self.optimization_goal == "minimum":
            self.worst_perf = perf_values.max()
        elif self.optimization_goal == "maximum":
            self.worst_perf = perf_values.min()
        else:
            raise ValueError(f"Unknown optimization goal: {self.optimization_goal}")

    def random_initialize_population(self, pop_size):
        seen = set()
        population = []
        attempts = 0
        max_attempts = pop_size * 10

        while len(population) < pop_size and attempts < max_attempts:
            config = [random.choice(values) for values in self.value_domains]
            config_tuple = tuple(config)
            if config_tuple not in seen:
                seen.add(config_tuple)
                population.append(config)
            attempts += 1

        if len(population) < pop_size:
            print(f"[WARN] Only generated {len(population)} unique configs out of requested {pop_size}")

        return population

    def sampling_configs(self, sampling_size, value_domains=None) -> List[List]:

        # adaptive value domains (PromiseTune use case)
        if value_domains is None:
            value_domains = self.value_domains

        seen = set()
        init_configs = []
        attempts = 0
        max_attempts = sampling_size * 10

        while len(init_configs) < sampling_size and attempts < max_attempts:
            config = [random.choice(values) for values in value_domains]
            config_tuple = tuple(config)
            if config_tuple not in seen:
                seen.add(config_tuple)
                init_configs.append(config)
            attempts += 1

        if len(init_configs) < sampling_size:
            print(f"[WARN] Only generated {len(init_configs)} unique configs out of requested {sampling_size}")

        return init_configs

    def maxmin_sampling_hamming(self, k: int) -> List[List]:
        # remove duplicates, keeping original types
        configs = list({tuple(cfg): cfg for cfg in self.config_space}.values())

        if len(configs) <= k:
            return configs

        # compute max-min sampling based on Hamming distance, transforming all config to strings
        configs_str = [list(map(str, cfg)) for cfg in configs]
        configs_str = [[self.to_str_clean(x) for x in cfg] for cfg in configs]
        config_space_str = np.array(configs_str)
        n = len(config_space_str)

        selected_indices = [np.random.randint(0, n)]

        while len(selected_indices) < k:
            selected_configs = config_space_str[selected_indices]
            remaining_indices = list(set(range(n)) - set(selected_indices))

            distances = []
            for idx in remaining_indices:
                candidate = config_space_str[idx]
                min_dist = np.min([
                    self.hamming_distance(candidate, sel) for sel in selected_configs
                ])
                distances.append((idx, min_dist))

            next_idx = max(distances, key=lambda x: x[1])[0]
            selected_indices.append(next_idx)

        # return the selected configurations with original types
        return [configs[idx] for idx in selected_indices]

    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
        return np.sum(a != b)

    def get_config_header(self):
        return self.data_df.columns[:].tolist()

    def build_balltree(self, sklearn=None):
        # transform config_space to string, then encode to integer array for BallTree
        self.encoder = OrdinalEncoder()
        str_configs = [[self.to_str_clean(v) for v in cfg] for cfg in self.config_perf_dict.keys()]
        self.encoded_configs = self.encoder.fit_transform(str_configs)  # shape: (n_samples, n_features)
        self.balltree = BallTree(self.encoded_configs, metric='hamming')

    def evaluate(self, config_list: List[List[float]], fallback='worst', alive=None,
                 knn_k: int = 200, knn_weighted: bool = False,
                 kernel: str = 'tricube', bandwidth=None, ridge: float = 1e-8, lam: float = 1.0):
        """
        fallback options：
          - 'worst'      worst observed performance + small random noise (0.001~0.009) to break ties
          - '1NN'        nearest neighbor performance
          - 'kNN'        average of k nearest neighbors
          - 'kNNw'       weighted average of k nearest neighbors (weight=1/(distance+eps))
          - 'loess'      local regression (kernel-weighted linear regression in the neighborhood)
        """
        if self.data_df is None or self.config_perf_dict is None:
            raise RuntimeError("Workload not set or config dict not built.")

        if not hasattr(self, 'fallback_registry'):
            self.init_fallback_registry()

        results = []
        for config in config_list:
            key = tuple(config)
            if key in self.config_perf_dict:
                perf = self.config_perf_dict[key]
            else:
                if fallback not in self.fallback_registry:
                    raise ValueError(f"Unknown fallback strategy: {fallback}")
                perf = self.fallback_registry[fallback](
                    config,
                    knn_k=knn_k,
                    kernel=kernel,
                    bandwidth=bandwidth,
                    ridge=ridge,
                    lam=lam,
                )
            results.append((config, perf))

            if alive is not None:
                alive -= 1
                if alive <= 0:
                    break

        return results, alive

    def to_str_clean(self, val):
        return str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)

    def detect_feature_types(self):
        """automatically detect feature types and normalization factors"""
        sample_cfgs = list(self.config_perf_dict.keys())
        zipped = list(zip(*sample_cfgs))

        self.feature_types = []
        self.norm_factors = []

        for values in zipped:
            try:
                float_vals = list(map(float, values))
                unique_vals = set(float_vals)
            except:
                # using categorical if any value can't be converted to float
                self.feature_types.append("categorical")
                self.norm_factors.append(1)
                continue

            # if there are only two unique values, and they are 0 and 1, treat as categorical (binary boolean)
            if unique_vals <= {0.0, 1.0} and len(unique_vals) <= 2:
                self.feature_types.append("categorical")
                self.norm_factors.append(1)
            else:
                # for the rest, if can be float, treat as numerical
                self.feature_types.append("numerical")
                max_val = max(float_vals)
                min_val = min(float_vals)
                norm = max_val - min_val
                self.norm_factors.append(norm if norm != 0 else 1)

    def mixed_distance(self, cfg1, cfg2):
        """support categorical and numerical mixed distance calculation (hamming+normalized L1)"""
        dist = 0
        for i, (v1, v2) in enumerate(zip(cfg1, cfg2)):
            if self.feature_types[i] == "categorical":
                dist += int(v1 != v2)
            elif self.feature_types[i] == "numerical":
                dist += abs(float(v1) - float(v2)) / self.norm_factors[i]
        return dist / len(cfg1)

    def find_nearest_by_mixed_distance(self, config):
        """search for the nearest configuration based on mixed distance"""
        min_dist = float("inf")
        nearest = None
        for candidate in self.config_perf_dict.keys():
            d = self.mixed_distance(config, candidate)
            if d < min_dist:
                min_dist = d
                nearest = candidate
        return nearest

    def find_k_nearest_by_mixed_distance(self, config, k: int = 20):
        """
        Return the k nearest neighbors based on mixed_distance.
        Each item: (candidate_cfg_tuple, distance, perf)
        """
        # guard
        if self.config_perf_dict is None:
            raise RuntimeError("system not set or config dict not built.")

        dists = []
        for candidate in self.config_perf_dict.keys():
            d = self.mixed_distance(config, candidate)
            dists.append((candidate, d, self.config_perf_dict[candidate]))

        # sort by distance and take top-k
        dists.sort(key=lambda x: x[1])
        return dists[:min(k, len(dists))]

    def knn_predict_perf(self, config, k: int = 20, weighted: bool = False, eps: float = 1e-9):
        """
        Use k-NN to estimate performance.
        - weighted=False: simple mean of neighbor perfs
        - weighted=True: inverse-distance weighted mean (1/(d+eps))
        """
        neighbors = self.find_k_nearest_by_mixed_distance(config, k)
        if not neighbors:
            # fallback to worst if something is wrong
            return self.worst_perf + np.random.uniform(0.001, 0.009)

        if not weighted:
            # uniform mean
            return float(np.mean([p for _, _, p in neighbors]))
        else:
            weights = [1.0 / (d + eps) for _, d, _ in neighbors]
            perfs = [p for _, _, p in neighbors]
            wsum = np.sum(weights)
            if wsum == 0:
                return float(np.mean(perfs))
            return float(np.dot(perfs, weights) / wsum)

    def _kernel_weights(self, dists, bandwidth=None, kernel='tricube', eps=1e-12):
        dists = np.asarray(dists, dtype=float)
        if bandwidth is None:
            # adaptive bandwidth: normalize by max distance (or a quantile)
            bandwidth = np.max(dists) + eps
        u = dists / max(bandwidth, eps)
        if kernel == 'tricube':
            w = np.where(u < 1.0, (1 - u ** 3) ** 3, 0.0)
        elif kernel == 'gaussian':
            w = np.exp(-0.5 * (u ** 2))
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        # make sure weights are non-negative and not all zero
        if np.all(w == 0):
            w = np.ones_like(w)
        return w

    def get_encoded_dim(self):
        any_cfg = next(iter(self.config_perf_dict.keys()))
        return len(self._design_vector_from_config(list(any_cfg)))

    def get_default_loess_k(self, factor=5, k_min=50, k_max=None):
        p = self.get_encoded_dim()
        k = max(k_min, factor * p)
        if k_max is not None:
            k = min(k, k_max)
        return k

    def local_regression_predict(self, config, k=None, kernel='tricube', bandwidth=None, ridge=1e-8):
        """
        Loess: weighted local regression. Fit a linear model in the neighborhood of the target point, using kernel weights.
        - k: No. of neighbors
        - kernel: 'tricube' | 'gaussian'
        - bandwidth: distance bandwidth for kernel weighting (if None, use max distance among neighbors for adaptive bandwidth)
        - ridge: regularization strength for ridge regression (default: 1e-8)
        """
        if k is None:
            k = self.get_default_loess_k(factor=10, k_min=50)

        neighbors = self.find_k_nearest_by_mixed_distance(config, k)
        if not neighbors:
            return self.worst_perf + np.random.uniform(0.001, 0.009)

        cfgs = [cfg for cfg, _, _ in neighbors]
        y = np.asarray([perf for _, _, perf in neighbors], dtype=float)
        d = np.asarray([dist for _, dist, _ in neighbors], dtype=float)

        X = self._design_matrix_from_configs(cfgs)
        x0 = self._design_vector_from_config(config)

        w = self._kernel_weights(d, bandwidth=bandwidth, kernel=kernel)
        W_sqrt = np.sqrt(w + 1e-12)

        # weighted regression (X^T W X + λI)β = X^T W y
        Xw = X * W_sqrt[:, None]
        yw = y * W_sqrt

        XtX = Xw.T @ Xw
        if ridge > 0:
            XtX = XtX + ridge * np.eye(XtX.shape[0], dtype=float)

        try:
            beta = np.linalg.solve(XtX, Xw.T @ yw)
            yhat = float(x0 @ beta)

            print("y range:", y.min(), y.max(), "yhat:", yhat)
            print("X shape:", X.shape, "cond:", np.linalg.cond(XtX))
            print("beta norm:", np.linalg.norm(beta))

        except np.linalg.LinAlgError:
            # fallback to kNN mean if regression fails (e.g. singular matrix)
            yhat = float(np.mean(y))
        return yhat

    def fit_random_forest(self, n_estimators=300, random_state=0, n_jobs=-1, **kwargs):
        X, y = self._get_surrogate_train_data()
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        self.rf_model.fit(X, y)

    def predict_rf(self, cfg, auto_fit=True):
        if self.rf_model is None:
            if not auto_fit:
                raise RuntimeError("RF model is not trained. Call fit_random_forest() first.")
            self.fit_random_forest()
        x = self._featurize_one(cfg).reshape(1, -1)
        return float(self.rf_model.predict(x)[0])

    def predict_rf_mean_std(self, cfg, auto_fit=True):
        """
        Return (mean, std) using tree disagreement as uncertainty proxy.
        std = std of individual tree predictions.
        Reuses self.rf_model (no duplicate fit logic).
        """
        if self.rf_model is None:
            if not auto_fit:
                raise RuntimeError("RF model is not trained.")
            self.fit_random_forest()

        x = self._design_vector_from_config(cfg).reshape(1, -1)
        preds = np.asarray([est.predict(x)[0] for est in self.rf_model.estimators_], dtype=float)
        return float(preds.mean()), float(preds.std(ddof=0))

    def predict_rf_uncertainty_aware(self, cfg, lam=1.0, auto_fit=True):
        """
        Risk-aware RF prediction:
          - maximum goal:  mean - lam * std
          - minimum goal:  mean + lam * std
        """
        mu, sigma = self.predict_rf_mean_std(cfg, auto_fit=auto_fit)

        if self.optimization_goal == "maximum":
            return mu - lam * sigma
        elif self.optimization_goal == "minimum":
            return mu + lam * sigma
        else:
            raise ValueError(f"Unknown optimization goal: {self.optimization_goal}")


    def fit_gaussian_process(self, max_train=20000, random_state=0, **kwargs):

        X, y = self._get_surrogate_train_data()

        # Subsample to avoid O(n^3) blowup
        if X.shape[0] > max_train:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(X.shape[0], size=max_train, replace=False)
            X_train, y_train = X[idx], y[idx]
        else:
            X_train, y_train = X, y

        # Standardize X and y for stability
        self.gp_x_scaler = StandardScaler()
        self.gp_y_scaler = StandardScaler()

        Xs = self.gp_x_scaler.fit_transform(X_train)
        ys = self.gp_y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

        # kernel: Constant * RBF + noise
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)

        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,  # we already scaled y
            random_state=random_state,
            **kwargs
        )
        self.gp_model.fit(Xs, ys)

    def predict_gp(self, cfg, return_std=False, auto_fit=True):
        if self.gp_model is None:
            if not auto_fit:
                raise RuntimeError("GP model is not trained. Call fit_gaussian_process() first.")
            self.fit_gaussian_process()

        x = self._featurize_one(cfg).reshape(1, -1)
        xs = self.gp_x_scaler.transform(x)

        if return_std:
            mean_s, std_s = self.gp_model.predict(xs, return_std=True)
            mean = self.gp_y_scaler.inverse_transform(mean_s.reshape(-1, 1))[0, 0]
            std = float(std_s[0] * self.gp_y_scaler.scale_[0])  # unscale std
            return float(mean), std
        else:
            mean_s = self.gp_model.predict(xs)
            mean = self.gp_y_scaler.inverse_transform(mean_s.reshape(-1, 1))[0, 0]
            return float(mean)

    # ---------------- Decision Tree surrogate ----------------
    def fit_decision_tree(self, random_state=0, **kwargs):
        cfgs = [list(cfg) for cfg in self.config_perf_dict.keys()]
        y = np.asarray([self.config_perf_dict[cfg] for cfg in self.config_perf_dict.keys()], dtype=float)
        X = self._design_matrix_from_configs(cfgs)

        self.dt_model = DecisionTreeRegressor(
            random_state=random_state,
            **kwargs
        )
        self.dt_model.fit(X, y)

    def predict_dt(self, cfg, auto_fit=True):
        if not hasattr(self, "dt_model") or self.dt_model is None:
            if not auto_fit:
                raise RuntimeError("DT model not trained.")
            self.fit_decision_tree()

        x = self._design_vector_from_config(cfg).reshape(1, -1)
        return float(self.dt_model.predict(x)[0])

    # ---------------- HistGradientBoosting surrogate ----------------
    def fit_hgb(self,
                random_state: int = 0,
                max_iter: int = 500,
                learning_rate: float = 0.05,
                max_depth: int = None,
                max_leaf_nodes: int = 31,
                min_samples_leaf: int = 20,
                l2_regularization: float = 0.0,
                **kwargs):
        """
        Train HistGradientBoostingRegressor on the whole table.
        Notes:
          - Works well on tabular data.
          - Does NOT natively output uncertainty.
        """
        cfgs = [list(cfg) for cfg in self.config_perf_dict.keys()]
        y = np.asarray([self.config_perf_dict[cfg] for cfg in self.config_perf_dict.keys()], dtype=float)
        X = self._design_matrix_from_configs(cfgs)

        self.hgb_model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
            **kwargs
        )
        self.hgb_model.fit(X, y)

    def predict_hgb(self, cfg, auto_fit: bool = True):
        """
        Predict with HGB surrogate.
        """
        if self.hgb_model is None:
            if not auto_fit:
                raise RuntimeError("HGB model is not trained. Call fit_hgb() first.")
            self.fit_hgb()

        x = self._design_vector_from_config(cfg).reshape(1, -1)
        return float(self.hgb_model.predict(x)[0])

    # ---------------- TabPFN surrogate ----------------
    def predict_tabpfn_subprocess(self, cfg, max_train=20000, device="cpu", seed=0, ignore_limits=True):
        """
        independent subprocess call to TabPFN predictor, to avoid dependency issues and environment pollution.
        """
        if self.data_df is None:
            raise RuntimeError("Workload not set (call set_system_scale first).")

        csv_path = getattr(self, "current_csv_path", None)
        if csv_path is None:
            raise RuntimeError("current_csv_path not found. Set self.current_csv_path in set_system_scale().")

        py = self.tabpfn_python
        script = os.path.join("tools", "tabpfn_service.py")

        # cfg is connected by comma, making sure it won't contain comma in the values (e.g. categorical values should not have comma)
        cfg_str = json.dumps([self.to_str_clean(v) for v in cfg])

        cmd = [
            py, script,
            "--csv", csv_path,
            "--cfg", cfg_str,
            "--cache_dir", self.tabpfn_cache_dir,
            "--max_train", str(int(max_train)),
            "--device", str(device),
            "--seed", str(int(seed)),
        ]
        if ignore_limits:
            cmd.append("--ignore_limits")

        out = subprocess.check_output(cmd, text=True).strip()
        return float(out)

    def _dal_recursive_dividing(self, node, depth, tree_, X, samples,
                                max_depth=2, min_samples=10, out=None):

        if out is None:
            out = []

        if depth <= max_depth:
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                feat = tree_.feature[node]
                thr = tree_.threshold[node]
                left, right = [], []
                for idx in samples:
                    if X[idx, feat] <= thr:
                        left.append(idx)
                    else:
                        right.append(idx)

                if (len(left) < min_samples) or (len(right) < min_samples):
                    out.append(samples)
                else:
                    self._dal_recursive_dividing(tree_.children_left[node], depth + 1, tree_, X, left,
                                                 max_depth, min_samples, out)
                    self._dal_recursive_dividing(tree_.children_right[node], depth + 1, tree_, X, right,
                                                 max_depth, min_samples, out)
            else:
                out.append(samples)
        else:
            out.append(samples)

        return out

    def fit_dal_rf(
            self,
            depth: int = 2,
            min_samples: int = 10,
            router_estimators: int = 300,
            local_estimators: int = 300,
            use_mi_weight: bool = True,
            seed: int = 0,
    ):
        """
        RF-only DaL:
          1) CART divisions
          2) RF router: X -> division_id (optionally MI-weighted)
          3) Local RF regressor per division
        """

        if self.data_df is None:
            raise RuntimeError("Call set_system_scale() before fit_dal_rf().")

        self.reset_dal()

        X, y, _ = self._get_full_Xy()
        n = X.shape[0]
        all_idx = list(range(n))

        # ---- DaL scaling (author-style) ----
        self._dal_fit_scalers(X, y)
        Xs = self._dal_transform_X(X)
        ys = self._dal_transform_y(y)

        # 1) CART for division
        cart = DecisionTreeRegressor(
            random_state=seed,
            criterion="squared_error",
            splitter="best"
        )
        cart.fit(Xs, ys)
        tree_ = cart.tree_

        clusters = self._dal_recursive_dividing(
            node=0, depth=1, tree_=tree_, X=Xs, samples=all_idx,
            max_depth=depth, min_samples=min_samples, out=[]
        )

        if len(clusters) <= 1:
            # division failed: disable DaL and fallback to global RF
            self.reset_dal()
            return

        division_ids = np.full(n, -1, dtype=int)
        for cid, idx_list in enumerate(clusters):
            for idx in idx_list:
                division_ids[idx] = cid

        # 2) Router input (optional MI weight)
        X_router = Xs
        if use_mi_weight:
            w = mutual_info_regression(Xs, ys, random_state=seed)
            w = np.where(w == 0, 1.0, w)
            X_router = X_router * w[None, :]
            self.dal_feature_weights = w
        else:
            self.dal_feature_weights = None

        router = RandomForestClassifier(
            n_estimators=router_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            criterion="gini",
        )
        router.fit(X_router, division_ids)

        # 3) Local RF models
        local_models = {}
        for cid, idx_list in enumerate(clusters):
            Xc = Xs[idx_list]
            yc = ys[idx_list]

            # if division is too small, local RF will be unstable: skip and fallback to global RF during prediction
            if len(idx_list) < max(5, min_samples):
                continue

            model = RandomForestRegressor(
                n_estimators=local_estimators,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(Xc, yc)
            local_models[cid] = model

        self.dal_cart = cart
        self.dal_division_ids = division_ids
        self.dal_router = router
        self.dal_local_models = local_models
        self.dal_depth = depth
        self.dal_min_samples = min_samples
        self.dal_enabled = True

    def _dal_fit_scalers(self, X: np.ndarray, y: np.ndarray):

        max_X = np.amax(X, axis=0)
        max_X = np.where(max_X == 0, 1.0, max_X)

        max_Y = float(np.max(y)) / 100.0
        if max_Y == 0:
            max_Y = 1.0

        self.dal_scaler_maxX = max_X
        self.dal_scaler_maxY = max_Y

    def _dal_transform_X(self, X: np.ndarray) -> np.ndarray:
        if self.dal_scaler_maxX is None:
            raise RuntimeError("DaL scalers not fitted.")
        return np.divide(X, self.dal_scaler_maxX)

    def _dal_transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.dal_scaler_maxY is None:
            raise RuntimeError("DaL scalers not fitted.")
        return np.divide(y, self.dal_scaler_maxY)

    def _dal_inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.dal_scaler_maxY is None:
            raise RuntimeError("DaL scalers not fitted.")
        return y_scaled * self.dal_scaler_maxY

    def dal_predict_rf(self, cfg):
        """
        Predict with RF-only DaL. If router is uncertain / missing local model -> fallback to global RF.
        """

        # guard: if DaL not properly set up, fallback to global RF
        if (not self.dal_enabled) or (self.dal_router is None) or (not self.dal_local_models):
            return self.predict_rf(cfg, auto_fit=True)

        x = self._design_vector_from_config(cfg).reshape(1, -1)
        xs = self._dal_transform_X(x)

        # router input * MI weights
        if self.dal_feature_weights is not None:
            x_router = xs * self.dal_feature_weights.reshape(1, -1)
        else:
            x_router = xs

        cid = int(self.dal_router.predict(x_router)[0])
        model = self.dal_local_models.get(cid, None)
        if model is None:
            return self.predict_rf(cfg, auto_fit=True)

        yhat_scaled = float(model.predict(xs)[0])
        yhat = float(self._dal_inverse_y(np.array([yhat_scaled]))[0])
        return yhat

    def init_fallback_registry(self):
        self.fallback_registry = {
            'worst': lambda cfg, **kw: self.worst_perf + np.random.uniform(0.001, 0.009),
            '1NN': lambda cfg, **kw: self.config_perf_dict[self.find_nearest_by_mixed_distance(cfg)],
            'kNN': lambda cfg, **kw: self.knn_predict_perf(cfg, k=kw.get('knn_k', 20), weighted=False),
            'kNNw': lambda cfg, **kw: self.knn_predict_perf(cfg, k=kw.get('knn_k', 20), weighted=True),
            'loess': lambda cfg, **kw: self.local_regression_predict(
                cfg,
                None,
                kernel=kw.get('kernel', 'tricube'),
                bandwidth=kw.get('bandwidth', None),
                ridge=kw.get('ridge', 1e-8),
            ),

            # --- NEW: surrogate models ---
            'rf': lambda cfg, **kw: self.predict_rf(cfg, auto_fit=True),
            'gp': lambda cfg, **kw: self.predict_gp(cfg, return_std=False, auto_fit=True),
            'rf_uncertainty': lambda cfg, **kw: self.predict_rf_uncertainty_aware(cfg, lam=kw.get('lam', 1.0)),
            'dt': lambda cfg, **kw: self.predict_dt(cfg),
            'hgb': lambda cfg, **kw: self.predict_hgb(cfg, auto_fit=True),
            'tabpfn': lambda cfg, **kw: self.predict_tabpfn_subprocess(
                cfg,
                max_train=kw.get("max_train", 1000),
                device=kw.get("device", "cpu"),
                seed=kw.get("seed", 0),
                ignore_limits=kw.get("ignore_pretraining_limits", True),
            ),
            'dal_rf': lambda cfg, **kw: self.dal_predict_rf(cfg),

        }

    def _design_matrix_from_configs(self, configs):
        """
    Convert the original configuration list into a numerical feature matrix:
    - Numerical: Directly convert to float (normalize if needed)
    - Categorical: Perform one-hot encoding based on self.encoder.categories_ (drop_first=True to avoid multicollinearity)
    Returns: X (n_samples, n_features_expanded)
    """
        if self.encoder is None or not hasattr(self.encoder, 'categories_'):
            # The encoder has already been fitted in build_balltree()
            raise RuntimeError("Encoder not initialized. Ensure build_balltree() has been called.")

        X = []
        # Pre-fetch the list of categories for each categorical feature
        cat_lists = [list(cats) for cats in self.encoder.categories_]

        for cfg in configs:
            row = []
            for j, val in enumerate(cfg):
                if self.feature_types[j] == "numerical":
                    row.append(float(val))
                else:
                   # Categorical -> one-hot
                    sval = self.to_str_clean(val)
                    cats = cat_lists[j]
                    # drop_first: only encode categories starting from index=1
                    for idx in range(1, len(cats)):
                        row.append(1.0 if sval == cats[idx] else 0.0)
            X.append(row)
        return np.asarray(X, dtype=float)

    def _design_vector_from_config(self, cfg):
        """Same logic as _design_matrix_from_configs, single-sample version; transforms the configuration to be predicted"""
        if self.encoder is None or not hasattr(self.encoder, 'categories_'):
            raise RuntimeError("Encoder not initialized.")
        row = []
        cat_lists = [list(cats) for cats in self.encoder.categories_]
        for j, val in enumerate(cfg):
            if self.feature_types[j] == "numerical":
                row.append(float(val))
            else:
                sval = self.to_str_clean(val)
                cats = cat_lists[j]
                for idx in range(1, len(cats)):
                    row.append(1.0 if sval == cats[idx] else 0.0)
        return np.asarray(row, dtype=float)

    def _get_surrogate_train_data(self):
        """
        Build X, y for surrogate training using the CURRENT dataset (self.config_perf_dict).
        X: one-hot + numeric design matrix
        y: performance vector
        """
        if self.config_perf_dict is None:
            raise RuntimeError("config_perf_dict is None. Call set_system_scale first.")

        cfgs = [list(cfg) for cfg in self.config_perf_dict.keys()]
        y = np.asarray([self.config_perf_dict[cfg] for cfg in self.config_perf_dict.keys()], dtype=float)
        X = self._design_matrix_from_configs(cfgs)
        return X, y

    def _featurize_one(self, cfg):
        """single config -> feature vector (consistent with _design_matrix_from_configs)"""
        return self._design_vector_from_config(cfg)

    def _get_full_Xy(self):
      # Full table configs (from the already loaded data_df)
        cfgs = [list(row[:-1]) for row in self.data_df.values]
        y = np.asarray([row[-1] for row in self.data_df.values], dtype=float)
        X = self._design_matrix_from_configs(cfgs)
        return X, y, cfgs

    def reset_dal(self):
        self.dal_enabled = False
        self.dal_depth = None
        self.dal_min_samples = None
        self.dal_cart = None
        self.dal_division_ids = None
        self.dal_router = None
        self.dal_local_models = {}
        self.dal_feature_weights = None
        self.dal_scaler_maxX = None
        self.dal_scaler_maxY = None

    def reset_surrogates(self):
        """call this when dataset changes (e.g., new scale/workload)"""
        self.rf_model = None
        self.gp_model = None
        self.gp_x_scaler = None
        self.gp_y_scaler = None
        self.dt_model = None
        self.hgb_model = None
