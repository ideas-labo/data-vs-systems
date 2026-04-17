import time
import random
import numpy as np


class LINEPoolTuner:
    """
    LINE: distance^2 seeding with mini-sampling, adapted to a pool-based tuning setting.

    Core idea:
    - Maintain a set of already evaluated configs.
    - At each iteration, sample a mini-batch from the remaining pool.
    - For each candidate, compute its distance to the nearest evaluated config.
    - Select the next config randomly with probability proportional to distance^2.
    - Evaluate it and repeat.

    Optimization:
    - Precompute per-dimension metadata (categorical vs numerical, numeric ranges)
    - Pre-encode pool configs into distance-friendly numeric representations
    """

    def __init__(self, config_manager, run, budget, logger=None, fallback=None):
        self.config_manager = config_manager
        self.run = run
        self.logger = logger
        self.fallback = fallback
        self.alive = budget

        # LINE parameter
        self.few = 64

        self.evaluated_configs = {}   # original config tuple -> perf
        self.all_results = []

        # distance metadata
        self.is_categorical = []
        self.numeric_ranges = []
        self.category_maps = []

    def tuning(self):
        start_time = time.time()

        pool_configs = self.get_pool_configs()
        if len(pool_configs) == 0:
            print("[INFO] Empty config pool, stopping.")
            self._finalize(start_time)
            return None

        if self.alive <= 0:
            print("[INFO] No budget available, stopping.")
            self._finalize(start_time)
            return None

        # Precompute metadata and encode all pool configs once
        self._prepare_distance_metadata()
        encoded_pool_configs = [self.encode_config(cfg) for cfg in pool_configs]

        # Keep aligned raw/encoded lists
        aligned_pool = list(zip(pool_configs, encoded_pool_configs))

        # Step 1: initialize with one config
        random.shuffle(aligned_pool)
        first_config, first_encoded = aligned_pool[0]

        evaluated_one, self.alive = self.config_manager.evaluate(
            [first_config], self.fallback, self.alive
        )

        if not evaluated_one:
            print("[WARN] Initialization evaluation returned empty result.")
            self._finalize(start_time)
            return None

        cfg, perf = evaluated_one[0]
        self.evaluated_configs[tuple(cfg)] = perf
        self.all_results.append((cfg, perf))

        try:
            print(f"[INFO] Evaluated initial config: {cfg} -> {perf:.4f}")
        except Exception:
            print(f"[INFO] Evaluated initial config: {cfg} -> {perf}")

        # Step 2: iterative distance^2 seeding
        while self.alive > 0:
            print(f"[Run {self.run}] | [LINE-Pool] Evaluations left: {self.alive}")

            remaining = [
                (raw_cfg, enc_cfg)
                for raw_cfg, enc_cfg in aligned_pool
                if tuple(raw_cfg) not in self.evaluated_configs
            ]

            if not remaining:
                print("[INFO] No unevaluated configs left in the pool, stopping early.")
                break

            # mini-sampling from remaining pool
            candidate_size = min(self.few, len(remaining))
            candidates = random.sample(remaining, candidate_size)

            chosen_encoded = [
                self.encode_config(list(cfg_key))
                for cfg_key in self.evaluated_configs.keys()
            ]

            # compute distance^2 weights
            weights = []
            for _, cand_encoded in candidates:
                min_dist = min(
                    self.config_distance_fast(cand_encoded, prev_encoded)
                    for prev_encoded in chosen_encoded
                )
                weights.append(min_dist ** 2)

            # handle degenerate case: all weights are zero
            if sum(weights) <= 0:
                next_config, _ = random.choice(candidates)
            else:
                next_config, _ = random.choices(candidates, weights=weights, k=1)[0]

            evaluated_one, self.alive = self.config_manager.evaluate(
                [next_config], self.fallback, self.alive
            )

            if not evaluated_one:
                print("[WARN] Evaluation returned empty result. Stopping to avoid infinite loop.")
                break

            cfg, perf = evaluated_one[0]
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))

            try:
                print(f"[INFO] Evaluated new config: {cfg} -> {perf:.4f}")
            except Exception:
                print(f"[INFO] Evaluated new config: {cfg} -> {perf}")

        self._finalize(start_time)
        return None

    # ------------------------------------------------------------------
    # Distance optimization
    # ------------------------------------------------------------------

    def _prepare_distance_metadata(self):
        """
        Precompute per-dimension metadata:
        - whether a dimension is categorical
        - numeric min/max range
        - categorical value -> index map
        """
        self.is_categorical = []
        self.numeric_ranges = []
        self.category_maps = []

        for domain in self.config_manager.value_domains:
            cat = any(isinstance(v, str) for v in domain)
            self.is_categorical.append(cat)

            if cat:
                cat_map = {v: idx for idx, v in enumerate(domain)}
                self.category_maps.append(cat_map)
                self.numeric_ranges.append((0.0, 1.0))  # placeholder
            else:
                self.category_maps.append(None)
                numeric_vals = [
                    float(v) for v in domain
                    if isinstance(v, (int, float, np.integer, np.floating))
                ]
                if len(numeric_vals) == 0:
                    self.numeric_ranges.append((0.0, 1.0))
                else:
                    self.numeric_ranges.append((min(numeric_vals), max(numeric_vals)))

    def encode_config(self, cfg):
        """
        Encode one config into a numeric-friendly representation:
        - categorical -> integer index
        - numerical   -> float
        """
        encoded = []
        for i, v in enumerate(cfg):
            if self.is_categorical[i]:
                encoded.append(self.category_maps[i][v])
            else:
                encoded.append(float(v))
        return encoded

    def config_distance_fast(self, cfg1, cfg2):
        """
        Mixed-type normalized distance on encoded configs:
        - categorical: 0 if same else 1
        - numeric: normalized absolute difference
        """
        total = 0.0
        dim = len(cfg1)

        for i, (v1, v2) in enumerate(zip(cfg1, cfg2)):
            if self.is_categorical[i]:
                total += 0.0 if v1 == v2 else 1.0
            else:
                lo, hi = self.numeric_ranges[i]
                if hi <= lo:
                    diff = 0.0
                else:
                    diff = abs(v1 - v2) / (hi - lo)
                total += diff

        return total / max(dim, 1)

    # ------------------------------------------------------------------
    # Pool utils
    # ------------------------------------------------------------------

    def get_pool_configs(self):
        """
        Get all configurations from config_manager.config_space.
        Convert each config to a Python list.
        """
        pool = self.config_manager.config_space

        if isinstance(pool, np.ndarray):
            if pool.ndim == 1:
                return [pool.tolist()]
            return [list(cfg) for cfg in pool.tolist()]

        processed_pool = []
        for cfg in pool:
            if isinstance(cfg, np.ndarray):
                processed_pool.append(cfg.tolist())
            else:
                processed_pool.append(list(cfg))
        return processed_pool

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self, start_time):
        if self.logger is not None:
            self.logger.log_all_results(self.all_results)
            end_time = time.time()
            self.logger.store_runtime_to_csv(end_time - start_time)