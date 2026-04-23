import time
import random
import numpy as np


class LINETuner:
    """
    Non-pool sampled LINE.

    Core idea:
    - Maintain a set of already evaluated configs.
    - At each iteration, sample a mini-batch from the configuration space.
    - For each candidate, compute its distance to the nearest evaluated config.
    - Select the next config randomly with probability proportional to distance^2.
    - Evaluate it and repeat.

    This version is suitable for real-system tuning where the full
    configuration space cannot be enumerated as a pool.
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

        # encoded evaluated configs for distance computation
        self.evaluated_encoded_configs = []

        # distance metadata
        self.is_categorical = []
        self.numeric_ranges = []
        self.category_maps = []

    # ============================================================
    # Main
    # ============================================================

    def tuning(self):
        start_time = time.time()

        if self.alive <= 0:
            print("[INFO] No budget available, stopping.")
            self._finalize(start_time)
            return None

        # Prepare distance metadata once
        self._prepare_distance_metadata()

        # --------------------------------------------------------
        # Step 1: initialize with one sampled config
        # --------------------------------------------------------
        init_configs = self.sample_unevaluated_configs(1)
        if len(init_configs) == 0:
            print("[INFO] No initial config can be sampled, stopping.")
            self._finalize(start_time)
            return None

        first_config = init_configs[0]

        evaluated_one, self.alive = self.config_manager.evaluate(
            [first_config], self.fallback, self.alive
        )

        if not evaluated_one:
            print("[WARN] Initialization evaluation returned empty result.")
            self._finalize(start_time)
            return None

        cfg, perf = evaluated_one[0]
        self.evaluated_configs[tuple(cfg)] = perf
        self.evaluated_encoded_configs.append(self.encode_config(cfg))
        self.all_results.append((cfg, perf))

        try:
            print(f"[INFO] Evaluated initial config: {cfg} -> {perf:.4f}")
        except Exception:
            print(f"[INFO] Evaluated initial config: {cfg} -> {perf}")

        # --------------------------------------------------------
        # Step 2: iterative distance^2 seeding
        # --------------------------------------------------------
        while self.alive > 0:
            print(f"[Run {self.run}] | [LINE] Evaluations left: {self.alive}")

            candidates = self.sample_unevaluated_configs(self.few)
            if len(candidates) == 0:
                print("[INFO] No unevaluated sampled configs available, stopping early.")
                break

            encoded_candidates = [(cfg, self.encode_config(cfg)) for cfg in candidates]

            # compute distance^2 weights
            weights = []
            for _, cand_encoded in encoded_candidates:
                min_dist = min(
                    self.config_distance_fast(cand_encoded, prev_encoded)
                    for prev_encoded in self.evaluated_encoded_configs
                )
                weights.append(min_dist ** 2)

            # handle degenerate case: all weights are zero
            if sum(weights) <= 0:
                next_config, _ = random.choice(encoded_candidates)
            else:
                next_config, _ = random.choices(encoded_candidates, weights=weights, k=1)[0]

            evaluated_one, self.alive = self.config_manager.evaluate(
                [next_config], self.fallback, self.alive
            )

            if not evaluated_one:
                print("[WARN] Evaluation returned empty result. Stopping to avoid infinite loop.")
                break

            cfg, perf = evaluated_one[0]
            self.evaluated_configs[tuple(cfg)] = perf
            self.evaluated_encoded_configs.append(self.encode_config(cfg))
            self.all_results.append((cfg, perf))

            try:
                print(f"[INFO] Evaluated new config: {cfg} -> {perf:.4f}")
            except Exception:
                print(f"[INFO] Evaluated new config: {cfg} -> {perf}")

        self._finalize(start_time)
        return None

    # ============================================================
    # Sampling helpers
    # ============================================================

    def sample_unevaluated_configs(self, target_n, max_rounds=10):
        """
        Sample configs from the configuration space and keep only:
        - unevaluated ones
        - unique ones within this batch

        Since this is the non-pool version, we may need to try multiple
        rounds to collect enough fresh candidates.
        """
        collected = []
        seen = set()

        rounds = 0
        while len(collected) < target_n and rounds < max_rounds:
            rounds += 1

            sampled = self.config_manager.sampling_configs(target_n)
            for cfg in sampled:
                key = tuple(cfg)
                if key in self.evaluated_configs:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                collected.append(cfg)
                if len(collected) >= target_n:
                    break

        return collected

    # ============================================================
    # Distance optimization
    # ============================================================

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

    # ============================================================
    # Finalization
    # ============================================================

    def _finalize(self, start_time):
        if self.logger is not None:
            self.logger.log_all_results(self.all_results)
            end_time = time.time()
            self.logger.store_runtime_to_csv(end_time - start_time)