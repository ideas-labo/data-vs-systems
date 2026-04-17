import time
import math
import random
import numpy as np


class EZRPoolTuner:
    """
    Core logic:
    - Warm-start with a small number of randomly evaluated configs from config_space.
    - Maintain two groups: Best and Rest.
    - Use a mixed-type Naive Bayes style likelihood:
        * numeric features -> Gaussian likelihood
        * categorical features -> frequency with smoothing
    - Use EZR's acquisition function:
        * xploit
        * xplore
        * adapt (default)
    - Sequentially select one config from the remaining pool, evaluate it, and update Best/Rest.
    """

    def __init__(self, config_manager, run, budget, logger=None, fallback=None):
        self.config_manager = config_manager
        self.run = run
        self.logger = logger
        self.fallback = fallback
        self.alive = budget

        # EZR-style parameters (close to source defaults)
        self.start = 4
        self.stop = budget
        self.guess = 0.5       # cut = round(n ** guess)
        self.few = 50          # candidate window size control
        self.acq = "adapt"     # xploit | xplore | adapt
        self.k = 1             # low frequency Bayes hack
        self.m = 2             # low frequency Bayes hack
        self.eps = 1e-32

        self.evaluated_configs = {}   # config tuple -> perf
        self.all_results = []

        # explicit dynamic groups
        self.done = []   # list of (cfg, perf)
        self.best = []   # list of (cfg, perf)
        self.rest = []   # list of (cfg, perf)

    def tuning(self):
        start_time = time.time()

        pool_configs = self.get_pool_configs()
        if len(pool_configs) == 0:
            print("[INFO] Empty config pool, stopping.")
            self._finalize(start_time)
            return None

        # shuffle pool once, similar to EZR's initial shuffle(data.rows)
        random.shuffle(pool_configs)

        # Step 1: warm start
        init_size = min(self.start, len(pool_configs), self.alive)
        if init_size <= 0:
            print("[INFO] No budget for initialization, stopping.")
            self._finalize(start_time)
            return None

        init_configs = pool_configs[:init_size]
        evaluated_init, self.alive = self.config_manager.evaluate(
            init_configs, self.fallback, self.alive
        )

        if not evaluated_init:
            print("[WARN] Initialization evaluation returned empty result.")
            self._finalize(start_time)
            return None

        for cfg, perf in evaluated_init:
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))
            self.done.append((cfg, perf))

        # initialize best/rest from initial done
        self._rebuild_best_rest_from_done()

        if len(self.best) == 0 or len(self.rest) == 0:
            print("[INFO] Not enough data to form both Best and Rest after initialization.")
            self._finalize(start_time)
            return None

        # Step 2: active learning loop
        while self.alive > 0 and len(self.done) < self.stop:
            print(f"[Run {self.run}] | [EZR-Pool] Evaluations left: {self.alive}")

            # remaining pool
            todo = [cfg for cfg in pool_configs if tuple(cfg) not in self.evaluated_configs]
            if not todo:
                print("[INFO] No unevaluated configs left in the pool, stopping early.")
                break

            n = len(self.done)

            # closer to source: only rank a front window todo[:few*2]
            candidate_front = todo[: min(len(todo), self.few * 2)]
            if not candidate_front:
                print("[INFO] No candidates available in the active window, stopping.")
                break

            scored = []
            progress = min(1.0, n / max(1, self.stop))
            for cfg in candidate_front:
                b_log = self.like(cfg, self.best, nall=n, nh=2)
                r_log = self.like(cfg, self.rest, nall=n, nh=2)
                score = self.acquire(progress, b_log, r_log)
                scored.append((score, cfg))

            scored.sort(key=lambda x: x[0], reverse=True)
            best_config = scored[0][1]

            evaluated_one, self.alive = self.config_manager.evaluate(
                [best_config], self.fallback, self.alive
            )

            if not evaluated_one:
                print("[WARN] Evaluation returned empty result. Stopping to avoid infinite loop.")
                break

            cfg, perf = evaluated_one[0]
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))
            self.done.append((cfg, perf))

            # dynamic maintenance closer to source logic:
            # 1) add new point into best
            self.best.append((cfg, perf))

            # 2) sort best according to performance under current optimization goal
            self.best = self.sort_configs_by_perf(self.best)

            # 3) keep only top round(n^guess) in best, move weakest to rest
            cut = self.compute_cut(len(self.done))
            while len(self.best) > cut:
                weakest = self.best.pop(-1)
                self.rest.append(weakest)

            # 4) optional cleanup to avoid duplication across best/rest
            self._deduplicate_best_rest()

            try:
                print(f"[INFO] Evaluated new config: {cfg} -> {perf:.4f}")
            except Exception:
                print(f"[INFO] Evaluated new config: {cfg} -> {perf}")

        self._finalize(start_time)
        return None

    # ------------------------------------------------------------------
    # EZR-style acquisition
    # ------------------------------------------------------------------

    def acquire(self, p, b_log, r_log):
        """
        EZR acquisition from source:

            b,r = exp(b_log), exp(r_log)
            q = 0 if xploit else 1 if xplore else 1-p
            score = (b + r*q) / abs(b*q - r + eps)

        where p is the progress ratio n / Stop.
        """
        b = math.exp(b_log)
        r = math.exp(r_log)

        if self.acq == "xploit":
            q = 0.0
        elif self.acq == "xplore":
            q = 1.0
        else:  # adapt
            q = 1.0 - p

        return (b + r * q) / abs(b * q - r + self.eps)

    # ------------------------------------------------------------------
    # EZR-style mixed-type Naive Bayes likelihood
    # ------------------------------------------------------------------

    def like(self, cfg, group, nall=100, nh=2):
        """
        Compute log-likelihood of cfg belonging to a group.
        - categorical: frequency with smoothing
        - numeric: Gaussian likelihood
        """
        if len(group) == 0:
            return math.log(self.eps)

        prior = (len(group) + self.k) / (nall + self.k * nh)

        log_terms = [math.log(max(prior, self.eps))]

        num_features = len(cfg)
        for i in range(num_features):
            value = cfg[i]
            if value == "?":
                continue

            domain = self.config_manager.value_domains[i]
            col_values = [gcfg[i] for gcfg, _ in group if gcfg[i] != "?"]

            if len(col_values) == 0:
                continue

            if self.is_categorical_domain(domain, value):
                prob = self.sym_likelihood(value, col_values, prior)
            else:
                prob = self.num_likelihood(value, col_values)

            if prob > 0:
                log_terms.append(math.log(max(prob, self.eps)))

        return sum(log_terms)

    def sym_likelihood(self, value, values, prior):
        """
        Symbolic likelihood:
            (count(value) + m * prior) / (n + m)
        """
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1

        return (counts.get(value, 0) + self.m * prior) / (len(values) + self.m + self.eps)

    def num_likelihood(self, value, values):
        """
        Numeric Gaussian likelihood.
        """
        arr = np.asarray(values, dtype=float)
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sd = sd + self.eps

        nom = math.exp(-1.0 * ((float(value) - mu) ** 2) / (2 * sd * sd))
        denom = math.sqrt(2 * math.pi * sd * sd)
        out = nom / (denom + self.eps)
        return max(self.eps, min(1.0, out))

    # ------------------------------------------------------------------
    # Best / Rest maintenance
    # ------------------------------------------------------------------

    def _rebuild_best_rest_from_done(self):
        """
        Build best/rest from all currently evaluated data.
        This is used at initialization.
        """
        ordered = self.sort_configs_by_perf(self.done)
        cut = self.compute_cut(len(ordered))
        self.best = ordered[:cut]
        self.rest = ordered[cut:]

    def _deduplicate_best_rest(self):
        """
        Ensure no config appears in both best and rest.
        Prefer keeping it in best.
        """
        best_keys = {tuple(cfg) for cfg, _ in self.best}
        new_rest = []
        seen = set()

        for cfg, perf in self.rest:
            key = tuple(cfg)
            if key in best_keys:
                continue
            if key in seen:
                continue
            seen.add(key)
            new_rest.append((cfg, perf))

        self.rest = new_rest

        # also deduplicate best itself, keep first occurrence
        new_best = []
        seen = set()
        for cfg, perf in self.best:
            key = tuple(cfg)
            if key in seen:
                continue
            seen.add(key)
            new_best.append((cfg, perf))
        self.best = new_best

    def compute_cut(self, n):
        cut = int(round(n ** self.guess))
        cut = max(1, min(cut, n - 1)) if n >= 2 else 1
        return cut

    def sort_configs_by_perf(self, cfg_perf_list):
        """
        Sort configs from best to worst according to optimization goal.
        """
        reverse = self.config_manager.optimization_goal == "maximum"
        return sorted(cfg_perf_list, key=lambda x: x[1], reverse=reverse)

    # ------------------------------------------------------------------
    # Pool / feature utilities
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

    def is_categorical_domain(self, domain, value):
        """
        Decide whether a feature should be treated as categorical.
        """
        if isinstance(value, str):
            return True

        for d in domain:
            if isinstance(d, str):
                return True

        return False

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self, start_time):
        if self.logger is not None:
            self.logger.log_all_results(self.all_results)
            end_time = time.time()
            self.logger.store_runtime_to_csv(end_time - start_time)