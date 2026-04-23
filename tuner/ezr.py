import time
import math
import random
import numpy as np
from collections import Counter


class EZRTuner:
    """
    Non-pool sampled EZR.

    Main idea:
    - Warm start with a few randomly sampled configurations.
    - Maintain labeled data in two groups: best and rest.
    - At each iteration, sample a batch of candidate configurations.
    - Select one candidate from that sampled batch using:
        * nearer()  : distance-based selection
        * likelier(): likelihood-based acquisition
    - Evaluate the selected configuration and update best/rest.

    This version is suitable for real-system tuning where the full
    configuration space cannot be enumerated as a pool.
    """

    def __init__(self, config_manager, run, budget, logger=None, fallback=None):
        self.config_manager = config_manager
        self.run = run
        self.logger = logger
        self.fallback = fallback
        self.alive = budget

        # EZR-style parameters
        self.start = 4               # number of initial random evaluations
        self.stop = budget           # total budget
        self.few = 128               # candidate window size
        self.sampling_size = 1000    # sampled candidate batch size
        self.acq = "adapt"           # near | xploit | xplor | bore | adapt
        self.m = 1                   # Bayes low-frequency param
        self.p = 2                   # distance coefficient
        self.eps = 1e-32

        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []

        self.done = []               # list of (cfg, perf)
        self.best = []               # list of (cfg, perf)
        self.rest = []               # list of (cfg, perf)

    # ============================================================
    # Main
    # ============================================================

    def tuning(self):
        start_time = time.time()

        # --------------------------------------------------------
        # Step 1: warm start
        # --------------------------------------------------------
        init_configs = self.config_manager.sampling_configs(self.start)
        init_configs = self.filter_unevaluated(init_configs)

        if len(init_configs) == 0:
            print("[INFO] No initial configs available, stopping.")
            self._finalize(start_time)
            return None

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

        # sort and split into best/rest
        self.done = self.sort_configs_by_perf(self.done)
        self.rebuild_best_rest()

        if len(self.best) == 0:
            print("[INFO] Empty best set after initialization, stopping.")
            self._finalize(start_time)
            return None

        # --------------------------------------------------------
        # Step 2: active learning loop
        # --------------------------------------------------------
        while self.alive > 0 and len(self.done) < self.stop:
            print(f"[Run {self.run}] | [EZR] Evaluations left: {self.alive}")

            sampled_configs = self.config_manager.sampling_configs(self.sampling_size)
            candidates = self.filter_unevaluated(sampled_configs)

            if len(candidates) == 0:
                print("[INFO] No unevaluated sampled candidates left, stopping early.")
                break

            if self.acq == "near":
                chosen_cfg = self.nearer(candidates)
            else:
                chosen_cfg = self.likelier(candidates)

            if chosen_cfg is None:
                print("[INFO] No candidate selected, stopping.")
                break

            evaluated_one, self.alive = self.config_manager.evaluate(
                [chosen_cfg], self.fallback, self.alive
            )

            if not evaluated_one:
                print("[WARN] Evaluation returned empty result. Stopping early.")
                break

            cfg, perf = evaluated_one[0]
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))
            self.done.append((cfg, perf))

            # source-style update: add new point into best first
            self.best.append((cfg, perf))

            # then prune best back to sqrt(|done|)
            threshold = max(1, round(len(self.done) ** 0.5))
            self.best = self.sort_configs_by_perf(self.best)
            while len(self.best) > threshold:
                weakest = self.best.pop(-1)
                self.rest.append(weakest)

            try:
                print(f"[INFO] Evaluated new config: {cfg} -> {perf:.4f}")
            except Exception:
                print(f"[INFO] Evaluated new config: {cfg} -> {perf}")

        self._finalize(start_time)
        return None

    # ============================================================
    # Candidate filtering
    # ============================================================

    def filter_unevaluated(self, configs):
        """
        Remove already evaluated configs while preserving order.
        """
        out = []
        seen = set()

        for cfg in configs:
            key = tuple(cfg)
            if key in self.evaluated_configs:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(cfg)

        return out

    # ============================================================
    # Selection strategies
    # ============================================================

    def nearer(self, candidates):
        """
        Sampled non-pool version of EZR's `nearer`.

        Original logic:
            randomly try up to Few candidates;
            return the first one that is closer to best_mid than rest_mid;
            otherwise fallback to a random candidate.
        """
        if len(candidates) == 0:
            return None

        if len(self.best) == 0 or len(self.rest) == 0:
            return random.choice(candidates)

        best_mid = self.group_mid(self.best)
        rest_mid = self.group_mid(self.rest)

        trials = min(self.few, len(candidates))
        probe = random.sample(candidates, trials)

        for row in probe:
            if self.distx(best_mid, row) < self.distx(rest_mid, row):
                return row

        return random.choice(candidates)

    def likelier(self, candidates):
        """
        Sampled non-pool version of EZR's `likelier`.

        Since there is no global rolling queue, we do:
        - score a front window from the sampled candidates
        - return the highest-scoring one

        We keep the Few*2 front-window spirit, but now the front is drawn
        from the newly sampled candidate batch of this iteration.
        """
        if len(candidates) == 0:
            return None

        nall = len(self.best) + len(self.rest)
        if nall <= 0:
            return random.choice(candidates)

        p = nall / max(1, self.stop)

        if self.acq == "xploit":
            q = 0.0
        elif self.acq == "xplor":
            q = 1.0
        else:  # adapt or bore
            q = 1.0 - p

        # preserve original front-window flavor
        if len(candidates) > self.few * 2:
            front = random.sample(candidates, self.few * 2)
        else:
            front = candidates

        def score_fn(row):
            b = math.exp(self.likes(self.best, row, nall=nall, nh=2))
            r = math.exp(self.likes(self.rest, row, nall=nall, nh=2))

            if self.acq == "bore":
                return b * b / (r + self.eps)

            return (b + r * q) / abs(b * q - r + self.eps)

        ranked = sorted(front, key=score_fn, reverse=True)
        return ranked[0]

    # ============================================================
    # Likelihood
    # ============================================================

    def likes(self, group, row, nall=100, nh=2):
        """
        Mixed-type log-likelihood of row under a group.
        """
        if len(group) == 0:
            return math.log(self.eps)

        prior = len(group) / (nall + self.eps)
        log_prior = math.log(max(prior, self.eps))
        log_terms = [log_prior]

        num_features = len(row)
        for i in range(num_features):
            v = row[i]
            if v == "?":
                continue

            values = [cfg[i] for cfg, _ in group if cfg[i] != "?"]
            if len(values) == 0:
                continue

            domain = self.config_manager.value_domains[i]
            if self.is_categorical_domain(domain, v):
                log_terms.append(self.like_sym(v, values, prior))
            else:
                log_terms.append(self.like_num(v, values))

        return sum(log_terms)

    def like_sym(self, value, values, prior):
        counts = Counter(values)
        tmp = (counts.get(value, 0) + self.m * prior) / (
            sum(counts.values()) + self.m + self.eps
        )
        return math.log(max(tmp, self.eps))

    def like_num(self, value, values):
        arr = np.asarray(values, dtype=float)
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        if sd == 0:
            return 0.0 if float(value) == mu else math.log(self.eps)

        var = sd * sd + self.eps
        log_nom = -1.0 * (float(value) - mu) ** 2 / (2 * var)
        log_denom = 0.5 * math.log(2 * math.pi * var)
        return log_nom - log_denom

    # ============================================================
    # Distance utilities (for acq="near")
    # ============================================================

    def group_mid(self, group):
        """
        Mid per dimension:
        - categorical -> mode
        - numeric -> mean
        """
        if len(group) == 0:
            return None

        mids = []
        num_features = len(group[0][0])

        for i in range(num_features):
            values = [cfg[i] for cfg, _ in group if cfg[i] != "?"]
            if len(values) == 0:
                mids.append("?")
                continue

            domain = self.config_manager.value_domains[i]
            if self.is_categorical_domain(domain, values[0]):
                mids.append(Counter(values).most_common(1)[0][0])
            else:
                mids.append(float(np.mean(np.asarray(values, dtype=float))))

        return mids

    def distx(self, row1, row2):
        """
        Mixed-type Minkowski distance.
        """
        ds = []
        for i, (a, b) in enumerate(zip(row1, row2)):
            domain = self.config_manager.value_domains[i]
            ds.append(self.aha(domain, a, b) ** self.p)

        return (sum(ds) / max(1, len(ds))) ** (1.0 / self.p)

    def aha(self, domain, a, b):
        """
        Source-inspired Aha distance:
        - both missing -> 1
        - symbolic -> 0/1
        - numeric -> normalized absolute difference
        """
        if a == "?" and b == "?":
            return 1.0

        if self.is_categorical_domain(domain, a) or self.is_categorical_domain(domain, b):
            return 0.0 if a == b else 1.0

        numeric_domain = [float(x) for x in domain if x != "?"]
        lo = min(numeric_domain)
        hi = max(numeric_domain)

        a = self.norm(a, lo, hi)
        b = self.norm(b, lo, hi)

        if a == "?":
            a = 0.0 if b > 0.5 else 1.0
        if b == "?":
            b = 0.0 if a > 0.5 else 1.0

        return abs(a - b)

    def norm(self, v, lo, hi):
        if v == "?":
            return "?"
        return (float(v) - lo) / (hi - lo + self.eps)

    # ============================================================
    # Best / Rest maintenance
    # ============================================================

    def rebuild_best_rest(self):
        """
        Rebuild best/rest from all labeled data.
        """
        ordered = self.sort_configs_by_perf(self.done)
        cut = max(1, round(len(ordered) ** 0.5))

        self.best = ordered[:cut]
        self.rest = ordered[cut:]

    def sort_configs_by_perf(self, cfg_perf_list):
        reverse = self.config_manager.optimization_goal == "maximum"
        return sorted(cfg_perf_list, key=lambda x: x[1], reverse=reverse)

    def is_categorical_domain(self, domain, value):
        if isinstance(value, str):
            return True
        for d in domain:
            if isinstance(d, str):
                return True
        return False

    # ============================================================
    # Finalization
    # ============================================================

    def _finalize(self, start_time):
        if self.logger is not None:
            self.logger.log_all_results(self.all_results)
            end_time = time.time()
            self.logger.store_runtime_to_csv(end_time - start_time)