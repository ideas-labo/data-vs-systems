import time
import math
import random
import numpy as np
from collections import Counter


class EZRPoolTuner:
    """
    Faithful pool-based reproduction of the original EZR active learner.

    Main ideas reproduced from source:
    1. Initialize an unlabeled queue x by shuffling the full pool once.
    2. Warm-start by popping Any items from x and evaluating them.
    3. Split labeled data into best/rest.
    4. Repeatedly choose one new config from x using:
         - nearer() if acq == "near"
         - likelier() otherwise
    5. After each new evaluation:
         - add it to done
         - temporarily add it to best
         - prune best back to sqrt(|done|), moving weakest to rest
    """

    def __init__(self, config_manager, run, budget, logger=None, fallback=None):
        self.config_manager = config_manager
        self.run = run
        self.logger = logger
        self.fallback = fallback
        self.alive = budget

        # Source-style defaults
        self.start = 4              # Any
        self.stop = budget          # Budget
        self.few = 128              # Few
        self.acq = "adapt"           # near | xploit | xplor | bore | adapt
        self.m = 1                  # Bayes low frequency param
        self.p = 2                  # distance coefficient
        self.eps = 1e-32

        self.evaluated_configs = {}   # config tuple -> perf
        self.all_results = []

        # faithful dynamic structures
        self.todo_queue = []          # corresponds to x.rows
        self.done = []                # corresponds to xy.rows, list of (cfg, perf)
        self.best = []                # list of (cfg, perf)
        self.rest = []                # list of (cfg, perf)

    # ============================================================
    # Main
    # ============================================================

    def tuning(self):
        start_time = time.time()

        pool_configs = self.get_pool_configs()
        if len(pool_configs) == 0:
            print("[INFO] Empty config pool, stopping.")
            self._finalize(start_time)
            return None

        # x = shuffle(rows[:])
        self.todo_queue = pool_configs[:]
        random.shuffle(self.todo_queue)

        # --------------------------------------------------------
        # Step 1: warm start by popping from x
        # source:
        # for _ in range(the.Any): add(xy, label(sub(x, x.rows.pop())))
        # --------------------------------------------------------
        init_size = min(self.start, len(self.todo_queue), self.alive)
        if init_size <= 0:
            print("[INFO] No budget for initialization, stopping.")
            self._finalize(start_time)
            return None

        init_configs = [self.todo_queue.pop() for _ in range(init_size)]

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

        # --------------------------------------------------------
        # Step 2: divide labelled items into best/rest
        # source:
        # xy.rows = distysort(xy); n = round(the.Any**.5)
        # adds(xy.rows[:n], best); adds(xy.rows[n:], rest)
        # --------------------------------------------------------
        self.done = self.sort_configs_by_perf(self.done)
        cut = max(1, round(self.start ** 0.5))
        self.best = self.done[:cut]
        self.rest = self.done[cut:]

        if len(self.best) == 0:
            print("[INFO] Empty best set after initialization, stopping.")
            self._finalize(start_time)
            return None

        # --------------------------------------------------------
        # Step 3: active learning loop
        # source:
        # while x.n > 2 and xy.n < the.Budget:
        # --------------------------------------------------------
        while self.alive > 0 and len(self.todo_queue) > 2 and len(self.done) < self.stop:
            print(f"[Run {self.run}] | [EZR-Pool] Evaluations left: {self.alive}")

            # guess = nearer if acq=="near" else likelier
            if self.acq == "near":
                chosen_cfg = self.nearer()
            else:
                chosen_cfg = self.likelier()

            if chosen_cfg is None:
                print("[INFO] No candidate returned, stopping.")
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

            # source:
            # add(xy, add(best, sub(x, label(guess(...)))))
            # i.e., new point is first added to best
            self.best.append((cfg, perf))

            # source:
            # if best.n > (xy.n**.5):
            #   best.rows = distysort(xy,best.rows)
            #   while best.n > (xy.n**.5):
            #       add(rest, sub(best, best.rows.pop(-1)))
            threshold = max(1, round(len(self.done) ** 0.5))
            if len(self.best) > threshold:
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
    # Original EZR acquisition methods
    # ============================================================

    def nearer(self):
        """
        Faithful to source:

            for _ in range(Few):
                row = x.rows[i := random.randrange(x.n)]
                if distx(xy, mids(best), row) < distx(xy, mids(rest), row):
                    return x.rows.pop(i)
            return x.rows.pop()

        Notes:
        - best/rest mids are computed from current groups.
        - if rest is empty, fallback to random pop.
        """
        if len(self.todo_queue) == 0:
            return None

        if len(self.best) == 0 or len(self.rest) == 0:
            return self.todo_queue.pop()

        best_mid = self.group_mid(self.best)
        rest_mid = self.group_mid(self.rest)

        trials = min(self.few, len(self.todo_queue))
        for _ in range(trials):
            i = random.randrange(len(self.todo_queue))
            row = self.todo_queue[i]
            if self.distx(best_mid, row) < self.distx(rest_mid, row):
                return self.todo_queue.pop(i)

        return self.todo_queue.pop()

    def likelier(self):
        """
        Faithful to source:

            first, *lst = sorted(x.rows[:Few*2], key=_fn, reverse=True)
            x.rows = lst[:Few] + x.rows[Few*2:] + lst[Few:]
            return first

        This dynamic queue reordering is the key behavior missing from
        the previous simplified implementation.
        """
        if len(self.todo_queue) == 0:
            return None

        nall = len(self.best) + len(self.rest)
        if nall <= 0:
            return self.todo_queue.pop()

        p = nall / max(1, self.stop)

        if self.acq == "xploit":
            q = 0.0
        elif self.acq == "xplor":
            q = 1.0
        else:  # adapt or bore or any other non-near mode
            q = 1.0 - p

        window_size = min(len(self.todo_queue), self.few * 2)
        front = self.todo_queue[:window_size]

        def score_fn(row):
            b = math.exp(self.likes(self.best, row, nall=nall, nh=2))
            r = math.exp(self.likes(self.rest, row, nall=nall, nh=2))

            if self.acq == "bore":
                return b * b / (r + self.eps)

            return (b + r * q) / abs(b * q - r + self.eps)

        ranked = sorted(front, key=score_fn, reverse=True)
        first = ranked[0]
        lst = ranked[1:]

        keep_front = lst[:self.few]
        push_back = lst[self.few:]

        # faithful queue update
        self.todo_queue = keep_front + self.todo_queue[window_size:] + push_back

        return first

    # ============================================================
    # Likelihood
    # ============================================================

    def likes(self, group, row, nall=100, nh=2):
        """
        Faithful to source:

            prior = data.n / (nall + eps)
            log_prior = log(max(prior, eps))
            tmp = [like(c, row[c.at]) ...]
            return log_prior + sum(tmp)

        Here we compute per-dimension summaries directly from group rows.
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
        tmp = (counts.get(value, 0) + self.m * prior) / (sum(counts.values()) + self.m + self.eps)
        return math.log(max(tmp, self.eps))

    def like_num(self, value, values):
        arr = np.asarray(values, dtype=float)
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        # faithful to source:
        # if sd == 0: return 0 if v == mu else 1E-32
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
        Mixed-type Minkowski distance, close in spirit to source distx().
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
    # Sorting / pool utilities
    # ============================================================

    def sort_configs_by_perf(self, cfg_perf_list):
        reverse = self.config_manager.optimization_goal == "maximum"
        return sorted(cfg_perf_list, key=lambda x: x[1], reverse=reverse)

    def get_pool_configs(self):
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