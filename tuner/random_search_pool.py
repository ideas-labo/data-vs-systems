import time
import random
import numpy as np


class RandomSearchPoolTuner:
    """
    Random Search baseline over a finite config pool:
    - Each iteration randomly selects ONE unevaluated config
      from config_manager.config_space.
    - No Cartesian-space sampling is used.
    - Stop early if the pool is exhausted.
    """

    def __init__(self, config_manager, run, budget, logger=None, fallback=None):
        self.config_manager = config_manager

        self.max_resample_trials = 100  # kept for style consistency

        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []
        self.logger = logger
        self.fallback = fallback
        self.run = run
        self.alive = budget

    def tuning(self):
        start_time = time.time()

        pool_configs = self.get_pool_configs()

        if len(pool_configs) == 0:
            print("[INFO] Empty config pool, stopping.")
            if self.logger is not None:
                self.logger.log_all_results(self.all_results)
                end_time = time.time()
                self.logger.store_runtime_to_csv(end_time - start_time)
            return None

        while self.alive > 0:
            print(f"[Run {self.run}] | [RandomSearch-Pool] Evaluations left: {self.alive}")

            # Get all unevaluated configs from the finite pool
            unevaluated_configs = [
                cfg for cfg in pool_configs
                if tuple(cfg) not in self.evaluated_configs
            ]

            if not unevaluated_configs:
                print("[INFO] RandomSearch-Pool stopped early: no unevaluated configs left in the pool.")
                break

            # Randomly choose ONE config from the remaining pool
            best_config = random.choice(unevaluated_configs)

            # Evaluate ONE config
            evaluated_best_config, self.alive = self.config_manager.evaluate(
                [best_config], self.fallback, self.alive
            )

            if not evaluated_best_config:
                print("[WARN] Evaluation returned empty result. Stopping to avoid infinite loop.")
                break

            cfg, perf = evaluated_best_config[0]
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))

            try:
                print(f"[INFO] Evaluated new config: {cfg} -> {perf:.4f}")
            except Exception:
                print(f"[INFO] Evaluated new config: {cfg} -> {perf}")

        if self.logger is not None:
            self.logger.log_all_results(self.all_results)
            end_time = time.time()
            self.logger.store_runtime_to_csv(end_time - start_time)

        return None

    def get_pool_configs(self):
        """
        Get all configurations from config_manager.config_space.
        Ensures each config is converted to a Python list for consistent processing.
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