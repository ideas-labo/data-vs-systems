import time
import numpy as np
import random

from sklearn.tree import DecisionTreeRegressor


class FLASHPoolTuner:
    def __init__(self, config_manager, run, budget, logger=None, fallback=None):

        self.config_manager = config_manager

        # FLASH parameters
        self.initial_size = 30

        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []
        self.logger = logger
        self.fallback = fallback  # Fallback strategy for configurations not found in the dataset
        self.run = run
        self.alive = budget

    def tuning(self):
        start_time = time.time()

        # Step 0: Get all configs from the tabular pool
        pool_configs = self.get_pool_configs()

        if len(pool_configs) == 0:
            print("[INFO] Empty config pool, stopping.")
            if self.logger is not None:
                self.logger.log_all_results(self.all_results)
                end_time = time.time()
                self.logger.store_runtime_to_csv(end_time - start_time)
            return None

        # Step 1: Initialize from config pool instead of Cartesian sampling
        init_size = min(self.initial_size, len(pool_configs))
        init_configs = random.sample(pool_configs, init_size)

        evaluated_configs, self.alive = self.config_manager.evaluate(
            init_configs, self.fallback, self.alive
        )

        for cfg, perf in evaluated_configs:
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))

        # Step 2: Sequential model-based search over the whole remaining pool
        while self.alive > 0:
            print(f"[Run {self.run}] | [FLASH-POOL] Evaluations left: {self.alive}")

            if len(evaluated_configs) == 0:
                print("[INFO] No evaluated configs available for training, stopping early.")
                break

            # Train CART on all evaluated samples
            model = DecisionTreeRegressor()

            train_configs = [config for config, _ in evaluated_configs]
            train_x = self.preprocess_configs_with_knobs_info(train_configs)
            train_y = [perf for _, perf in evaluated_configs]

            model.fit(train_x, train_y)

            # Get all unevaluated configs from the pool
            unevaluated_configs = [
                cfg for cfg in pool_configs
                if tuple(cfg) not in self.evaluated_configs
            ]

            if not unevaluated_configs:
                print("[INFO] No unevaluated configs left in the pool, stopping early.")
                break

            # Predict all remaining pool configs
            test_x = self.preprocess_configs_with_knobs_info(unevaluated_configs)
            predicted_performances = model.predict(test_x)

            # Select the best according to optimization goal
            if self.config_manager.optimization_goal == "maximum":
                best_config_index = int(np.argmax(predicted_performances))
            else:  # minimum
                best_config_index = int(np.argmin(predicted_performances))

            best_config = unevaluated_configs[best_config_index]

            # Evaluate the selected config
            evaluated_best_config, self.alive = self.config_manager.evaluate(
                [best_config], self.fallback, self.alive
            )

            if not evaluated_best_config:
                print("[INFO] Evaluation returned empty result, stopping early.")
                break

            evaluated_configs += evaluated_best_config

            cfg, perf = evaluated_best_config[0]
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))

            print(f"[INFO] Evaluated new config: {cfg} -> {perf:.4f}")

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

        # already list-like
        processed_pool = []
        for cfg in pool:
            if isinstance(cfg, np.ndarray):
                processed_pool.append(cfg.tolist())
            else:
                processed_pool.append(list(cfg))
        return processed_pool

    def preprocess_configs_with_knobs_info(self, configs):
        """
        Preprocess configs for dataset mode:
        - If a dimension contains string values (categorical),
          convert them to their index in self.config_manager.value_domains[i].
        - If it's numeric, keep the value as is.

        :param configs: list of configs (each config = list of values)
        :return: processed list of numeric configs
        """
        processed_configs = []
        value_domains = self.config_manager.value_domains  # all domains from ConfigManager

        for cfg in configs:
            processed_cfg = []
            for i, val in enumerate(cfg):
                domain = value_domains[i]

                # categorical / non-numeric -> map to domain index
                if isinstance(val, str) or not isinstance(val, (int, float, np.integer, np.floating)):
                    processed_cfg.append(domain.index(val))
                else:
                    processed_cfg.append(val)

            processed_configs.append(processed_cfg)

        return processed_configs