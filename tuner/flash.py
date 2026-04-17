import time

import numpy as np
import random

from typing import List, Tuple
from copy import deepcopy
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeRegressor


class FLASHTuner:
    def __init__(self, config_manager, run, budget, logger=None, fallback=None):

        self.config_manager = config_manager

        # FLASH parameters
        self.initial_size = 30
        self.sampling_size = 1000

        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []
        self.logger = logger
        self.fallback = fallback  # Fallback strategy for configurations not found in the dataset
        self.run = run
        self.alive = budget

    def tuning(self):
        start_time = time.time()
        init_configs = self.config_manager.sampling_configs(self.initial_size)
        evaluated_configs, self.alive = self.config_manager.evaluate(init_configs, self.fallback, self.alive)
        for cfg, perf in evaluated_configs:
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))

        while self.alive > 0:
            print(f"[Run {self.run}] | [FLASH] Evaluations left: {self.alive}")

            # Step 1: Train a CART: regression decision tree model
            model = DecisionTreeRegressor()
            configs = [config for config, _ in evaluated_configs]
            train_x = self.preprocess_configs_with_knobs_info(configs)
            train_y = [perf for _, perf in evaluated_configs]
            model.fit(train_x, train_y)

            # Step 2: Sample a set of configurations
            sampled_configs = self.config_manager.sampling_configs(self.sampling_size)
            # Step 3: Filter those configs that haven't been evaluated in the past
            unevaluated_configs = [
                cfg for cfg in sampled_configs
                if tuple(cfg) not in self.evaluated_configs
            ]
            if not unevaluated_configs:
                print("[INFO] No unevaluated configs left, stopping early.")
                break

            # Step 4: Pre-process and predict their performances
            # As some of the configs that may take values as strings, preprocess is needed
            test_x = self.preprocess_configs_with_knobs_info(unevaluated_configs)
            predicted_performances = model.predict(test_x)

            # Step 5: select the best config according to optimization_goal
            if self.config_manager.optimization_goal == "maximum":
                best_config_index = int(np.argmax(predicted_performances))
            else:  # minimum
                best_config_index = int(np.argmin(predicted_performances))
            best_config = unevaluated_configs[best_config_index]

            # Step 6 Evaluate the best configuration
            evaluated_best_config, self.alive = self.config_manager.evaluate([best_config], self.fallback, self.alive)
            evaluated_configs += evaluated_best_config
            cfg, perf = evaluated_best_config[0]
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))

            print(f"[INFO] Evaluated new config: {cfg} -> {perf:.4f}")

        self.logger.log_all_results(self.all_results)
        end_time = time.time()
        self.logger.store_runtime_to_csv(end_time - start_time)
        return None



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
        value_domains = self.config_manager.value_domains  # 从 ConfigManager 拿到所有维度的取值空间

        for cfg in configs:
            processed_cfg = []
            for i, val in enumerate(cfg):
                domain = value_domains[i]
                # if the value is string or non-numeric, map to its index in the domain
                if isinstance(val, str) or not isinstance(val, (int, float)):
                    processed_cfg.append(domain.index(val))
                else:
                    processed_cfg.append(val)
            processed_configs.append(processed_cfg)

        return processed_configs



