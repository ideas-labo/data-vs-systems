import time

import numpy as np
import random

from typing import List, Tuple
from copy import deepcopy
from scipy.spatial.distance import cdist

from openbox import Optimizer
from openbox import ParallelOptimizer
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from tqdm import tqdm


class SMACTuner:
    def __init__(self, config_manager, run, budget, logger=None, fallback=None):

        self.config_manager = config_manager

        # SMAC parameters
        self.config_space = self.get_configspace()
        self.initial_size = 30

        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []
        self.logger = logger
        self.fallback = fallback  # Fallback strategy for configurations not found in the dataset
        self.run = run
        self.alive = budget
        self.budget = budget

    def to_py_scalar(self, x):
        # transform numpy scalar to python native scalar, avoiding serialization/comparison issues
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x

    # def get_configspace(self):
    #     """
    #     Define configuration space entirely with categorical enums
    #     based on dataset info from self.config_manager.
    #     """
    #     cs = ConfigurationSpace()
    #
    #     value_domains = self.config_manager.value_domains
    #     # only take configuration column names (first config_dim columns)
    #     header = self.config_manager.data_df.columns[:self.config_manager.config_dim].tolist()
    #
    #     for i, domain in enumerate(value_domains):
    #         name = header[i] if i < len(header) else f"param_{i}"
    #         # keep the order as in the dataset, convert numpy scalars to native types
    #         choices = [self.to_py_scalar(v) for v in domain]
    #         cs.add_hyperparameter(CategoricalHyperparameter(name, choices))
    #
    #     return cs

    def _normalize_domain_vals(self, domain):
        # 去掉 None，并把 numpy 标量转成原生
        vals = [self.to_py_scalar(v) for v in domain if v is not None]
        if not vals:
            raise ValueError("Empty value domain for a knob.")
        return vals

    def _is_all_bool(self, vals):
        # bool 是 int 的子类，必须优先判定
        return all(isinstance(v, bool) for v in vals)

    def _is_all_int(self, vals):
        # 允许纯整数；排除 bool
        return all(isinstance(v, int) and not isinstance(v, bool) for v in vals)

    def _is_all_float_like(self, vals):
        # 只要出现 float（或混合 int+float），视为连续浮点
        # 排除 bool
        return all((isinstance(v, (int, float)) and not isinstance(v, bool)) for v in vals) \
            and any(isinstance(v, float) for v in vals)

    def get_configspace(self):
        """
        连续/整数 -> UniformFloat/UniformInteger；字符串/布尔/混合 -> Categorical
        """
        cs = ConfigurationSpace()

        value_domains = self.config_manager.value_domains
        header = self.config_manager.data_df.columns[:self.config_manager.config_dim].tolist()

        for i, domain in enumerate(value_domains):
            name = header[i] if i < len(header) else f"param_{i}"
            vals = self._normalize_domain_vals(domain)
            uniq = sorted(set(vals), key=lambda x: (str(type(x)), x))

            # 1) 纯布尔：枚举（通常就是 [False, True]）
            if self._is_all_bool(vals):
                cs.add_hyperparameter(CategoricalHyperparameter(name, sorted(set(vals))))
                continue

            # 2) 纯整数且不全相等 -> 区间整数
            if self._is_all_int(vals) and len(uniq) > 1:
                lo, hi = int(min(vals)), int(max(vals))
                cs.add_hyperparameter(UniformIntegerHyperparameter(name, lo, hi))
                continue

            # 3) 浮点型（或 int+float 混合）且不全相等 -> 区间浮点
            if self._is_all_float_like(vals) and (min(vals) != max(vals)):
                lo, hi = float(min(vals)), float(max(vals))
                cs.add_hyperparameter(UniformFloatHyperparameter(name, lo, hi))
                continue

            # 4) 其他情况（字符串、混合类型、只有一个取值等） -> 枚举
            cs.add_hyperparameter(CategoricalHyperparameter(name, uniq))

        return cs

    def objective_function(self, config):
        """
        Objective function for dataset-based evaluation in SMAC.
        Ensures config order matches dataset column order.
        """
        # sort config according to dataset column order
        header = self.config_manager.data_df.columns[:self.config_manager.config_dim].tolist()
        cfg_dict = config.get_dictionary()
        cfg_list = [cfg_dict[name] for name in header]

        # Evaluate the configuration using ConfigManager
        evaluated_config, self.alive = self.config_manager.evaluate([cfg_list], self.fallback, self.alive)
        cfg, perf, *_ = evaluated_config[0]
        self.evaluated_configs[tuple(cfg)] = perf
        self.all_results.append((cfg, perf))

        # SMAC/OpenBox: default setting is minimization;
        if self.config_manager.optimization_goal == 'maximum':
            perf = -perf
        # SMAC expects a dict with 'objectives' key
        return {"objectives": [perf]}

    def tuning(self):
        start_time = time.time()
        # Initialization
        optimizer = Optimizer(
            objective_function=self.objective_function,
            config_space=self.config_space,
            max_runs=self.budget,
            task_id=f"SMAC_run_{self.run}",
            advisor_type='default',
            surrogate_type='prf',
            acq_type='ei',
            init_strategy='random',
            initial_runs=self.initial_size,
        )

        history = optimizer.run()

        # Obtain all the evaluated configs and corresponding performances
        all_configs = history.get_config_dicts()
        all_objectives = history.get_objectives()

        # Get the best config and corresponding perf
        best_index = min(range(len(all_objectives)), key=lambda i: all_objectives[i])
        best_config = all_configs[best_index]
        best_perf = all_objectives[best_index][0]

        if self.config_manager.optimization_goal == 'maximum':
            best_perf = -best_perf
        print(f"[SMAC] Best config found: {best_config} -> {best_perf}")

        self.logger.log_all_results(self.all_results)
        end_time = time.time()
        self.logger.store_runtime_to_csv(end_time - start_time)
        return None
