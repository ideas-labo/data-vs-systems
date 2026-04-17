import time

import numpy as np
import random

from typing import List, Tuple
from copy import deepcopy
from scipy.spatial.distance import cdist

from tqdm import tqdm
from utils.logger import Logger
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace


class HEBOTuner:
    def __init__(self, config_manager, run, budget, logger=None, fallback=None):

        self.config_manager = config_manager

        # HEBO parameters

        self.initial_size = 30

        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []
        self.logger = logger
        self.fallback = fallback  # Fallback strategy for configurations not found in the dataset
        self.run = run
        self.alive = budget
        self.budget = budget

        self.search_space = self.create_hebo_search_space()
        self.optimizer = HEBO(space=self.search_space, rand_sample=30)


    def tuning(self):
        start_time = time.time()
        seen_configs = set()

        header = self.config_manager.data_df.columns[:self.config_manager.config_dim].tolist()

        while self.alive > 0:
            print(f"[Run {self.run}] | [HEBO] Evaluations left: {self.alive}")

            # 1) Request a configuration from HEBO
            rec = self.optimizer.suggest(n_suggestions=1)
            config_dict = rec.iloc[0].to_dict()

            # 2) Convert to a list following the dataset column order
            cfg_list = [config_dict[name] for name in header]
            cfg_key = tuple(cfg_list)

            # Skip this iteration if the configuration has already been evaluated
            if cfg_key in seen_configs:
                continue

            # 3) look up table or fallback
            evaluated, self.alive = self.config_manager.evaluate(
                [cfg_list], self.fallback, self.alive
            )
            # evaluate will returen (cfg, perf)； compatible with index-based and key-based lookup
            cfg_eval, perf = evaluated[0][:2]

            # 4) record the evaluation result
            self.evaluated_configs[tuple(cfg_eval)] = perf
            self.all_results.append((cfg_eval, perf))
            seen_configs.add(cfg_key)

            # 5) Direction adaptation: HEBO minimizes by default
            goal = getattr(self.config_manager, "optimization_goal", "minimum")
            obj = -perf if goal == "maximum" else perf

            # 6) return observation to HEBO
            self.optimizer.observe(rec, np.array([[obj]]))

            print(f"[HEBO] Evaluated config: {cfg_eval} -> perf = {perf:.4f}")

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.log_all_results(self.all_results)

        print("[HEBO] Tuning finished.")
        end_time = time.time()
        self.logger.store_runtime_to_csv(end_time - start_time)
        return None



    def to_py_scalar(self, x):
        # transform numpy scalar to python native scalar, avoiding serialization/comparison issues
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x

    def _normalize_domain_vals(self, domain):
        # Remove None values and convert numpy scalars to native Python scalars
        vals = [self.to_py_scalar(v) for v in domain if v is not None]
        if not vals:
            raise ValueError("Empty value domain for a knob.")
        return vals

    def _is_all_bool(self, vals):
        return all(isinstance(v, bool) for v in vals)

    def _is_all_int(self, vals):
        return all(isinstance(v, int) and not isinstance(v, bool) for v in vals)

    def _is_all_float_like(self, vals):
        return all((isinstance(v, (int, float)) and not isinstance(v, bool)) for v in vals) \
            and any(isinstance(v, float) for v in vals)

    def create_hebo_search_space(self):
        params = []
        value_domains = self.config_manager.value_domains
        header = self.config_manager.data_df.columns[:self.config_manager.config_dim].tolist()

        for i, domain in enumerate(value_domains):
            name = header[i] if i < len(header) else f"param_{i}"
            vals = self._normalize_domain_vals(domain)
            uniq = sorted(set(vals), key=lambda x: (str(type(x)), x))

            if self._is_all_bool(vals):
                params.append({'name': name, 'type': 'cat', 'categories': uniq})
                continue

            if self._is_all_int(vals) and len(uniq) > 1:
                lo, hi = int(min(vals)), int(max(vals))
                params.append({'name': name, 'type': 'int', 'lb': lo, 'ub': hi})
                continue

            if self._is_all_float_like(vals) and (min(vals) != max(vals)):
                lo, hi = float(min(vals)), float(max(vals))
                params.append({'name': name, 'type': 'num', 'lb': lo, 'ub': hi})
                continue

            params.append({'name': name, 'type': 'cat', 'categories': uniq})

        return DesignSpace().parse(params)


