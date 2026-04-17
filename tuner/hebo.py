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

            # 1) 从 HEBO 请求一个配置
            rec = self.optimizer.suggest(n_suggestions=1)
            config_dict = rec.iloc[0].to_dict()

            # 2) 按数据集列顺序转成 list
            cfg_list = [config_dict[name] for name in header]
            cfg_key = tuple(cfg_list)

            # 如果已经评估过，就跳过这轮
            if cfg_key in seen_configs:
                continue

            # 3) 查表评价
            evaluated, self.alive = self.config_manager.evaluate(
                [cfg_list], self.fallback, self.alive
            )
            # evaluate 现在返回 (cfg, perf)；如果你之后加了 index，这里也兼容
            cfg_eval, perf = evaluated[0][:2]

            # 4) 记录结果（和 FLASH 风格保持一致）
            self.evaluated_configs[tuple(cfg_eval)] = perf
            self.all_results.append((cfg_eval, perf))
            seen_configs.add(cfg_key)

            # 5) 方向适配：HEBO 默认最小化
            goal = getattr(self.config_manager, "optimization_goal", "minimum")
            obj = -perf if goal == "maximum" else perf

            # 6) 回传给 HEBO
            self.optimizer.observe(rec, np.array([[obj]]))

            print(f"[HEBO] Evaluated config: {cfg_eval} -> perf = {perf:.4f}")

        # 和 FLASH 一样，结束时把所有结果写日志
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

    def create_hebo_search_space(self):
        params = []
        value_domains = self.config_manager.value_domains
        header = self.config_manager.data_df.columns[:self.config_manager.config_dim].tolist()

        for i, domain in enumerate(value_domains):
            name = header[i] if i < len(header) else f"param_{i}"
            vals = self._normalize_domain_vals(domain)
            uniq = sorted(set(vals), key=lambda x: (str(type(x)), x))

            # 1) 纯布尔：枚举（通常就是 [False, True]）
            if self._is_all_bool(vals):
                params.append({'name': name, 'type': 'cat', 'categories': uniq})
                continue

            # 2) 纯整数且不全相等 -> 区间整数
            if self._is_all_int(vals) and len(uniq) > 1:
                lo, hi = int(min(vals)), int(max(vals))
                params.append({'name': name, 'type': 'int', 'lb': lo, 'ub': hi})
                continue

            # 3) 浮点型（或 int+float 混合）且不全相等 -> 区间浮点
            if self._is_all_float_like(vals) and (min(vals) != max(vals)):
                lo, hi = float(min(vals)), float(max(vals))
                params.append({'name': name, 'type': 'num', 'lb': lo, 'ub': hi})
                continue

            # 4) 其他情况（字符串、混合类型、只有一个取值等） -> 枚举
            params.append({'name': name, 'type': 'cat', 'categories': uniq})

        return DesignSpace().parse(params)


