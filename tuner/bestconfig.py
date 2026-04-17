import math
import random
import time
from typing import List, Tuple

import numpy as np


class BestConfigTuner:
    """
    Dataset-based BestConfig tuner.

    严格保留原算法逻辑：
    - 初始 DDS global sampling
    - 基于当前 best 定义边界（RBS）
    - 在边界内 DDS sampling 多轮（max_rounds）
    - 若无提升，则重新 global DDS sampling
    - 直到 alive（预算）用完

    区别：
    - 不再使用 target_system / knobs_info / workload_controller
    - 配置是 list，而不是 dict
    - 采样和边界完全基于 config_manager.value_domains + feature_types
    - 评价通过 config_manager.evaluate(config_list)
    """

    def __init__(self, config_manager, run: int, budget: int, logger=None, fallback: str = "worst"):
        self.config_manager = config_manager
        self.logger = logger
        self.run = run
        self.fallback = fallback

        # 搜索参数（保持原设定）
        self.max_rounds = 5
        self.sampling_size = 8

        # 预算：和 FLASH 一样，用 alive 计数
        self.alive = budget

        # 记录 evaluated configs 和所有结果
        # evaluated_configs: List[(cfg_list, perf)]
        self.evaluated_configs: List[Tuple[List, float]] = []
        self.all_results: List[Tuple[List, float]] = []

        # 一些便捷属性
        self.feature_names = self.config_manager.data_df.columns[:self.config_manager.config_dim].tolist()
        # detect_feature_types 在 ConfigManager.set_system_scale() 已经调用过
        self.feature_types = getattr(self.config_manager, "feature_types", ["categorical"] * self.config_manager.config_dim)
        self.value_domains = self.config_manager.value_domains

    # ================== 主入口 ==================

    def tuning(self) -> Tuple[List, float]:
        """
        主调方法：与 FLASH 一致，内部完成整个搜索；
        返回 (best_config, best_perf)，同时将所有结果写入 logger。
        """
        start_time = time.time()

        # 初始化 best_perf
        if self.config_manager.optimization_goal == "maximum":
            best_perf = -math.inf
        else:
            best_perf = math.inf
        best_config = None

        # ===== Step 0: 初始 DDS global sampling =====
        init_configs = self.dds_sampling(self.sampling_size)
        evaluated, self.alive = self.config_manager.evaluate(init_configs, self.fallback, self.alive)
        # evaluated: List[(cfg, perf)]
        self.evaluated_configs.extend(evaluated)
        self.all_results.extend(evaluated)

        # 初始化 best_config
        for cfg, perf in evaluated:
            if self._is_better(perf, best_perf):
                best_perf = perf
                best_config = cfg

        if best_config is None:
            print("[BestConfig] No valid config found in initial sampling.")
            if self.logger is not None:
                self.logger.log_all_results(self.all_results)
            return None, None

        # ===== 主循环：直到 alive 用完 =====
        while self.alive > 0:
            print(f"[Run {self.run}] | [BestConfig] Evaluations left: {self.alive}")
            found_better = False

            # ===== 局部 RBS + DDS 采样，多轮 =====
            for round_id in range(self.max_rounds):
                print(f"[BestConfig] RBS round {round_id + 1}/{self.max_rounds}")

                bounded_space = self.define_bounded_space(best_config, self.evaluated_configs)
                bounded_samples = self.dds_sampling_within_bounds(bounded_space, self.sampling_size)

                if not bounded_samples:
                    print("[BestConfig] No bounded samples generated, break RBS.")
                    break

                new_evaluated, self.alive = self.config_manager.evaluate(bounded_samples, self.fallback, self.alive)
                self.evaluated_configs.extend(new_evaluated)
                self.all_results.extend(new_evaluated)

                # 更新 best
                for cfg, perf in new_evaluated:
                    if self._is_better(perf, best_perf):
                        best_perf = perf
                        best_config = cfg
                        found_better = True

                if self.alive <= 0:
                    print("[BestConfig] Budget exhausted during RBS.")
                    if self.logger is not None:
                        self.logger.log_all_results(self.all_results)
                    return best_config, best_perf

                if not found_better:
                    # RBS 内没有找到更好的，就退出 RBS 阶段
                    print("[BestConfig] No better config found in this RBS phase, break.")
                    break

            if self.alive <= 0:
                break

            # ===== 若预算还没用完：global DDS 再采一波 =====
            print("[BestConfig] Global DDS resampling...")
            global_samples = self.dds_sampling(self.sampling_size)
            new_evaluated, self.alive = self.config_manager.evaluate(global_samples, self.fallback, self.alive)
            self.evaluated_configs.extend(new_evaluated)
            self.all_results.extend(new_evaluated)

            for cfg, perf in new_evaluated:
                if self._is_better(perf, best_perf):
                    best_perf = perf
                    best_config = cfg

        # 记录全部结果
        if self.logger is not None:
            self.logger.log_all_results(self.all_results)

        print(f"[BestConfig] Finished. Best config: {best_config}, perf = {best_perf}")
        end_time = time.time()
        self.logger.store_runtime_to_csv(end_time - start_time)
        return best_config, best_perf

    # ================== 辅助：比较好坏 ==================

    def _is_better(self, perf, best_perf) -> bool:
        if self.config_manager.optimization_goal == "maximum":
            return perf > best_perf
        else:
            return perf < best_perf

    # ================== DDS Sampling（全局） ==================

    def dds_sampling(self, sampling_size: int) -> List[List]:
        """
        Dataset 版本 DDS：
        - 每一维使用 value_domains[j] 作为离散取值域；
        - 将 [0..sampling_size-1] 视为“采样区间 index”，
          按照 index -> domain index（用步长 step）进行映射；
        - 保证每一维在 sampling_size 次采样中，每个 index 至多被用一次
          （保留原始 DDS 的“divide & diverge”思想）。
        """
        return self._dds_sampling_from_domains(self.value_domains, sampling_size)

    # ================== DDS Sampling（局部边界内） ==================

    def dds_sampling_within_bounds(self, bounded_space: dict, sampling_size: int) -> List[List]:
        """
        在边界内使用 DDS 采样。
        bounded_space: {feature_name: (lower_bound, upper_bound)}
        数值维：根据边界过滤 domain 中的值；
        枚举维：保持原逻辑 —— 原代码中 enum 在 get_partitions_within_bounds 里实际没用到 bounds，
                这里也“忽略边界”，仍然在整个枚举域上采样。
        """

        restricted_domains = []
        for j, name in enumerate(self.feature_names):
            dom = self.value_domains[j]
            ftype = self.feature_types[j] if j < len(self.feature_types) else "categorical"

            if ftype == "numerical" and name in bounded_space:
                lower, upper = bounded_space[name]
                try:
                    low_v = float(lower)
                    up_v = float(upper)
                    filtered = [v for v in dom if low_v <= float(v) <= up_v]
                    if not filtered:
                        # 边界过滤结果为空时，退回到原始 domain
                        filtered = dom
                except Exception:
                    filtered = dom
                restricted_domains.append(filtered)
            else:
                # categorical 或无边界信息：保留原域（和原始 enum 逻辑一致）
                restricted_domains.append(dom)

        return self._dds_sampling_from_domains(restricted_domains, sampling_size)

    # ================== DDS 核心（对任意 domains 列表） ==================

    def _dds_sampling_from_domains(self, domains: List[List], sampling_size: int) -> List[List]:
        num_dims = len(domains)
        if num_dims == 0:
            return []

        # 每一维的可用 index（0..sampling_size-1）
        available_indices = [list(range(sampling_size)) for _ in range(num_dims)]
        samples = []

        for _ in range(sampling_size):
            cfg = []
            for dim, dom in enumerate(domains):
                if not dom:
                    raise ValueError(f"[BestConfig] Empty domain at dimension {dim}.")

                if not available_indices[dim]:
                    # 如果某一维 index 用完了，就重置（防止 sampling_size > len(domain) 的情况下崩掉）
                    available_indices[dim] = list(range(sampling_size))

                idx = random.choice(available_indices[dim])
                available_indices[dim].remove(idx)

                num_values = len(dom)
                step = max(1, sampling_size // num_values)
                dom_idx = min(idx // step, num_values - 1)
                val = dom[dom_idx]
                cfg.append(val)

            samples.append(cfg)

        return samples

    # ================== 定义 RBS 边界 ==================

    def define_bounded_space(self, best_config: List, evaluated_configs: List[Tuple[List, float]]) -> dict:
        """
        复刻原始 define_bounded_space 逻辑，只是：
        - config 从 dict 改为 list；
        - 通过 feature_names + index 定位；
        - 数值维：边界 = 历史评估中小于/大于 best_value 的最近值；
        - 枚举维：边界 = domain 中前一个 / 后一个枚举值。
        返回:
            bounded_space: {feature_name: (lower_bound, upper_bound)}
        """
        bounded_space = {}

        for j, name in enumerate(self.feature_names):
            best_value = best_config[j]
            ftype = self.feature_types[j] if j < len(self.feature_types) else "numerical"
            dom = self.value_domains[j]

            if ftype == "categorical":
                # 枚举维：根据 domain 的邻居确定边界
                if best_value not in dom:
                    lower_bound = upper_bound = best_value
                else:
                    idx = dom.index(best_value)
                    lower_bound = dom[idx - 1] if idx > 0 else best_value
                    upper_bound = dom[idx + 1] if idx < len(dom) - 1 else best_value

            else:
                # 数值维：从历史 data 中找最近的小于/大于 best_value 的值
                try:
                    best_v = float(best_value)
                except Exception:
                    # fallback，当作无边界
                    bounded_space[name] = (best_value, best_value)
                    continue

                smaller = []
                larger = []
                for cfg, _perf in evaluated_configs:
                    v = cfg[j]
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if fv < best_v:
                        smaller.append(fv)
                    elif fv > best_v:
                        larger.append(fv)

                lower_v = max(smaller) if smaller else best_v
                upper_v = min(larger) if larger else best_v

                lower_bound = lower_v
                upper_bound = upper_v

            bounded_space[name] = (lower_bound, upper_bound)

        return bounded_space
