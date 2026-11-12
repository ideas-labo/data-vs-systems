import math
import random
import time
from typing import List, Tuple

import numpy as np


class BestConfigTuner:
    """
    Dataset-based BestConfig tuner.

    - Initial DDS global sampling
    - Define boundaries (RBS) based on the current best
    - Perform DDS sampling within boundaries for multiple rounds (max_rounds)
    - If no improvement, perform global DDS sampling again
    - Repeat until alive (budget) is exhausted
    """

    def __init__(self, config_manager, run: int, budget: int, logger=None, fallback: str = "worst"):
        self.config_manager = config_manager
        self.logger = logger
        self.run = run
        self.fallback = fallback

        self.max_rounds = 5
        self.sampling_size = 8
        self.alive = budget

        # evaluated_configs: List[(cfg_list, perf)]
        self.evaluated_configs: List[Tuple[List, float]] = []
        self.all_results: List[Tuple[List, float]] = []

        self.feature_names = self.config_manager.data_df.columns[:self.config_manager.config_dim].tolist()
        self.feature_types = getattr(self.config_manager, "feature_types", ["categorical"] * self.config_manager.config_dim)
        self.value_domains = self.config_manager.value_domains

    # Tuning loop

    def tuning(self) -> Tuple[List, float]:

        start_time = time.time()
        if self.config_manager.optimization_goal == "maximum":
            best_perf = -math.inf
        else:
            best_perf = math.inf
        best_config = None

        # Initialize DDS global sampling.
        init_configs = self.dds_sampling(self.sampling_size)
        evaluated, self.alive = self.config_manager.evaluate(init_configs, self.fallback, self.alive)
        # evaluated: List[(cfg, perf)]
        self.evaluated_configs.extend(evaluated)
        self.all_results.extend(evaluated)

        # initialize best_config
        for cfg, perf in evaluated:
            if self._is_better(perf, best_perf):
                best_perf = perf
                best_config = cfg

        if best_config is None:
            print("[BestConfig] No valid config found in initial sampling.")
            if self.logger is not None:
                self.logger.log_all_results(self.all_results)
            return None, None

        # Continue until the evaluation budget is exhausted.
        while self.alive > 0:
            print(f"[Run {self.run}] | [BestConfig] Evaluations left: {self.alive}")
            found_better = False

            # Local RBS and DDS sampling.
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

                # update best
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
                    print("[BestConfig] No better config found in this RBS phase, break.")
                    break

            if self.alive <= 0:
                break

            # Resume global DDS sampling if budget remains.
            print("[BestConfig] Global DDS resampling...")
            global_samples = self.dds_sampling(self.sampling_size)
            new_evaluated, self.alive = self.config_manager.evaluate(global_samples, self.fallback, self.alive)
            self.evaluated_configs.extend(new_evaluated)
            self.all_results.extend(new_evaluated)

            for cfg, perf in new_evaluated:
                if self._is_better(perf, best_perf):
                    best_perf = perf
                    best_config = cfg

        # Log all evaluated configurations.
        if self.logger is not None:
            self.logger.log_all_results(self.all_results)

        print(f"[BestConfig] Finished. Best config: {best_config}, perf = {best_perf}")
        end_time = time.time()
        self.logger.store_runtime_to_csv(end_time - start_time)
        return best_config, best_perf



    def _is_better(self, perf, best_perf) -> bool:
        if self.config_manager.optimization_goal == "maximum":
            return perf > best_perf
        else:
            return perf < best_perf


    def dds_sampling(self, sampling_size: int) -> List[List]:
        """
        - Use `value_domains[j]` as the discrete value domain for each dimension.
          - Treat `[0..sampling_size-1]` as the "sampling interval index",
            and map the index to the domain index using a step size.
          - Ensure that for each dimension, each index is used at most once during `sampling_size` samples
            (preserving the original DDS "divide & diverge" concept).
        """
        return self._dds_sampling_from_domains(self.value_domains, sampling_size)



    def dds_sampling_within_bounds(self, bounded_space: dict, sampling_size: int) -> List[List]:
        """
        Perform DDS sampling within bounds.
        bounded_space: {feature_name: (lower_bound, upper_bound)}
        Numerical dimensions: filter domain values based on bounds;
        Categorical dimensions: keep the original logic — the original code in `get_partitions_within_bounds` does not actually use bounds,
                                so bounds are "ignored" here, and sampling is still performed over the entire categorical domain.
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
                        # If the boundary filtering result is empty, fallback to the original domain
                        filtered = dom
                except Exception:
                    filtered = dom
                restricted_domains.append(filtered)
            else:
                # categorical or no boundary information: keep the original domain (consistent with the original enum logic)
                restricted_domains.append(dom)

        return self._dds_sampling_from_domains(restricted_domains, sampling_size)



    def _dds_sampling_from_domains(self, domains: List[List], sampling_size: int) -> List[List]:
        num_dims = len(domains)
        if num_dims == 0:
            return []

       # Available indices for each dimension (0..sampling_size-1)
        available_indices = [list(range(sampling_size)) for _ in range(num_dims)]
        samples = []

        for _ in range(sampling_size):
            cfg = []
            for dim, dom in enumerate(domains):
                if not dom:
                    raise ValueError(f"[BestConfig] Empty domain at dimension {dim}.")

                if not available_indices[dim]:
                    # If the indices for a dimension are exhausted, reset them (to prevent crashes when sampling_size > len(domain))
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


    def define_bounded_space(self, best_config: List, evaluated_configs: List[Tuple[List, float]]) -> dict:
        """
        - Locate using feature_names + index.
        - Numerical dimensions: boundaries = the closest values smaller/greater than best_value in the evaluation history.
        - Categorical dimensions: boundaries = the previous/next value in the domain.
        Returns:
            bounded_space: {feature_name: (lower_bound, upper_bound)}
        """
        bounded_space = {}

        for j, name in enumerate(self.feature_names):
            best_value = best_config[j]
            ftype = self.feature_types[j] if j < len(self.feature_types) else "numerical"
            dom = self.value_domains[j]

            if ftype == "categorical":
                if best_value not in dom:
                    lower_bound = upper_bound = best_value
                else:
                    idx = dom.index(best_value)
                    lower_bound = dom[idx - 1] if idx > 0 else best_value
                    upper_bound = dom[idx + 1] if idx < len(dom) - 1 else best_value

            else:
                try:
                    best_v = float(best_value)
                except Exception:
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
