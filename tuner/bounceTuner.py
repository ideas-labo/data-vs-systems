import time
import random
import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

from tuner.bounce.bounce import Bounce
from tuner.bounce.benchmarks import Benchmark
from tuner.bounce.util.benchmark import Parameter, ParameterType


class DatasetBenchmarkAdapter(Benchmark):
    """
    Adapter: Bounce -> benchmark(x) -> decode x -> config(list) -> dataset evaluate -> y

    - Bounce minimizes y.
    - We convert dataset perf into a minimization objective:
        if optimization_goal == "maximum": y = -perf
        else (minimum):                    y = perf

    Duplicate handling:
      - cache: return cached y without consuming budget
      - penalize: return large y
    """

    def __init__(
        self,
        parameters: List[Parameter],
        tuner: "BounceTuner",
        use_cache: bool = True,
        duplicate_policy: str = "cache",   # "cache" | "penalize"
        duplicate_penalty: float = 1e9,
    ):
        super().__init__(parameters=parameters, noise_std=None, flip=False)

        self.tuner = tuner

        self.use_cache = use_cache
        self.duplicate_policy = duplicate_policy
        self.duplicate_penalty = float(duplicate_penalty)

        # key: tuple(config) -> (y, perf)
        self.cache: Dict[Tuple[Any, ...], Tuple[float, float]] = {}

        # Pre-compute slices for decoding embedded x -> per-parameter segment
        self._slices = self._build_param_slices()

    def _build_param_slices(self):
        slices = []
        start = 0
        for p in self.parameters:
            width = p.dims_required
            slices.append((p, start, start + width))
            start += width
        return slices

    @staticmethod
    def _reflect_into_bounds(v: float, lb: float, ub: float) -> float:
        """
        Reflect v into [lb, ub] deterministically (no clamp bias).
        """
        if lb >= ub:
            return lb
        span = ub - lb
        x = (v - lb) % (2.0 * span)
        if x > span:
            x = 2.0 * span - x
        return lb + x

    def _decode_one_to_config(self, x_row: np.ndarray) -> List[Any]:
        cfg: List[Any] = []

        for dim_i, (p, s, e) in enumerate(self._slices):
            domain = self.tuner.config_manager.value_domains[dim_i] or []
            k = len(domain)

            # No domain info: just return raw scalar
            if k == 0:
                cfg.append(float(x_row[s]))
                continue

            if p.type == ParameterType.CONTINUOUS:
                # 1-d index variable stored at x_row[s]
                v = float(x_row[s])
                lb = float(p.lower_bound)
                ub = float(p.upper_bound)

                # reflect into [lb, ub] to avoid out-of-range
                if lb < ub:
                    span = ub - lb
                    x = (v - lb) % (2.0 * span)
                    if x > span:
                        x = 2.0 * span - x
                    v = lb + x
                else:
                    v = lb

                idx = int(round(v))
                idx = min(max(idx, 0), k - 1)
                cfg.append(domain[idx])
                continue

            elif p.type == ParameterType.BINARY:
                v = float(x_row[s])
                bit = 1 if v > 0.5 else 0
                cfg.append(domain[bit] if k == 2 else bit)
                continue

            elif p.type == ParameterType.CATEGORICAL:
                # one-hot-like segment x_row[s:e]
                seg = x_row[s:e]
                j = int(np.argmax(seg))
                j = min(max(j, 0), k - 1)
                cfg.append(domain[j])
                continue

            else:
                raise ValueError(f"Unsupported ParameterType: {p.type}")

        return cfg

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        x: torch.Tensor (batch, dim), dtype double
        returns y: torch.Tensor (batch,), dtype double
        """
        # Stop if dataset budget exhausted
        if self.tuner.alive <= 0:
            raise RuntimeError("Budget exhausted (alive<=0).")

        device = x.device
        x_np = x.detach().cpu().numpy()

        ys: List[float] = []

        for i in range(x_np.shape[0]):
            # decode to config(list)
            cfg = self._decode_one_to_config(x_np[i])
            key = tuple(cfg)

            # duplicate handling
            if key in self.cache:
                if self.duplicate_policy == "cache":
                    y_cached, _ = self.cache[key]
                    ys.append(float(y_cached))
                    continue
                elif self.duplicate_policy == "penalize":
                    ys.append(float(self.duplicate_penalty))
                    continue
                else:
                    y_cached, _ = self.cache[key]
                    ys.append(float(y_cached))
                    continue

            # Evaluate one config via dataset manager (consumes alive)
            evaluated, self.tuner.alive = self.tuner.config_manager.evaluate([cfg], self.tuner.fallback, self.tuner.alive)
            if not evaluated:
                # treat as worst
                perf = 0.0
            else:
                _, perf = evaluated[0]

            y = self.tuner.to_minimization_objective(perf)

            if self.use_cache:
                self.cache[key] = (float(y), float(perf))

            self.tuner._record_evaluation(cfg, perf)


            # if budget exhausted, stop Bounce by raising
            if self.tuner.alive <= 0:
                raise RuntimeError("Budget exhausted during Bounce evaluation.")

            ys.append(float(y))

        return torch.tensor(ys, dtype=torch.double, device=device)


class BounceTuner:
    """
    Dataset-mode Bounce tuner (FLASH-aligned outer loop).

    Interface matches FLASHTuner:
      __init__(config_manager, run, budget, logger=None, fallback=None)
      tuning()
    """

    def __init__(self, config_manager, run, budget, logger=None, fallback=None):
        self.config_manager = config_manager

        # Budget (same naming as FLASH)
        self.run = run
        self.alive = budget

        self.logger = logger
        self.fallback = fallback

        # For logging consistency with FLASH
        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []

        # Bounce hyperparams (safe defaults, can be overridden by external if you want)
        self.initial_size = 30
        self.batch_size = 1

        # Bounce internal hyperparams (keep conservative defaults)
        self.initial_target_dimensionality = 4
        self.number_new_bins_on_split = 1
        self.desired_final_dimensionality = None
        self.maximum_number_evaluations_until_input_dim = None

        # duplicates
        self.duplicate_policy = "cache"
        self.duplicate_penalty = 1e9

        # # seed (optional)
        # self.seed = run
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)

    def to_minimization_objective(self, perf: float) -> float:
        """
        Convert dataset perf to a minimization objective for Bounce.
        Mirrors FLASH's `optimization_goal`.
        """
        if getattr(self.config_manager, "optimization_goal", "maximum") == "maximum":
            return -float(perf)
        return float(perf)

    def _record_evaluation(self, cfg: List[Any], perf: float):
        """
        Keep the same result bookkeeping as FLASHTuner.
        """
        self.evaluated_configs[tuple(cfg)] = perf
        self.all_results.append((cfg, perf))

    def _build_bounce_parameters_from_value_domains(self) -> List[Parameter]:
        """
        Large-domain dataset mode:
        - Represent each dimension as CONTINUOUS index in [0, K-1] (1 dim only).
        - Decode by rounding to nearest index, then map to value_domains[i][idx].

        NOTE: Do NOT use ORDINAL because Bounce implementation doesn't support it
              (Random sign for ordinal parameters not implemented).
        """
        params: List[Parameter] = []
        vds = self.config_manager.value_domains
        if vds is None:
            raise ValueError("config_manager.value_domains is required for Bounce dataset adapter.")

        for i, domain in enumerate(vds):
            name = f"x{i}"

            if domain is None or len(domain) == 0:
                params.append(Parameter(name=name, type=ParameterType.CONTINUOUS, lower_bound=0.0, upper_bound=1.0))
                continue

            k = len(domain)
            if k == 1:
                params.append(Parameter(name=name, type=ParameterType.CONTINUOUS, lower_bound=0.0, upper_bound=0.0))
            else:
                # 1-d index variable
                params.append(
                    Parameter(name=name, type=ParameterType.CONTINUOUS, lower_bound=0.0, upper_bound=float(k - 1)))

        return params

    def tuning(self):
        start_time = time.time()

        # ---- build benchmark adapter and run Bounce ----
        parameters = self._build_bounce_parameters_from_value_domains()

        benchmark = DatasetBenchmarkAdapter(
            parameters=parameters,
            tuner=self,
            use_cache=True,
            duplicate_policy=self.duplicate_policy,
            duplicate_penalty=self.duplicate_penalty,
        )

        desired_final_dim = benchmark.dim if self.desired_final_dimensionality is None else int(self.desired_final_dimensionality)

        optimizer = Bounce(
            benchmark=benchmark,
            number_initial_points=self.initial_size,
            initial_target_dimensionality=min(self.initial_target_dimensionality, desired_final_dim),
            number_new_bins_on_split=self.number_new_bins_on_split,
            maximum_number_evaluations=10**12,  # stop by raising when alive<=0
            batch_size=self.batch_size,
            results_dir=os.path.join("logs", "bounce_tmp_results"),  # dataset mode: no need to save Bounce internal files
            desired_final_dimensionality=desired_final_dim,
            maximum_number_evaluations_until_input_dim=(
                None if self.maximum_number_evaluations_until_input_dim is None
                else int(self.maximum_number_evaluations_until_input_dim)
            ),
        )

        try:
            optimizer.run()
        except Exception as e:
            # Budget exhausted or Bounce internal early stop
            print(f"[BounceTuner] Optimization stopped: {e}")

        # ---- logging (same as FLASH) ----
        if self.logger is not None:
            self.logger.log_all_results(self.all_results)
            end_time = time.time()
            self.logger.store_runtime_to_csv(end_time - start_time)

        return None
