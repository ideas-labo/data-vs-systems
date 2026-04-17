import time
from typing import Optional


class RandomSearchTuner:
    """
    Random Search baseline (single-evaluation per iteration):
    - Each iteration samples ONE config.
    - If sampled config has been evaluated, re-sample.
    - Try up to `max_resample_trials` times; if still duplicated, stop early.
    """

    def __init__(self, config_manager, run, budget, logger=None, fallback=None):
        self.config_manager = config_manager



        self.max_resample_trials = 100

        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []
        self.logger = logger
        self.fallback = fallback
        self.run = run
        self.alive = budget

    def tuning(self):
        start_time = time.time()

        while self.alive > 0:
            print(f"[Run {self.run}] | [RandomSearch] Evaluations left: {self.alive}")

            # Sample ONE unevaluated config, with up to 100 resample trials
            best_config = None
            for t in range(self.max_resample_trials):
                cfg = self.config_manager.sampling_configs(1)[0]
                if tuple(cfg) not in self.evaluated_configs:
                    best_config = cfg
                    break

            if best_config is None:
                print(
                    f"[INFO] RandomSearch stopped early: "
                    f"failed to sample a new config after {self.max_resample_trials} trials."
                )
                break

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
