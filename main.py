import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from utils.logger import Logger
from utils.config_manager import ConfigManager
from tuner.ga import GATuner
from tuner.flash import FLASHTuner
from tuner.smac import SMACTuner
from tuner.promiseTune import PromiseTuner
from tuner.hebo import HEBOTuner
from tuner.bestconfig import BestConfigTuner
from tuner.random_search import RandomSearchTuner
from tuner.bounceTuner import BounceTuner
from tuner.flash_pool import FLASHPoolTuner
from tuner.random_search_pool import RandomSearchPoolTuner
from tuner.ezr_pool import EZRPoolTuner
from tuner.line_pool import LINEPoolTuner


def run_one_experiment(fallback, run_id, budgets, systems, compared_algorithms):
    """
    单个 (fallback, run_id) 的完整实验流程：
    - 遍历 system
    - 遍历 scale (dataset)
    - 遍历 algorithm
    完全复用你原来的逻辑，只是从 main 里抽出来，方便多线程调用。
    """
    start_time = time.time()
    print(f"\n========================Run {run_id + 1} (fallback={fallback}) START==========================")

    for system in systems:
        budget = budgets[system]
        optimization_goal = 'maximum' if system in ['mysql', 'postgresql', 'tomcat', 'httpd'] else 'minimum'
        scales = sorted([f[:-4] for f in os.listdir(f'datasets/{system}') if f.endswith('.csv')])

        config_manager = ConfigManager(system, optimization_goal)

        for scale_id, scale_name in enumerate(scales):
            config_manager.set_system_scale(scale_name)

            for algo_name in compared_algorithms:
                header = config_manager.get_config_header()
                logger = Logger(
                    base_dir=f"results/run{run_id}/{fallback}",
                    algorithm=f"{algo_name}",
                    system=f"{system}",
                    scale=f"{scale_name}",
                    header=header
                )

                print(
                    f"\n[INFO] Running {algo_name} on {system} "
                    f"| run: {run_id} | scale: {scale_name} | fallback: {fallback}"
                )

                if algo_name == 'ga':
                    tuner = GATuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'flash':
                    tuner = FLASHTuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'smac':
                    tuner = SMACTuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'promisetune':
                    tuner = PromiseTuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'hebo':
                    tuner = HEBOTuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'bestconfig':
                    tuner = BestConfigTuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'random_search':
                    tuner = RandomSearchTuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'bounce':
                    tuner = BounceTuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'flash_pool':
                    tuner = FLASHPoolTuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'random_search_pool':
                    tuner = RandomSearchPoolTuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'ezr_pool':
                    tuner = EZRPoolTuner(config_manager, run_id, budget, logger, fallback)
                elif algo_name == 'line_pool':
                    tuner = LINEPoolTuner(config_manager, run_id, budget, logger, fallback)
                else:
                    raise ValueError(f"Unknown algorithm: {algo_name}")

                tuner.tuning()

    cost = time.time() - start_time
    print(f"\n========================Run {run_id + 1} (fallback={fallback}) DONE "
          f"in {cost:.2f}s==========================")


def main():
    budgets = {'mysql': 350, 'postgresql': 350, 'httpd': 250, 'tomcat': 250, 'gcc': 350, 'clang': 350, 'x264': 200, 'x265':200}
    runs = 10
    systems = ['mysql', 'postgresql', 'httpd', 'tomcat', 'gcc', 'clang', 'x264', 'x265']
    # compared_algorithms = ['flash', 'ga', 'smac', 'hebo', 'bestconfig']
    # compared_algorithms = ['flash_pool', 'random_search_pool', 'ezr_pool', 'line_pool']
    compared_algorithms = ['promisetune']
    

    # fall back strategies; gp 跟 dt 可以去掉；加sota; 算法方面加一下 pool-based 跟 query-based 的 baseline；
    # fallbacks = ['rf_uncertainty', 'gp', 'rf', 'loess', 'hgb', 'dal_rf']
    fallbacks = ['1NN']

    global_start = time.time()

    # 线程数
    max_workers = runs

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for fallback in fallbacks:
            for i in range(runs):
                # 每个 (fallback, i) 提交成一个独立任务
                futures.append(
                    executor.submit(run_one_experiment, fallback, i, budgets, systems, compared_algorithms)
                )

        # 等所有任务完成（可以顺便打印一下谁先完成）
        for fut in as_completed(futures):
            try:
                fut.result()  # 捕获异常（如果某个线程里抛错误，这里能及时看到）
            except Exception as e:
                print(f"[ERROR] One run failed with error: {e}")

    print(f"\n[INFO] All experiments finished in {time.time() - global_start:.2f}s")


if __name__ == "__main__":
    main()
