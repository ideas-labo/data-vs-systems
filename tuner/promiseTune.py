import time

import numpy as np
import random

from typing import List, Tuple
from copy import deepcopy

import pandas as pd
from scipy.spatial.distance import cdist
import copy
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import lingam
from causallearn.search.ConstraintBased.FCI import fci
from tqdm import tqdm


class PromiseTuner:
    def __init__(self, config_manager, run, budget, logger=None, fallback=None):

        self.config_manager = config_manager

        # PromiseTune parameters
        self.initial_size = 10
        self.l_value = 10
        self.k_percent = 0.1
        self.evaluated_data = []

        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []
        self.logger = logger
        self.fallback = fallback  # Fallback strategy for configurations not found in the dataset
        self.run = run
        self.alive = budget
        self.budget = budget

    def tuning(self):
        start_time = time.time()

        init_configs = self.config_manager.sampling_configs(self.initial_size)
        t0 = time.time()
        evaluated_configs, self.alive = self.config_manager.evaluate(init_configs, self.fallback, self.alive)
        t1 = time.time()

        print(f"[TIME] initial evaluate: {t1 - t0:.4f}s")

        for cfg, perf in evaluated_configs:
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))
            self.evaluated_data.append((cfg, perf))

        refresh_interval = 10
        purified_rules = None
        last_refresh_eval_count = -1

        while self.alive > 0:
            loop_start = time.time()
            current_eval_count = len(self.evaluated_data)
            if purified_rules is None or current_eval_count-last_refresh_eval_count >= refresh_interval:
                t_rule0 = time.time()
                rules = self.extract_rules_from_rf()
                t_rule1 = time.time()
                rule_features, labels = self.featurize_rules(rules)
                t_rule2 = time.time()
                purified_rules = self.purify_rules_by_causality(rule_features, labels, rules)
                t_rule3 = time.time()
                last_refresh_eval_count = current_eval_count

                print(f"[TIME] extract_rules_from_rf: {t_rule1 - t_rule0:.4f}s")
                print(f"[TIME] featurize_rules: {t_rule2 - t_rule1:.4f}s")
                print(f"[TIME] purify_rules_by_causality: {t_rule3 - t_rule2:.4f}s")
                print(f"[INFO] #rules={len(rules)}, #purified_rules={len(purified_rules)}")

            t_cand0 = time.time()
            candidate_pool = self.sample_candidates_with_rules(purified_rules)
            t_cand1 = time.time()
            print(f"[TIME] sample_candidates_with_rules: {t_cand1 - t_cand0:.4f}s")
            print(f"[INFO] #candidate_pool={len(candidate_pool)}")

            t_sel0 = time.time()
            best_config = self.select_by_ei(candidate_pool, self.config_manager.value_domains)
            t_sel1 = time.time()
            print(f"[TIME] select_by_ei: {t_sel1 - t_sel0:.4f}s")

            t_eval0 = time.time()
            evaluated_configs, self.alive = self.config_manager.evaluate([best_config], self.fallback, self.alive)
            t_eval1 = time.time()
            print(f"[TIME] evaluate one config: {t_eval1 - t_eval0:.4f}s")

            for cfg, perf in evaluated_configs:
                self.evaluated_configs[tuple(cfg)] = perf
                self.all_results.append((cfg, perf))
                self.evaluated_data.append((cfg, perf))

            loop_end = time.time()
            print(
                f"[Run {self.run}] | [PromiseTune] Evaluations left: {self.alive} | loop time: {loop_end - loop_start:.4f}s")

            print(f"[Run {self.run}] | [PromiseTune] Evaluations left: {self.alive}")

            #

        self.logger.log_all_results(self.all_results)
        end_time = time.time()
        self.logger.store_runtime_to_csv(end_time - start_time)
        return None

    def extract_rules_from_rf(self):
        configs = [c for c, _ in self.evaluated_data]
        y = [p for _, p in self.evaluated_data]
        feature_names = self.config_manager.get_config_header()[:-1]
        # feature_names = list(self.target_system.knobs_info.keys())
        rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=self.l_value)
        x = self.preprocess_configs_with_knobs_info(configs)
        rf.fit(x, y)
        all_rules = []
        for tree in rf.estimators_:
            rules = self.get_tree_paths(tree, feature_names)
            all_rules.extend([r for r in rules if r not in all_rules])
        return all_rules

    def get_tree_paths(self, tree, feature_names):
        paths = []
        stack = [(0, [])]
        while stack:
            node_id, path = stack.pop()
            feature_index = tree.tree_.feature[node_id]
            threshold = tree.tree_.threshold[node_id]
            if feature_index != -2:
                name = feature_names[feature_index]
                stack.append((tree.tree_.children_left[node_id], path + [(name, threshold, 'L')]))
                stack.append((tree.tree_.children_right[node_id], path + [(name, threshold, 'R')]))
            else:
                paths.append(path)
        return paths

    def featurize_rules(self, rules):
        configs = [c for c, _ in self.evaluated_data]
        feature_names = self.config_manager.get_config_header()[:-1]
        feature_matrix = []
        for config in configs:
            row = []
            for rule in rules:
                row.append(1 if self.config_in_rule(config, rule, feature_names) else 0)
            feature_matrix.append(row)
        labels = [p for _, p in self.evaluated_data]
        return feature_matrix, labels

    def config_in_rule(self, config, rule, feature_names):
        value_domains = self.config_manager.value_domains
        for key, threshold, direction in rule:

            if key not in feature_names:
                raise KeyError(f"Feature '{key}' not found in dataset feature list.")
            idx = feature_names.index(key)

            val = config[idx]
            domain = value_domains[idx]
            if isinstance(val, str) or not isinstance(val, (int, float)):
                val = domain.index(val)

            if direction == 'L' and not val <= threshold:
                return False
            if direction == 'R' and not val > threshold:
                return False
        return True

    def purify_rules_by_causality(self, X, y, rules):
        """
        Perform causal filtering using DirectLiNGAM on rule-feature matrix.
        Rules with negative ACE (average causal effect) on performance are retained.
        """

        if not rules:
            return []

            # 先准备一个轻量兜底版本
        filtered_rules = self.quick_filter_rules(X, y, rules, top_k=20, min_support=2)
        if not filtered_rules:
            return []

        # 1. 构造 DataFrame：每列是一个 rule_feature，最后一列是 performance
        df = pd.DataFrame(X, columns=[f'rule_{i}' for i in range(len(rules))])
        df['perf'] = y
        data = df.to_numpy()
        num_features = data.shape[1]

        # 2. 初始化因果结构 prior_knowledge
        try:
            graph, _ = fci(data, show_progress=False, verbose=False)
            adjacency_matrix = self.get_adjacency_matrix(graph)
            if adjacency_matrix.shape != (num_features, num_features):
                raise ValueError("Adjacency matrix shape mismatch.")
            adjacency_matrix = adjacency_matrix.T
            adjacency_matrix[:, -1] = 0  # 保证 performance 没有 outgoing edge
        except Exception as e:
            print(f"[DEBUG] FCI failed or matrix invalid: {e}", flush=True)
            return filtered_rules
            # print(f"[DEBUG] FCI failed or matrix invalid: {e}")
            # adjacency_matrix = np.zeros((num_features, num_features), dtype=int)

        # 3. 构造 prior_knowledge
        prior_knowledge = adjacency_matrix.copy()
        prior_knowledge[-1, :-1] = 1  # perf 可以被所有 rule 影响
        prior_knowledge[-1, -1] = -1  # perf 不影响任何其他变量

        # 4. 利用 DirectLiNGAM 拟合因果模型
        purified = []
        try:
            model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
            model.fit(data)
            target_idx = num_features - 1
            print(f"[DEBUG] Num samples: {len(y)}, Num rules: {len(rules)}")
            for i in range(len(rules)):
                ace = model.estimate_total_effect(data, i, target_idx)
                print(f"[DEBUG] ACE[{i}]: {ace:.4f}")
                if ace < 0:
                    purified.append(rules[i])
        except Exception as e:
            print(f"[DEBUG] Causality analysis failed: {e}")
            return filtered_rules

        return purified if purified else filtered_rules

    def get_adjacency_matrix(self, graph):
        nodes = graph.get_nodes()
        node_count = len(nodes)
        matrix = np.zeros((node_count, node_count), dtype=int)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        for edge in graph.get_graph_edges():
            i = node_index[edge.node1]
            j = node_index[edge.node2]
            matrix[i, j] = 1
        return matrix

    def sample_candidates_with_rules(self, rules):
        value_domains = self.config_manager.value_domains
        feature_names = self.config_manager.get_config_header()[:-1]

        all_candidates = []

        configs = [c for c, _ in self.evaluated_data]
        perfs = [p for _, p in self.evaluated_data]
        X = self.preprocess_configs_with_knobs_info(configs)
        model = RandomForestRegressor()
        model.fit(X, perfs)
        estimators = model.estimators_

        eta = min(perfs) if self.config_manager.optimization_goal == 'minimum' else max(perfs)

        # fallback if no rule
        if not rules:
            candidates, _ = self.ei_guided_sampling_from_domain(value_domains, estimators, eta)
            return candidates

        for rule in rules:
            modified_knobs_info = self.modify_domains_by_rule(rule, value_domains, feature_names)
            if modified_knobs_info is None:
                continue
            candidates, _ = self.ei_guided_sampling_from_domain(modified_knobs_info, estimators, eta)
            all_candidates.extend(candidates)

        return all_candidates

    def modify_domains_by_rule(self, rule, original_domains, feature_names):
        """
        按规则裁剪每一维的取值域（dataset mode）。
        - 枚举维：阈值视为“索引阈值”，保留 index<=thr 或 >thr（保持原顺序）
        - 数值维：阈值视为“数值阈值”，保留 <=thr 或 >thr 的取值（按域中出现的值）
        返回裁剪后的 domains；若某维被裁到空，返回 None。
        """
        domains = copy.deepcopy(original_domains)

        for r_knob, threshold, direction in rule:
            if r_knob not in feature_names:
                # 规则中的特征不在数据集里，跳过此约束
                continue
            idx = feature_names.index(r_knob)
            dom = domains[idx]

            # 判定该维是否枚举（包含字符串或非纯数字）
            is_enum = any(isinstance(v, str) or not isinstance(v, (int, float)) for v in dom)

            try:
                # 规则生成阶段对枚举维通常给的是“索引阈值”
                thr = int(threshold) if is_enum else float(threshold)
            except Exception:
                return None

            if is_enum:
                # 枚举按“索引阈值”裁剪（保持原域顺序）
                if direction == 'L':
                    new_dom = [v for i, v in enumerate(dom) if i <= thr]
                else:  # 'R'
                    new_dom = [v for i, v in enumerate(dom) if i > thr]
            else:
                # 数值维按“数值阈值”裁剪（域本身是有限值集）
                if direction == 'L':
                    new_dom = [v for v in dom if float(v) <= thr]
                else:  # 'R'
                    new_dom = [v for v in dom if float(v) > thr]

            if not new_dom:
                return None
            domains[idx] = new_dom

        return domains

    # def ei_guided_sampling_from_domain(self, value_domains, estimators, eta, max_iterations=1000,
    #                                    stop_threshold=0.01):
    #
    #     kde_interval = 500  # 每多少次评估做一次 KDE
    #     min_kde_points = 50  # 至少多少个点后再启用 KDE 判断
    #     feature_names = self.config_manager.get_config_header()[:-1]
    #     values, configs, best = [], [], float('-inf')
    #
    #     for i in range(max_iterations):
    #
    #         sampled_config = self.config_manager.sampling_configs(1, value_domains)[0]
    #         x = self.preprocess_configs_with_knobs_info([sampled_config])[0]
    #         preds = [e.predict([x])[0] for e in estimators]
    #         mean = np.mean(preds)
    #         std = np.std(preds)
    #         if self.config_manager.optimization_goal == 'minimum':  # minimization
    #             z = (eta - mean) / std if std > 0 else 0
    #             ei = (eta - mean) * norm.cdf(z) + std * norm.pdf(z) if std > 0 else 0
    #         else:  # throughput: maximization
    #             z = (mean - eta) / std if std > 0 else 0
    #             ei = (mean - eta) * norm.cdf(z) + std * norm.pdf(z) if std > 0 else 0
    #
    #         values.append(ei)
    #         configs.append(sampled_config)
    #         best = max(best, ei)
    #
    #         # if i > 0:  # kde requires at least two points
    #         #     try:
    #         #         kde = stats.gaussian_kde(values)
    #         #         if 1 - kde.integrate_box_1d(-np.inf, best) < stop_threshold:
    #         #             break
    #         #     except Exception as e:
    #         #         break
    #
    #         # ---- 只在满足两个条件时才做 KDE 检查 ----
    #         # 1) 样本数量 >= min_kde_points
    #         # 2) 当前迭代索引满足间隔（例如每 500 次）
    #         if i >= min_kde_points and ((i + 1) % kde_interval == 0):
    #             try:
    #                 v = np.array(values)
    #                 kde = stats.gaussian_kde(v)
    #                 tail_mass = 1 - kde.integrate_box_1d(-np.inf, best)
    #                 if tail_mass < stop_threshold:
    #                     break
    #             except Exception:
    #                 # KDE 偶尔会数值不稳定，直接忽略错误，继续搜索
    #                 pass
    #
    #     return configs, best

    def ei_guided_sampling_from_domain(self, value_domains, estimators, eta, n_samples=1000):
        """
        Sample a batch of configurations from the given domain, compute EI in batch,
        and return only the single best candidate.
        """
        # 1. 一次性采样
        sampled_configs = self.config_manager.sampling_configs(n_samples, value_domains)
        if not sampled_configs:
            return None, None

        # 2. 批量预处理
        X = self.preprocess_configs_with_knobs_info(sampled_configs)

        # 3. 批量预测：shape = (n_samples, n_estimators)
        preds = np.array([e.predict(X) for e in estimators]).T

        mean = preds.mean(axis=1)
        std = preds.std(axis=1)
        std[std == 0] = 1e-8  # 防止除零

        # 4. 批量计算 EI
        if self.config_manager.optimization_goal == 'minimum':
            z = (eta - mean) / std
            ei = (eta - mean) * norm.cdf(z) + std * norm.pdf(z)
        else:
            z = (mean - eta) / std
            ei = (mean - eta) * norm.cdf(z) + std * norm.pdf(z)

        # 5. 只取 EI 最好的那个
        best_idx = np.argmax(ei)
        best_config = sampled_configs[best_idx]
        best_ei = ei[best_idx]

        return [best_config], best_ei

    def select_by_ei(self, candidates, value_domains):
        if not candidates:
            return random.choice(self.config_manager.sampling_configs(1, value_domains))
        train_x = self.preprocess_configs_with_knobs_info([c for c, _ in self.evaluated_data])
        train_y = [p for _, p in self.evaluated_data]
        model = RandomForestRegressor()
        model.fit(train_x, train_y)

        test_x = self.preprocess_configs_with_knobs_info(candidates)
        preds = np.array([e.predict(test_x) for e in model.estimators_]).T
        mean = preds.mean(axis=1)
        std = preds.std(axis=1)
        std[std == 0] = 1e-8
        best_seen = min(train_y) if self.config_manager.optimization_goal == 'minimum' else max(train_y)
        if self.config_manager.optimization_goal == 'minimum':  # minimization
            z = (best_seen - mean) / std
            ei = (best_seen - mean) * norm.cdf(z) + std * norm.pdf(z)
        else:  # maximization
            z = (mean - best_seen) / std
            ei = (mean - best_seen) * norm.cdf(z) + std * norm.pdf(z)

        return candidates[np.argmax(ei)]

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

    def quick_filter_rules(self, X, y, rules, top_k=20, min_support=2):
        """
        Cheap rule filtering without causal discovery.
        Keep rules that:
        1) are not constant
        2) have enough support
        3) lead to better average performance on covered samples
        """
        if not rules:
            return []

        X_arr = np.asarray(X)  # shape = (n_samples, n_rules)
        y_arr = np.asarray(y, dtype=float)

        global_mean = np.mean(y_arr)
        scored = []

        for j, rule in enumerate(rules):
            col = X_arr[:, j]

            # 去掉常数列
            if np.all(col == col[0]):
                continue

            covered_idx = np.where(col == 1)[0]
            support = len(covered_idx)

            # 去掉覆盖太少的 rule
            if support < min_support:
                continue

            covered_mean = np.mean(y_arr[covered_idx])

            # minimization: covered_mean 越小越好
            # maximization: covered_mean 越大越好
            if self.config_manager.optimization_goal == 'minimum':
                improvement = global_mean - covered_mean
            else:
                improvement = covered_mean - global_mean

            # 一个简单分数：改进幅度 * 支持度权重
            score = improvement * np.log1p(support)

            scored.append((score, support, rule))

        # 分数高的在前
        scored.sort(key=lambda x: x[0], reverse=True)

        filtered_rules = [rule for _, _, rule in scored[:top_k]]
        return filtered_rules