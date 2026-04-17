import time

import numpy as np
import random

from typing import List, Tuple
from copy import deepcopy
from scipy.spatial.distance import cdist


class GATuner:
    def __init__(self, config_manager, run, budget, logger=None, fallback=None):

        self.config_manager = config_manager

        # GA parameters
        self.pop_size = 20
        self.mutation_rate = 1 / int(self.config_manager.config_dim)
        self.crossover_rate = 0.9

        self.evaluated_configs = {}  # config tuple -> perf
        self.all_results = []
        self.logger = logger
        self.fallback = fallback  # Fallback strategy for configurations not found in the dataset
        self.run = run
        self.alive = budget

    def tuning(self) -> List[Tuple[List[float], float]]:
        start_time = time.time()
        generation = 0
        population = deepcopy(self.config_manager.random_initialize_population(self.pop_size))
        if len(population) < self.pop_size:
            extra = self.config_manager.random_initialize_population(self.pop_size - len(population))
            population.extend(extra)
        evaluated_pop, self.alive = self.config_manager.evaluate(population, self.fallback, self.alive)


        for cfg, perf in evaluated_pop:
            self.evaluated_configs[tuple(cfg)] = perf
            self.all_results.append((cfg, perf))
        parents = self.environment_selection(evaluated_pop, self.pop_size, self.config_manager.optimization_goal)
        self.logger.log_generation(generation, parents)

        while self.alive > 0:
            print(f"[Run {self.run}] | [GA] Generation {generation + 1}")

            offspring = []
            seen = set(tuple(cfg) for cfg, _ in self.all_results)
            while len(offspring) < self.pop_size:
                p1, p2 = self.tournament_selection(parents, self.config_manager.optimization_goal)
                c1, c2 = self.uniform_crossover(p1, p2)
                c1 = self.mutation(c1)
                c2 = self.mutation(c2)

                for child in [c1, c2]:
                    t = tuple(child)
                    if t not in self.evaluated_configs and t not in seen:
                        offspring.append(child)
                        seen.add(t)
                    if len(offspring) >= self.pop_size:
                        break

            evaluated_offspring, self.alive = self.config_manager.evaluate(offspring, self.fallback, self.alive)
            for cfg, perf in evaluated_offspring:
                self.evaluated_configs[tuple(cfg)] = perf
                self.all_results.append((cfg, perf))

            combined = parents + evaluated_offspring
            parents = self.environment_selection(combined, self.pop_size, self.config_manager.optimization_goal)
            generation += 1
            self.logger.log_generation(generation, parents)
        self.logger.log_all_results(self.all_results)
        end_time = time.time()
        self.logger.store_runtime_to_csv(end_time - start_time)
        return parents

    def tournament_selection(self, parents, optimization_goal):
        ind1a, ind1b = random.sample(parents, 2)
        ind2a, ind2b = random.sample(parents, 2)

        if optimization_goal == "maximum":
            p1 = ind1a if ind1a[1] > ind1b[1] else ind1b
            p2 = ind2a if ind2a[1] > ind2b[1] else ind2b
        else:  # assume "minimum"
            p1 = ind1a if ind1a[1] < ind1b[1] else ind1b
            p2 = ind2a if ind2a[1] < ind2b[1] else ind2b

        return p1[0], p2[0]

    def uniform_crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)

        child1, child2 = [], []
        for v1, v2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(v1)
                child2.append(v2)
            else:
                child1.append(v2)
                child2.append(v1)
        return child1, child2

    def mutation(self, individual: List[float]) -> List[float]:
        mutated = individual[:]
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                candidates = self.config_manager.value_domains[i]
                original = mutated[i]
                alternatives = [v for v in candidates if v != original]
                if alternatives:
                    mutated[i] = random.choice(alternatives)
        return mutated

    def environment_selection(self, evaluated_configs, pop_size, optimization_goal):
        reverse = optimization_goal == "maximum"
        sorted_pop = sorted(evaluated_configs, key=lambda x: x[1], reverse=reverse)
        return sorted_pop[:pop_size]
