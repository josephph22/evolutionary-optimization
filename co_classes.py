import numpy as np
from abc import ABC, abstractmethod

class Individual(ABC):
    def __init__(self, value=None, init_params=None):
        if value is not None:
            self.value = value
        else:
            self.value = self._random_init(init_params)

    @abstractmethod
    def pair(self, other, pair_params):
        pass

    @abstractmethod
    def mutate(self, mutate_params):
        pass

    @abstractmethod
    def _random_init(self, init_params):
        pass

class QAP(Individual):
    def __init__(self, value=None, init_params=None):
        super().__init__(value, init_params)

    def pair(self, other, pair_params):
        try:
            self_head = self.value[:int(len(self.value) * pair_params['alpha'])].copy()
            self_tail = self.value[int(len(self.value) * pair_params['alpha']):].copy()
            other_tail = other.value[int(len(other.value) * pair_params['alpha']):].copy()

            mapping = {other_tail[i]: self_tail[i] for i in range(len(self_tail))}

            for i in range(len(self_head)):
                while self_head[i] in other_tail:
                    self_head[i] = mapping[self_head[i]]

            return QAP(np.hstack([self_head, other_tail]))
        except (ValueError, KeyError) as e:
            print(f"Crossover error: {e}")
            return QAP(self.value.copy())

    def mutate(self, mutate_params):
        value = self.value.copy()
        for _ in range(mutate_params['rate']):
            i, j = np.random.choice(range(len(value)), 2, replace=False)
            value[i], value[j] = value[j], value[i]
        self.value = value

    def _random_init(self, init_params):
        return np.random.permutation(init_params['n_facilities'])

class Population:
    def __init__(self, size, fitness, individual_class, init_params):
        self.fitness = fitness
        self.individuals = [individual_class(init_params=init_params) for _ in range(size)]
        if self.fitness is not None:
            self.individuals.sort(key=lambda x: self.fitness(x, 0), reverse=False)

    def replace(self, new_individuals):
        size = len(self.individuals)
        # Keep top 15% as elite
        self.individuals.sort(key=lambda x: self.fitness(x, 0), reverse=False)
        elite_size = int(0.15 * size)
        elite = self.individuals[:elite_size]
        # Keep top 25% of sorted individuals
        keep_size = int(0.25 * size)
        keep = self.individuals[:keep_size]
        # Add new individuals
        self.individuals = elite + keep + new_individuals
        # Maintain diversity
        random_size = size - len(self.individuals)
        if random_size > 0:
            random_indices = np.random.choice(range(len(new_individuals)), random_size, replace=False)
            random_individuals = [new_individuals[i] for i in random_indices]
            self.individuals += random_individuals
        # Ensure population size
        self.individuals = self.individuals[:size]
        self.individuals.sort(key=lambda x: self.fitness(x, 0), reverse=False)

    def get_parents(self, n_offsprings):
        mothers = []
        fathers = []
        tournament_size = 15
        for _ in range(n_offsprings):
            tournament = np.random.choice(self.individuals, tournament_size, replace=True)
            mother = min(tournament, key=lambda x: self.fitness(x, 0))
            tournament = np.random.choice(self.individuals, tournament_size, replace=True)
            father = min(tournament, key=lambda x: self.fitness(x, 0))
            mothers.append(mother)
            fathers.append(father)
        return mothers, fathers

class CCEvolution:
    def __init__(self, pool_size, individual_class, n_offsprings, pair_params, mutate_params, init_params_list):
        self.pool_size = pool_size
        self.pair_params = pair_params
        self.mutate_params = mutate_params
        self.n_offsprings = n_offsprings
        self.pools = [
            Population(pool_size, None, individual_class, init_params)
            for init_params in init_params_list
        ]

    def step(self):
        for pool in self.pools:
            mothers, fathers = pool.get_parents(self.n_offsprings)
            offsprings = []
            for mother, father in zip(mothers, fathers):
                for _ in range(2):
                    offspring = mother.pair(father, self.pair_params)
                    offspring.mutate(self.mutate_params)
                    offsprings.append(offspring)
            pool.replace(offsprings)