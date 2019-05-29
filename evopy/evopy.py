"""Module used for the execution of the evolutionary algorithm."""
import numpy as np

from evopy.individual import Individual
from evopy.strategy import Strategy
from evopy.utils import random_with_seed
from evopy.progress_report import ProgressReport


class EvoPy:
    """Main class of the EvoPy package."""

    def __init__(self, fitness_function, individual_length, warm_start=None, generations=100,
                 population_size=30, num_children=1, mean=0, std=1, maximize=False,
                 strategy=Strategy.SINGLE_VARIANCE, random_seed=None, reporter=None):
        """Initializes an EvoPy instance.

        :param fitness_function: the fitness function on which the individuals are evaluated
        :param individual_length: the length of each individual
        :param warm_start: the individual to start from
        :param generations: the number of generations to execute
        :param population_size: the population size of each generation
        :param num_children: the number of children generated per parent individual
        :param mean: the mean for sampling the random offsets of the initial population
        :param std: the standard deviation for sampling the random offsets of the initial population
        :param maximize: whether the fitness function should be maximized or minimized
        :param strategy: the strategy used to generate offspring by individuals. For more
                         information, check the Strategy enum
        :param random_seed: the seed to use for the random number generator
        :param reporter: callback to be invoked at each generation with a ProgressReport as argument
        """
        self.fitness_function = fitness_function
        self.individual_length = individual_length
        self.warm_start = np.zeros(self.individual_length) if warm_start is None else warm_start
        self.generations = generations
        self.population_size = population_size
        self.num_children = num_children
        self.mean = mean
        self.std = std
        self.maximize = maximize
        self.strategy = strategy
        self.random_seed = random_seed
        self.reporter = reporter

    def run(self):
        """Run the evolutionary strategy algorithm.

        :return the best genotype found
        """
        if self.individual_length == 0:
            return None

        population = self._init_population()
        best = sorted(population, reverse=self.maximize,
                      key=lambda individual: individual.evaluate(self.fitness_function))[0]

        for generation in range(self.generations):
            children = [parent.reproduce() for _ in range(self.num_children)
                        for parent in population]
            population = sorted(children + population, reverse=self.maximize,
                                key=lambda individual: individual.evaluate(self.fitness_function))
            population = population[:self.population_size]
            if not self.maximize:
                best = population[0] if population[0].fitness > best.fitness else best
            else:
                best = population[0] if population[0].fitness < best.fitness else best

            if self.reporter is not None:
                self.reporter(ProgressReport(generation, best.genotype, best.fitness))

        return best.genotype

    def _init_population(self):
        if self.strategy == Strategy.SINGLE_VARIANCE:
            strategy_parameters = [random_with_seed(self.random_seed).randn()]
        elif self.strategy == Strategy.MULTIPLE_VARIANCE:
            strategy_parameters = random_with_seed(self.random_seed).randn(self.individual_length)
        elif self.strategy == Strategy.FULL_VARIANCE:
            strategy_parameters = random_with_seed(self.random_seed).randn(
                int((self.individual_length + 1) * self.individual_length / 2))
        else:
            raise ValueError("Provided strategy parameter was not an instance of Strategy")
        return [
            Individual(
                self.warm_start + random_with_seed(self.random_seed).normal(
                    loc=self.mean, scale=self.std, size=self.individual_length
                ),
                self.strategy, strategy_parameters,
                random_seed=self.random_seed
            ) for _ in range(self.population_size)
        ]
