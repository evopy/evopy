"""Module containing the individuals of the evolutionary strategy algorithm."""
import numpy as np

from evopy.utils import random_with_seed

class Individual:
    """The individual of the evolutionary strategy algorithm.

    This class handles the reproduction of the individual, using both the genotype and the specified
    strategy.
    """

    def __init__(self, genotype, strategy, random_seed=None):
        """Initialize the Individual.

        :param genotype: the genotype of the individual
        :param strategy: the strategy parameters of the individual
        :param random_seed: the seed to use for the random number generator
        """
        self.genotype = genotype
        self.strategy = strategy
        self.random_seed = random_seed
        self.fitness = None

    def reproduce(self):
        """Create a single offspring individual from the set genotype and strategy.

        :return: an individual which is the offspring of the current instance
        """
        new_genotype = self.genotype + self.strategy * random_with_seed(self.random_seed) \
                    .randn(len(self.genotype))
        scale_factor = random_with_seed(self.random_seed).randn() / (2 * len(self.genotype))
        new_strategy = max(self.strategy * np.exp(scale_factor), 0.01)
        return Individual(new_genotype, new_strategy)

    def evaluate(self, fitness_function):
        """Evaluate the genotype of the individual using the provided fitness function.

        :param fitness_function: the fitness function to evaluate the individual with
        :return: the value of the fitness function using the individuals genotype
        """
        self.fitness = fitness_function(self.genotype)

        return self.fitness
