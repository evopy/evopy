"""Module containing the individuals of the evolutionary strategy algorithm."""
import numpy as np

from evopy.strategy import Strategy


class Individual:
    """The individual of the evolutionary strategy algorithm.

    This class handles the reproduction of the individual, using both the genotype and the specified
    strategy.
    """

    def __init__(self, genotype, strategy, strategy_parameters):
        """Initialize the Individual.

        :param genotype: the genotype of the individual
        :param strategy: the strategy chosen to reproduce. See the Strategy enum for more
                         information
        :param strategy_parameters: the parameters required for the given strategy, as a list
        """
        self.genotype = genotype
        self.length = len(genotype)
        self.fitness = None
        if strategy == Strategy.SINGLE_VARIANCE and len(strategy_parameters) == 1:
            self.strategy = strategy
            self.strategy_parameters = strategy_parameters
            self.reproduce = self._reproduce_single_variance
        elif strategy == Strategy.MULTIPLE_VARIANCE and len(strategy_parameters) == self.length:
            self.strategy = strategy
            self.strategy_parameters = strategy_parameters
            self.reproduce = self._reproduce_multiple_variance

    def evaluate(self, fitness_function):
        """Evaluate the genotype of the individual using the provided fitness function.

        :param fitness_function: the fitness function to evaluate the individual with
        :return: the value of the fitness function using the individuals genotype
        """
        self.fitness = fitness_function(self.genotype)

        return self.fitness

    def _reproduce_single_variance(self):
        """Create a single offspring individual from the set genotype and strategy parameters.

        This function uses the single variance strategy.

        :return: an individual which is the offspring of the current instance
        """
        new_genotype = self.genotype + \
                       self.strategy_parameters[0] * np.random.randn(self.length)
        scale_factor = np.random.randn() / (2 * self.length)
        new_strategy = [max(self.strategy_parameters[0] * np.exp(scale_factor), 0.01)]
        return Individual(new_genotype, self.strategy, new_strategy)

    def _reproduce_multiple_variance(self):
        """Create a single offspring individual from the set genotype and strategy.

        This function uses the multiple variance strategy.

        :return: an individual which is the offspring of the current instance
        """
        new_genotype = self.genotype + [self.strategy_parameters[i] * np.random.randn()
                                        for i in range(self.length)]
        global_scale_factor = np.random.randn() / (2 * self.length)
        scale_factors = [np.random.randn() / 2 * np.sqrt(self.length)]
        new_strategy = [self.strategy_parameters[i] + np.exp(global_scale_factor + scale_factors[i])
                        for i in range(self.length)]
        return Individual(new_genotype, self.strategy, new_strategy)
