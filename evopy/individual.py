"""Module containing the individuals of the evolutionary strategy algorithm."""
import numpy as np


class Individual:
    """The individual of the evolutionary strategy algorithm.

    This class handles the reproduction of the individual, using both the weights and the specified
    strategy.
    """

    def __init__(self, weights, strategy):
        """Initialize the Individual.

        :param weights: the weights of the individual
        :param strategy: the strategy parameters of the individual
        """
        self.weights = weights
        self.strategy = strategy
        self.fitness = None

    def reproduce(self):
        """Create a single offspring individual from the set weights and strategy.

        :return: an individual which is the offspring of the current instance
        """
        new_weights = self.weights + self.strategy * np.random.randn(len(self.weights))
        scale_factor = np.random.randn() / (2 * len(self.weights))
        new_strategy = max(self.strategy * np.exp(scale_factor), 0.01)
        return Individual(new_weights, new_strategy)

    def evaluate(self, fitness_function):
        """Evaluate the weights of the individual using the provided fitness function.

        :param fitness_function: the fitness function to evaluate the individual with
        :return: the value of the fitness function using the individuals weight
        """
        self.fitness = fitness_function(self.weights)

        return self.fitness
