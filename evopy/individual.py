"""Module containing the individuals of the evolutionary strategy algorithm."""
import numpy as np

from evopy.strategy import Strategy


class Individual:
    """The individual of the evolutionary strategy algorithm.

    This class handles the reproduction of the individual, using both the genotype and the specified
    strategy.
    """
    _BETA = 0.0873
    _EPSILON = 0.01

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
        self.strategy = strategy
        self.strategy_parameters = strategy_parameters
        if strategy == Strategy.SINGLE_VARIANCE and len(strategy_parameters) == 1:
            self.reproduce = self._reproduce_single_variance
        elif strategy == Strategy.MULTIPLE_VARIANCE and len(strategy_parameters) == self.length:
            self.reproduce = self._reproduce_multiple_variance
        elif strategy == Strategy.FULL_VARIANCE and len(strategy_parameters) == self.length * (
                self.length + 1) / 2:
            self.reproduce = self._reproduce_full_variance

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
        scale_factor = np.random.randn() * np.sqrt(1 / (2 * self.length))
        new_parameters = [max(self.strategy_parameters[0] * np.exp(scale_factor), self._EPSILON)]
        return Individual(new_genotype, self.strategy, new_parameters)

    def _reproduce_multiple_variance(self):
        """Create a single offspring individual from the set genotype and strategy.

        This function uses the multiple variance strategy.

        :return: an individual which is the offspring of the current instance
        """
        new_genotype = self.genotype + [self.strategy_parameters[i] * np.random.randn()
                                        for i in range(self.length)]
        global_scale_factor = np.random.randn() * np.sqrt(1 / (2 * self.length))
        scale_factors = [np.random.randn() * np.sqrt(1 / 2 * np.sqrt(self.length))
                         for _ in range(self.length)]
        new_parameters = [max(np.exp(global_scale_factor + scale_factors[i])
                              * self.strategy_parameters[i], self._EPSILON)
                          for i in range(self.length)]
        return Individual(new_genotype, self.strategy, new_parameters)

    # pylint: disable=C0103
    # Notation used in Evolution Strategies I paper
    def _reproduce_full_variance(self):
        """Create a single offspring individual from the set genotype and strategy.

        This function uses the full variance strategy.

        :return: an individual which is the offspring of the current instance
        """
        global_scale_factor = np.random.randn() * np.sqrt(1 / (2 * self.length))
        scale_factors = [np.random.randn() * np.sqrt(1 / 2 * np.sqrt(self.length))
                         for _ in range(self.length)]
        new_variances = [max(np.exp(global_scale_factor + scale_factors[i])
                             * self.strategy_parameters[i], self._EPSILON)
                         for i in range(self.length)]
        new_rotations = [self.strategy_parameters[i] + np.random.randn() * self._BETA
                         for i in range(self.length, len(self.strategy_parameters))]
        new_rotations = [rotation if rotation < np.pi / 2 else rotation - np.pi
                         for rotation in new_rotations]
        new_rotations = [rotation if rotation > -np.pi / 2 else rotation + np.pi
                         for rotation in new_rotations]
        T = np.identity(self.length)
        for p in range(self.length - 1):
            for q in range(p + 1, self.length):
                j = int((2 * self.length - p) * (p + 1) / 2 - 2 * self.length + q)
                T_pq = np.identity(self.length)
                T_pq[p][p] = T_pq[q][q] = np.cos(new_rotations[j])
                T_pq[p][q] = -np.sin(new_rotations[j])
                T_pq[q][p] = -T_pq[p][q]
                T = np.matmul(T, T_pq)
        new_genotype = self.genotype + np.matmul(T, np.random.randn(self.length))
        return Individual(new_genotype, self.strategy, new_variances + new_rotations)
