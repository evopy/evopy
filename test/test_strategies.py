"""Tests for the different sampling strategies implemented in evopy."""
from unittest import TestCase

from parameterized import parameterized

from evopy import EvoPy, Strategy


def single_dimensional_fitness(individual):
    """A simple single-dimensional fitness function."""
    return pow(individual, 2)


def multi_dimensional_fitness(individual):
    """A simple two-dimensional fitness function."""
    return pow(individual[0], 2) + pow(individual[1], 3)


class TestStrategies(TestCase):
    """Tests the different strategies in evopy against single and multi dimensional problems."""

    @parameterized.expand([
        [Strategy.SINGLE_VARIANCE, False],
        [Strategy.SINGLE_VARIANCE, True],
        [Strategy.MULTIPLE_VARIANCE, False],
        [Strategy.MULTIPLE_VARIANCE, True],
        [Strategy.FULL_VARIANCE, False],
        [Strategy.FULL_VARIANCE, True],
    ])
    def test_strategies(self, strategy, multi_dimensional):
        """Tests the given strategy with the given problem dimensionality."""
        if multi_dimensional:
            fitness = multi_dimensional_fitness
        else:
            fitness = single_dimensional_fitness

        evopy = EvoPy(fitness, 2 if multi_dimensional else 1, strategy=strategy)
        evopy.run()
