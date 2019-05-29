"""End to end tests for evopy."""
import numpy as np

from evopy import EvoPy


def test_simple_use_case():
    """Test whether evopy can successfully run for a simple evaluation function."""
    evopy = EvoPy(lambda x: pow(x, 2), 1)
    best_individual = evopy.run()
    assert best_individual is not None
    assert isinstance(best_individual, np.ndarray)
    assert best_individual.size == 1


def test_empty_input_array():
    """Test whether evopy can successfully run for a simple evaluation function."""
    evopy = EvoPy(lambda x: 0, 0)
    assert evopy.run() is None


def test_multi_dimensional_use_case():
    """Test whether evopy can successfully run for a multi-dimensional evaluation function
    (the Rastrigin function)."""
    evopy = EvoPy(lambda X: 5 + sum([(x ** 2 - 5 * np.cos(2 * np.pi * x)) for x in X]), 2,
                  generations=1000, population_size=100)
    best_individual = evopy.run()
    assert best_individual is not None
    assert isinstance(best_individual, np.ndarray)
    assert best_individual.size == 2
