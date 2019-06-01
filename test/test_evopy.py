"""End to end tests for evopy."""
import numpy as np

from evopy import EvoPy


def test_random_consistency():
    """Test whether the random state gives consistent runs when initialized."""
    x_first = EvoPy(lambda x: pow(x, 2), 1, random_seed=42).run()
    x_second = EvoPy(lambda x: pow(x, 2), 1, random_seed=42).run()
    assert x_first == x_second


def test_random_consistency_multiple_runs():
    """"Test whether the random state is not re-used in sequential runs"""
    evopy = EvoPy(lambda x: pow(x, 2), 1, random_seed=42)
    x_first = evopy.run()
    x_second = evopy.run()
    assert x_first != x_second


def test_simple_use_case():
    """Test whether evopy can successfully run for a simple evaluation function."""
    evopy = EvoPy(lambda x: pow(x, 2), 1)
    best_individual = evopy.run()
    assert best_individual is not None
    assert isinstance(best_individual, np.ndarray)
    assert best_individual.size == 1


def test_early_timed_stop():
    """Test whether evopy can successfully stop early when given a specified time constraint."""
    count = [0]

    def increment_reporter(report):
        count[0] = report.generation + 1

    evopy = EvoPy(lambda x: pow(x, 2), 1, max_run_time=0, reporter=increment_reporter)
    evopy.run()

    assert count[0] == 1


def test_early_target_value_stop():
    """Test whether evopy can successfully stop after achieving a specified target value."""
    count = [0]

    def increment_reporter(report):
        count[0] = report.generation + 1

    evopy = EvoPy(lambda x: 0, 1, target_fitness_value=0, reporter=increment_reporter)
    evopy.run()

    assert count[0] == 1


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
