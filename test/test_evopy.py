"""End to end tests for evopy."""
import numpy as np

from evopy import EvoPy


def test_simple_use_case():
    """Test whether evopy can successfully run for a simple evaluation function."""
    evopy = EvoPy(lambda x: pow(x, 2), 1)
    evopy.run()


def test_multi_dimensional_use_case():
    """Test whether evopy can successfully run for a multi-dimensional evaluation function
    (the Rastrigin function)."""
    evopy = EvoPy(lambda X: 5 + sum([(x ** 2 - 5 * np.cos(2 * np.pi * x)) for x in X]), 2,
                  generations=1000, population_size=100)
    print(evopy.run())


def test_progress_reporting():
    """Test whether all generations are reported."""
    count = [0]

    def reporter(progress_report):
        assert progress_report.generation == count[0]
        count[0] += 1

    evopy = EvoPy(lambda x: pow(x, 2), 1, reporter=reporter)
    evopy.run()

    assert count[0] == 100
