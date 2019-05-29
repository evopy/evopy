"""End to end tests for evopy."""
from evopy import EvoPy, Strategy


def test_simple_single_variance():
    """Test whether evopy can successfully run for a simple evaluation function, with
    the single variance strategy."""
    evopy = EvoPy(lambda x: pow(x, 2), 1, strategy=Strategy.SINGLE_VARIANCE)
    evopy.run()


def test_simple_multiple_variance():
    """Test whether evopy can successfully run for a simple evaluation function, with
    the multiple variance strategy."""
    evopy = EvoPy(lambda x: pow(x[0], 2) + pow(x[1], 3), 2, strategy=Strategy.MULTIPLE_VARIANCE)
    evopy.run()


def test_simple_full_variance():
    """Test whether evopy can successfully run for a simple evaluation function, with
    the full variance strategy."""
    evopy = EvoPy(lambda x: pow(x[0], 2) + pow(x[1], 3), 2, strategy=Strategy.FULL_VARIANCE)
    evopy.run()


def test_progress_reporting():
    """Test whether all generations are reported."""
    count = [0]

    def reporter(progress_report):
        assert progress_report.generation == count[0]
        count[0] += 1

    evopy = EvoPy(lambda x: pow(x, 2), 1, reporter=reporter)
    evopy.run()

    assert count[0] == 100
