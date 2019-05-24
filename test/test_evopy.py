"""End to end tests for evopy."""
from evopy import EvoPy, Strategy


def simple_single_test():
    """Test whether evopy can successfully run for a simple evaluation function, with
    the single variance strategy."""
    evopy = EvoPy(lambda x: pow(x, 2), 1, strategy=Strategy.SINGLE_VARIANCE)
    evopy.run()


def simple_multiple_test():
    """Test whether evopy can successfully run for a simple evaluation function, with
    the multiple variance strategy."""
    evopy = EvoPy(lambda x: pow(x, 2), 1, strategy=Strategy.MULTIPLE_VARIANCE)
    evopy.run()
