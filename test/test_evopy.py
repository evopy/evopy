"""End to end tests for evopy."""
from evopy import EvoPy


def simple_test():
    """Test whether evopy can successfully run for a simple evaluation function."""
    evopy = EvoPy(lambda x: pow(x, 2), 1)
    evopy.run()
