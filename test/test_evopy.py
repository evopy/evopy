"""End to end tests for evopy."""
import evopy


def simple_test():
    """Test whether evopy can successfully run for a simple evaluation function."""
    evopy.run(lambda x: pow(x, 2), 1)
