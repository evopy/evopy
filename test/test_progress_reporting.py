"""Tests for progress reporting functionality."""

from evopy import EvoPy


def test_progress_reporting():
    """Test whether all generations are reported."""
    count = [0]

    def reporter(progress_report):
        assert progress_report.generation == count[0]
        count[0] += 1

    evopy = EvoPy(lambda x: pow(x, 2), 1, reporter=reporter)
    evopy.run()

    assert count[0] == 100
