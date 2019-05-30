"""Tests for (non-)deterministic behavior in evopy."""
import numpy as np
from nose.tools import raises

from evopy.utils import random_with_seed


def random_integer_seed_test():
    """Test if integers are correctly used."""
    random = random_with_seed(42)
    assert random.randint(100) == 51


def random_state_seed_test():
    """Test if states are correctly used."""
    random = random_with_seed(np.random.RandomState(42))
    assert random.randint(100) == 51


def random_none_seed_test():
    """Test if none is given the original random is used."""
    np.random.seed(42)
    random = random_with_seed(None)
    assert random.randint(100) == 51


@raises(ValueError)
def random_invalid_seed_test():
    """Test if an error is raised when an incorrect parameter is supplied."""
    random_with_seed(4.0)
