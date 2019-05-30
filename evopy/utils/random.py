"""A utility module for random number generation."""
import numpy as np


def random_with_seed(seed):
    """Return RandomState instances based on given seed.

    :param seed: the seed to use for the random number generator
    """
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('Seed must either be an integer or an instance of numpy.random.RandomState')
