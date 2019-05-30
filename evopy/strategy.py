"""Module containing enum Strategy describing different strategies available to the algorithm."""
from enum import Enum


class Strategy(Enum):
    """Enum used to distinguish different types of strategies.

    These strategies are used to determine the mechanism which each individual can use to control
    its own mutability. The three strategies which are included are:

    - SINGLE_VARIANCE: the same variance is used for each allele, no covariances
    - MULTIPLE_VARIANCE: each allele has its own variance, no covariances
    - FULL VARIANCE: each allele has its own variance, complete variances
                     (encoded as rotation angles)
    """
    SINGLE_VARIANCE = 1
    MULTIPLE_VARIANCE = 2
    FULL_VARIANCE = 3
