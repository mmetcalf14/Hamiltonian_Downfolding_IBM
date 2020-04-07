# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

import math


def round_nearest_halfway_from_zero(x, infinity=1e+20):
    """ Rounds the argument to the nearest integer.

    For values like 1.5 the intetger with greater absolute value is returned.
    This treats positive and negative values in a symmetric manner.
    This is called "round half away from zero"


    Args:
        x: the value to round
        infinity: the model's infinity value. All values above infinity are set to +INF

    Returns:
        an integer value

    Example:
        round_nearest(0) = 0
        round_nearest(1.1) = 1
        round_nearest(1.5) = 2
        round_nearest(1.49) = 1
    """
    if x == 0:
        return 0
    elif x >= infinity:
        return infinity
    elif x <= -infinity:
        return -infinity
    else:
        raw_nearest = my_round_even(x)  # math.floor(x + 0.5)
        return int(raw_nearest)


def my_round_even(number):
    """
    Simplified version from future
    """
    from decimal import Decimal, ROUND_HALF_EVEN

    d = Decimal.from_float(number).quantize(1, rounding=ROUND_HALF_EVEN)
    return int(d)


def round_nearest_towards_infinity(x, infinity=1e+20):
    """ Rounds the argument to the nearest integer.

    For ties like 1.5 the ceiling integer is returned.
    This is called "round towards infinity"

    Args:
        x: the value to round
        infinity: the model's infinity value. All values above infinity are set to +INF

    Returns:
        an integer value

    Example:
        round_nearest(0) = 0
        round_nearest(1.1) = 1
        round_nearest(1.5) = 2
        round_nearest(1.49) = 1
    """
    if x == 0:
        return 0
    elif x >= infinity:
        return infinity
    elif x <= -infinity:
        return -infinity
    else:
        raw_nearest = math.floor(x + 0.5)
        return int(raw_nearest)
