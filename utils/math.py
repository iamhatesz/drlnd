import math
from typing import Callable


def exp_decay(start: float, end: float, decay: float) -> Callable[[int], float]:
    return lambda x: end + (start - end) * math.exp(-1. * x / decay)


def linear_decay(start: float, end: float, decay: float) -> Callable[[int], float]:
    return lambda x: max(end, min(start + (end - start) / decay * x, start))
