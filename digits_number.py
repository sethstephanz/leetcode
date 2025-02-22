from functools import lru_cache
from typing import List
from collections import defaultdict, Counter, deque
import heapq
import itertools


def sum_digits(my_int):
    """
    Take integer and return sum of its individual digits
    """
    return sum(map(int, str(my_int)))
