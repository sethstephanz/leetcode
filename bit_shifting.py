from functools import lru_cache
from typing import List
from collections import defaultdict, Counter, deque
import heapq
import itertools

def convert_digits_to_binary(self, array):
    """
    Given 0s and 1s in an array, convert to number
    """
    res = 0
    for digit in array:
        res = 2 * res + digit

    # return bin(res)     # for binary representation
    return res          # for digit representation
