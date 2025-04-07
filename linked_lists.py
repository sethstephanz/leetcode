from functools import lru_cache
from typing import List
from collections import defaultdict, Counter, deque
import heapq
import itertools

# Setup
############################################################


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_to_ll(self, my_list):
    """
    For testing LL questions locally: Convert list of ints to 
    linked list (Leetcode does something like this behind the scenes).
    """
    dummy = ListNode()
    head = ListNode()
    dummy.next = head

    for n in range(len(my_list)):
        new_node = ListNode(my_list[n])
        head.next = new_node
        head = head.next

    return dummy.next  

# Setup
############################################################

def reverse_ll(self, head):
    """
    Reverse a linked list. NRMR: next, reverse, move, repeat
    """

    if not head:
        return head

    curr = head
    prev = None

    while curr:
        nxt = curr.next     # n: save next node
        curr.next = prev    # r: reverse the link
        prev = curr         # m: move prev to curr
        curr = nxt          # r: move curr to next

    return prev
