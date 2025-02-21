from collections import deque
from functools import lru_cache
from collections import defaultdict

##################################################
# mon: sliding window
##################################################


def max_sum_subarray(arr, k):
    # fixed size sliding window
    max_sum, window_sum = 0, sum(arr[:k])  # Initialize sum of first k elements
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]  # Slide the window
        max_sum = max(max_sum, window_sum)
    return max_sum


def longest_substr_k_distinct(s, k):
    # variable size sliding window (at most k distinct chars)
    char_count = {}
    left, max_length = 0, 0
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1  # Shrink window
        max_length = max(max_length, right - left + 1)
    return max_length


##################################################
# tues: linked list (reversals and cycles)
##################################################


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head):
    # reverse linked list, iterative
    prev, curr = None, head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev

# reverse linked list, recursive


def detect_cycle(head):
    # detect cycle in linked list (Floyd's algo)
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def merge_two_lists(l1, l2):
    # merge two sorted lists
    dummy = ListNode()
    curr = dummy

    while l1 and l2:
        if l1.val < l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    curr.next = l1 if l1 else l2
    return dummy.next


##################################################
# wed: bin search
##################################################

def bin_search(nums, target):
    # binary search, iterative
    l, r = 0, len(nums) - 1

    while l <= r:
        m = l + (r - l) // 2
        if nums[m] == target:
            return m
        elif nums[m] > target:
            r = m - 1
        else:
            l = m + 1

    return -1  # not found


def binary_search_recursive(arr, target, left, right):
    # binary search, recursive
    if left > right:
        return -1
    mid = left + (right - left) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


def find_first_last(arr, target):
    # first and last position of target
    def search_left():
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def search_right():
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right

    left, right = search_left(), search_right()
    return [left, right] if left <= right else [-1, -1]


def search_rotated(arr, target):
    # search in rotated sorted array
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid

        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:  # Target is in the left half
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:  # Target is in the right half
                left = mid + 1
            else:
                right = mid - 1

    return -1  # Target not found

##################################################
# thurs: dynamic programming
##################################################


@lru_cache(None)
def fibonacci(n):
   # fibonacci, recursive
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


def fibonacci_bottom_up(n):
    # fibonacci, dp (memoization + bottom-up)
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


def climb_stairs(n):
    # climbing stairs (1d dp)
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

# 0/1 knapsack


##################################################
# fri: graphs and trees (bfs/dfs)
##################################################

def bfs(graph, start):
    # bfs graph traversal
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            queue.extend(graph[node])


def dfs(graph, start, visited=set()):
    # dfs graph traversal
    if start not in visited:
        print(start, end=" ")
        visited.add(start)
        for neighbor in graph[start]:
            dfs(graph, neighbor, visited)


def num_islands(grid):
    # find number of islands (dfs)
    if not grid:
        return 0

    rows, cols, count = len(grid), len(grid[0]), 0

    def dfs(r, c):
        # bounds check
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return

        grid[r][c] = '0'

        # run dfs on each direction
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count

# djikstra's algo (shortest path)


def create_ad_list(binary_graph):
    # create adjacency list from list of edges
    # map each node to all of its neighbors
    graph = defaultdict(list)
    for u, v in binary_graph:
        graph[u].append(v)
        graph[v].append(u)

    return graph


def validPath(edges, start, end):
    ad_list = create_ad_list(edges)
    visited = set()

    def dfs(node):
        if node == end:  # found path
            return True
        visited.add(node)

        # traverse neighbors
        for neighbor in ad_list[node]:
            if neighbor not in visited and dfs(neighbor):
                return True
            return False  # path not found

        return dfs(start)

##################################################
# sat: sort, two-pointer
##################################################


def quick_sort(arr):
    # quick sort
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr):
    # merge sort
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def move_zeros(nums):
    # move zeros to end
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1


##################################################
# sun: recursion and backtracking
##################################################

def subsets(nums):
    # generate all subsets (O(2^n))
    result = []

    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result


def permute(nums):
    # generate all permutations
    result = []

    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return result


def solve_n_queens(n):
    # n-queens (backtracking)
    result = []
    board = [["."] * n for _ in range(n)]

    def is_safe(row, col):
        for i in range(row):
            if board[i][col] == "Q":
                return False
            if col - (row - i) >= 0 and board[i][col - (row - i)] == "Q":
                return False
            if col + (row - i) < n and board[i][col + (row - i)] == "Q":
                return False
        return True

    def backtrack(row):
        if row == n:
            result.append(["".join(r) for r in board])
            return
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = "Q"
                backtrack(row + 1)
                board[row][col] = "."

    backtrack(0)
    return result
