from typing import List
from collections import defaultdict
from collections import Counter
from collections import deque
import heapq
import itertools


def bfs_traversal(self, edges, source, destination):
    # bfs, iterative
    # given start, end, and list of edges of bidirectional graph,find if graph traversal is possible
    # 1. Convert graph to adjacency list
    ad_list = defaultdict(list)

    # assuming bidirectional graph. map both nodes as neighbors of each other
    for u, v in edges:
        ad_list[u].append(v)
        ad_list[v].append(u)

    # 2. Use set to track visited nodes
    # Use queue to track nodes to return to during bfs (standard bfs)
    q = deque([source])
    visited = set([source])

    # 3. Perform bfs to traverse graph
    while q:
        node = q.popleft()
        if node == destination:
            return True
        for neighbor in ad_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)

    return False


def dfs_traversal(self, edges, source, destination):
    # dfs, recursive
    # given start, end, and list of edges of bidirectional graph,find if graph traversal is possible
    # 1. Convert graph to adjacency list
    ad_list = defaultdict(list)

    # assuming bidirectional graph. map both nodes as neighbors of each other
    for u, v in edges:
        ad_list[u].append(v)
        ad_list[v].append(u)

    # 2. Use set to track visited nodes
    visited = set()

    # 3. Perform dfs to traverse graph
    def dfs(node):
        # have reached dst successfully. return true
        if node == destination:
            return True

        # add node to visited
        visited.add(node)

        # traverse neighbors of node
        for neighbor in ad_list[node]:
            if neighbor not in visited and dfs(neighbor):
                return True

        return False

    # 4. Call dfs on starting node and return result
    return dfs(source)
