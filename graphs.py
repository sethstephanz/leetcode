from functools import lru_cache
from typing import List, Tuple
from collections import defaultdict, Counter, deque
from heapq import heappush, heappop
import itertools

"""
B-F: Use with negative or positive edge wweights, can limit stops.
Adjikstra: Handles only positive edges, needs modification for handling limited stops. Fastest.
BFS: Use with unweighted graphs. Naturally handles stops. 
"""


def bellman_ford(n: int, edges: List[Tuple[int, int, int]], src: int,
                 max_edges: int = None) -> List[float]:
    """
    Computes shortest distances from src to all nodes using Bellman-Ford.

    Args:
        n: Number of nodes (0-indexed).
        edges: List of edges in the form (u, v, weight).
        src: Starting node.
        max_edges: Optional. If set, computes shortest paths using at most max_edges edges.

    Returns:
        List of distances from src to each node (inf if unreachable).
    """
    inf = float('inf')

    # Initialize distances
    dist = [inf] * n
    dist[src] = 0

    # Number of iterations:
    # - If max_edges given, relax edges max_edges times
    # - Else, relax n-1 times (standard Bellman-Ford)
    iterations = max_edges if max_edges is not None else n - 1

    for _ in range(iterations):
        temp = dist.copy()  # Avoid using updated values in the same iteration
        for u, v, w in edges:
            if dist[u] + w < temp[v]:
                temp[v] = dist[u] + w
        dist = temp

    return dist


def dijkstra(n: int, edges: List[Tuple[int, int, int]], src: int) -> List[float]:
    """
    Computes shortest distances from src to all nodes using Dijkstra's algorithm.

    Args:
        n: Number of nodes (0-indexed)
        edges: List of edges (u, v, weight)
        src: Starting node

    Returns:
        List of shortest distances from src to each node
    """

    inf = float('inf')

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))

    # Distance array
    dist = [inf] * n
    dist[src] = 0

    # Min-heap: (current_distance, node)
    heap = [(0, src)]

    while heap:
        d, u = heappop(heap)
        # Skip if we already found a better distance
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(heap, (dist[v], v))

    return dist


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
