"""
0/1 knapsack problems: Look for pick/leave type subproblems. Construct a grid.
    For each cell in the grid, pick the optimal choices for that cell, deciding
    between above cell and new choice.
    cell[r][c] = max(previous max (grid[r-1][j]), 
                        value of current item + value of remaining space
                                                    (cell[r-1][j - item's weight])
                    )

"""


def make_grid(m, n):
    """
    Make blank m x n grid 
    """
    return [[0 for _ in range(n)] for _ in range(m)]
