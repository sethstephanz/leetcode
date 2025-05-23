class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

from collections import deque

def build_tree(vals):
    """Builds a binary tree from a list of values (level order)."""
    if not vals:
        return None
    root = TreeNode(vals[0])
    queue = deque([root])
    i = 1
    while queue and i < len(vals):
        node = queue.popleft()
        if vals[i] is not None:
            node.left = TreeNode(vals[i])
            queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])
            queue.append(node.right)
        i += 1
    return root


def print_tree_bfs(root):
    """Print tree level-by-level (BFS)."""
    if not root:
        return
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(str(node.val) if node else "None")
            if node:
                queue.append(node.left)
                queue.append(node.right)
        print(" ".join(level)
