# simple util function to build a bst for testing

class ListNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def bst_init(root_val, insert_vals):
    for val in insert_vals:
        bst_insert(val)

def insert(root, num):
    if num == root.val:
        print('Error: Duplicate values')
        return
    elif num < root.val:
        if root.left:
            insert(root.left, num)
        else:
            root.left = ListNode(num)
    elif num > root.val:
        if root.right:
            insert(root.right, num)
        else:
            root.right = ListNode(num)

def print_node(node):
    print('----------')
    print("node.val: ", node.val)
    if node.left:
        print("node left: ", node.left.val)
    else:
        print("node left: ", None)
    if node.right:
        print("node right: ", node.right.val)
    else:
        print("node right: ", None)
    print('----------')

# TODO: printing the tree in a more legible graphical representation
# would be good here


def dfs(node):
    if not node:
        return
    dfs(node.right)
    dfs(node.left)
    # print_node(node)

# dfs(my_bst)

