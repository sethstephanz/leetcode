def reverse_linked_list(self, head):
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
