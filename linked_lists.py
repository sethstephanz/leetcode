def reverse_linked_list(self, head):
    """
    Reverse a linked list. NRMR: next, reverse, move, repeat
    Save node, reverse current link, move prev to current node, move curr to next node.
    Return prev at end, not head.
    """

    if not head:
        return head

    curr = head
    prev = None

    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt

    return prev
