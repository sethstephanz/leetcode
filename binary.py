def convert_digits_to_binary(array):
    """
    Given 0s and 1s in an array, convert to number
    """
    res = 0
    for digit in array:
        res = 2 * res + digit

    # return bin(res)     # for binary representation
    return res          # for digit representation
