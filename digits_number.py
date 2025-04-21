def sum_digits(num):
    """
    Take integer and return sum of its individual digits
    """
    # convert int to char -> map int to each char -> sum ints
    return sum(map(int, str(num)))
