def combination_with_repetition(m, k):
    """
    Given k balls and m boxes, place k balls into boxes allowing empty boxes, return all possible ways of arrangement

    For example: k=2, m=3, returns [[0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]

    args:
        m: Number of boxes, corresponding to the number of training sets
        k: Number of balls, corresponding to the batch_size in model training process

    return:
        All possible ways of arrangement, each element is a list of length m, representing the number of balls in each box
    """

    # Base case 1: When there are no boxes, return empty list if there are also no balls, otherwise return empty result
    if m == 0:
        return [[]] if k == 0 else []
    
    # Base case 2: When there are no balls, return a list containing only one empty list
    if k == 0:
        return [[0]*m]
    
    result = []
    # For the first box, we can put 0 to k balls
    for i in range(k+1):
        # Recursively calculate all possible ways to place k-i balls into m-1 boxes
        for rest in combination_with_repetition(m-1, k-i):
            # Combine the number of balls in current box with the rest results
            result.append([i] + rest)
    return result

def get_P_fix(number_of_training_set, batch_size):
    P_fix = combination_with_repetition(number_of_training_set, batch_size)
    for i, comb in enumerate(P_fix):
        p = [x / batch_size for x in comb]
        P_fix[i] = p
    return P_fix


if __name__ == "__main__":
    print("Testing get_P_fix(2, 3):")
    print(get_P_fix(2, 3))
    print("\nTesting get_P_fix(3, 2):")
    print(get_P_fix(3, 2))
    print("\nTesting get_P_fix(12, 16):")
    res = get_P_fix(12, 16)
    print(len(res))