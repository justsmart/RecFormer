import numpy as np
import random

def splitDigitData(N, V, inCP, seed):
    """
    Function to split digit data based on specified criteria.

    Parameters:
        N (int): The total number of items.
        V (int): The number of groups.
        inCP (float): The fraction of items to be excluded initially.
        seed (int): The random seed for reproducibility.

    Returns:
        splitInd (numpy.ndarray): An array indicating the split of data.
    """
    # Initialize the split index matrix
    splitInd = np.ones((N, V), dtype=int)
    
    # Create a list for random permutations of indices
    indCell = [None] * V
    
    # Calculate the number of elements to delete
    delNum = int(np.floor(N * inCP))
    
    # Set the random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate random permutations and initialize the splitInd matrix
    for i in range(V):
        indCell[i] = np.random.permutation(N)
        splitInd[indCell[i][:delNum], i] = 0
    
    # Counter to track the next index to switch to 0 in each group
    counter = np.array([delNum + 1] * V)
    
    # Resolve cases where a row in splitInd is all zeros
    while True:
        zerosInd = np.where(np.sum(splitInd, axis=1) == 0)[0]
        if zerosInd.size == 0:
            break
        else:
            i = random.randint(0, V - 1)
            splitInd[zerosInd[0], i] = 1
            if counter[i] < N:  # Check to avoid index error
                splitInd[indCell[i][counter[i]], i] = 0
                counter[i] += 1
    
    return splitInd
