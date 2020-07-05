import numpy as np
import scipy.sparse as scsp


def SingleSource(W : scsp.csr_matrix, D : np.array, source : int, num_iter : int = 20) :
    U = []
    D = D.reshape((D.shape[1],))
    U.append(np.zeros((D.shape)))
    result = np.zeros((D.shape))

    U[0][source] = 1

    for k in range(1, num_iter) :
        U.append(W @ U[k - 1])
    
    for k in range(0, num_iter) :
        U[k] = np.multiply(D, U[k])
    
    for k in range(num_iter - 1, 0, -1) :
        result += U[k - 1] + W.T @ U[k]
    
    return result
