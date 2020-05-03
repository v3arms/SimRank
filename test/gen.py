import numpy as np
import scipy.sparse as scsp
from scipy.sparse.linalg import gmres, LinearOperator


W = np.array([[0, 0, 0.5, 0], 
             [0, 0.5, 0, 1.0/3], 
             [0.5, 0.5, 0.5, 1.0/3], 
             [0.5, 0, 0, 1.0/3]])
print(W)


W = scsp.csr_matrix(W)

print(W)


def matvec3(x) :
    XS = scsp.csr_matrix(scsp.diags(x, 0))
    XS += W.T @ XS @ W
    return XS.diagonal()


def matvec(x) :
    XS = scsp.csr_matrix(scsp.diags(x, 0))
    for _ in range(10):
        XS += scsp.diags(np.array(((W.T @ XS).multiply(W.T)).sum(-1)).squeeze())
    return XS.diagonal()

def matvec2(x) :
    return W.dot(x)

def matvec4(x) :
    XS = scsp.csr_matrix(scsp.diags(x, 0))
    return (W.T @ XS @ W.T).diagonal()

LA = scsp.linalg.LinearOperator((4, 4), matvec=matvec, dtype=np.float64)
b = [1, 1, 1, 1]

x, _ = gmres(LA, b)

print(x)
print(LA(x))
