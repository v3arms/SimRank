from petsc4py import PETSc
import scipy.sparse as scsp
import numpy as np
from os.path import splitext
import scipy.io
import sys

try :
    from PetscBinaryIO import PetscBinaryIO
except ModuleNotFoundError :
    print("You should do this in your shell:\nexport PYTHONPATH=${PETSC_DIR}/lib/petsc/bin:PYTHONPATH")
    sys.exit(0)


def matMatlabToPetsc(fname : str, arg : str) : 
    M = scipy.io.loadmat(fname)[arg].tocsr()

    petsc_mat = PETSc.Mat().createAIJ(
                                size=M.shape, 
                                csr = (M.indptr, M.indices, M.data)
                            )
    vw = PETSc.Viewer().createBinary(splitext(fname)[0] + ".petsc", "w")
    vw(petsc_mat)


def matPetscToMatlab(fname : str, arg : str) : 
    M = PetscBinaryIO().readBinaryFile(fname, mattype="scipy.sparse")[0]
    scipy.io.savemat(splitext(fname)[0] + ".mat", {arg : M})


if __name__ == "__main__" :
    if len(sys.argv) < 3 :
        print(".mat/.petsc file and/or variable is not specified. Abort.")
        sys.exit(0)

    fname = sys.argv[1]
    arg   = sys.argv[2]

    _, ext = splitext(fname)
    if ext == ".mat" :
        matMatlabToPetsc(fname, arg)
    elif ext == ".petsc" :
        matPetscToMatlab(fname, arg)
    else :
        print(".mat and .petsc mtx formats supported only.")


