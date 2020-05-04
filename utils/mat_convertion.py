from petsc4py import PETSc
import scipy.sparse as scsp
import numpy as np
from os.path import splitext
import scipy.io
import sys

# "export PYTHONPATH=${PETSC_DIR}/lib/petsc/bin:PYTHONPATH"
# should export to PYTHONPATH as line above in your current shell
from PetscBinaryIO import PetscBinaryIO


def wMatlabToPetsc(fname : str, arg : str) : 
    M = scipy.io.loadmat(fname)[arg].tocsr()

    petsc_mat = PETSc.Mat().createAIJ(
                                size=M.shape, 
                                csr = (M.indptr, M.indices, M.data)
                            )
    vw = PETSc.Viewer().createBinary(splitext(fname)[0] + ".petsc", "w")
    vw(petsc_mat)


def wPetscToMatlab(fname : str, arg : str) : 
    M = PetscBinaryIO().readBinaryFile(fname, mattype="scipy.sparse")[0]
    scipy.io.savemat(splitext(fname)[0] + ".mat", {"W" : M})


if __name__ == "__main__" :
    if len(sys.argv) < 3 :
        print(".mat/.petsc file and/or variable is not specified. Abort.")
        sys.exit(0)

    fname = sys.argv[1]
    arg   = sys.argv[2]

    _, ext = splitext(fname)
    if ext == ".mat" :
        wMatlabToPetsc(fname, arg)
    else:
        if ext == ".petsc" :
            wPetscToMatlab(fname, arg)
        else :
            print(".mat and .petsc mtx formats supported only.")


