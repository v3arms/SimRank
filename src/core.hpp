#include <petscksp.h>
#include <petscviewer.h>


class SimrankEstimator {
    public :
        SimrankEstimator();
        ~SimrankEstimator();

        void matLoadPetsc(const char fname[]);
        void solveD(PetscScalar tol = 0.00001);
        void exportD() = delete;
        const Mat getW();
        const Vec getD();

        static PetscScalar SparseVecDot(
            const PetscInt    *ids1, 
            const PetscScalar *vals1, 
            const PetscInt     ncols1,
            const PetscInt    *ids2, 
            const PetscScalar *vals2,
            const PetscInt     ncols2
        );

    private : 
        static PetscErrorCode MFFD_Matvec(Mat, Vec xx, Vec y);
        
        Mat         W, WT, BUF, MFFD;
        Vec         x;
        PetscScalar tol, c;
        PetscInt    num_iter, N, argc;
        char        **argv;

};


// #include "core.hpp"


SimrankEstimator::SimrankEstimator()
: tol(0.00001)
, c(0.8)
, num_iter(10) {
    MatCreate(PETSC_COMM_WORLD, &W);
    MatCreate(PETSC_COMM_WORLD, &WT);
    MatCreate(PETSC_COMM_WORLD, &BUF);

    MatSetType(W,   MATMPIAIJ);
    MatSetType(WT,  MATMPIAIJ);
    MatSetType(BUF, MATMPIAIJ);

    VecCreate (PETSC_COMM_WORLD, &x);
    VecSetType(x,   VECMPI);
}


SimrankEstimator::~SimrankEstimator() {
    MatDestroy(&W);
    MatDestroy(&WT);
    MatDestroy(&BUF);
    VecDestroy(&x);
}


void SimrankEstimator::matLoadPetsc(const char fname[]) {
    PetscViewer vw;
    Mat TMP;
    PetscErrorCode ierr;
    
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname, FILE_MODE_READ, &vw);
    MatCreate(PETSC_COMM_WORLD, &TMP);
    MatLoad(TMP, vw);
    PetscViewerDestroy(&vw);

    MatTranspose(TMP, MAT_INPLACE_MATRIX, &TMP);
    MatGetSize  (TMP, &N, &N);
    MatDuplicate(TMP, MAT_COPY_VALUES, &W);
    MatDuplicate(TMP, MAT_COPY_VALUES, &BUF);

    int i1, i2, ncols;
    const PetscScalar *vals;
    const PetscInt *ids;

    MatGetOwnershipRange(TMP, &i1, &i2);
    for (int i = i1, sum = 0; i < i2; i++, sum = 0) {
        MatGetRow    (TMP, i, &ncols, &ids, &vals);
        PetscScalar *new_vals = new PetscScalar[ncols], sqrtc = sqrt(c);
        for (int j = 0; j < ncols; j++)
            sum += vals[j];
        for (int j = 0; j < ncols; j++)
            new_vals[j] = sqrtc * vals[j] / sum;
        MatSetValues (W, 1, &i, ncols, ids, new_vals, INSERT_VALUES);
        MatRestoreRow(TMP, i, &ncols, &ids, &vals);
    }
    
    
    MatAssemblyBegin(W, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (W, MAT_FINAL_ASSEMBLY);
    MatTranspose    (W, MAT_INPLACE_MATRIX, &W);
    // MatCreateTranspose(W, &WT);
    MatTranspose    (W, MAT_INITIAL_MATRIX, &WT);
    MatDestroy      (&TMP);
}


const Mat SimrankEstimator::getW() {
    return W;
}


/*
PetscErrorCode SimrankEstimator::MFFD_Matvec(Mat MFFD, Vec x, Vec f) {
    void* to_mffd;
    MatShellGetContext(MFFD, &to_mffd);
    Mat     **m = (Mat**)to_mffd;
    Mat M = *m[1];
    Mat C;
    MatCreate(PETSC_COMM_WORLD, &C);
    

    MatDuplicate(M, MAT_COPY_VALUES, &C);

    MatDiagonalScale(C, NULL, x);
    MatMatMult(C, M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
    MatGetDiagonal(C, f);
    VecAXPY(f, 1, x);

    // auto ierr = MatMult(M, x, f);
    return 0;
}
*/



PetscScalar SimrankEstimator::SparseVecDot(
    const PetscInt    *ids1, 
    const PetscScalar *vals1, 
    const PetscInt     ncols1,
    const PetscInt    *ids2, 
    const PetscScalar *vals2,
    const PetscInt     ncols2
) {
    PetscScalar s = 0;
    for (int i1 = 0, i2 = 0; i1 < ncols1 && i2 < ncols2;) {
        // PetscPrintf(PETSC_COMM_WORLD, "%d %d\n", i1, i2);
        if (ids1[i1] == ids2[i2]) {
            s += vals1[i1] * vals2[i2];
            i1++, i2++;
        } else
            ids1[i1] < ids2[i2] ? i1++ : i2++;
    }
    return s;
}


/*
PetscErrorCode SimrankEstimator::MFFD_Matvec(Mat MFFD, Vec xx, Vec y) {
    void* to_mffd;
    MatShellGetContext(MFFD, &to_mffd);
    Mat     **m = (Mat**)to_mffd;
    Mat      W = *m[0], WT = *m[1], BUF = *m[2];
    Mat C;
    PetscInt N;
    PetscInt num_iter = *(int*)m[3];

    // MatDuplicate(W, MAT_COPY_VALUES, &C);
    // MatZeroEntries(C);
    // MatDiagonalSet(C, xx, INSERT_VALUES);
    
    MatGetSize(BUF, &N, &N);
    MatCopy(WT, BUF, SAME_NONZERO_PATTERN);
    MatDiagonalScale(BUF, NULL, xx);

    // MatPtAP(C, W, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
    // MatView(C, PETSC_VIEWER_STDOUT_WORLD);
    MatMatMult(BUF, W, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
    MatGetDiagonal(C, y);
    // VecView(y, PETSC_VIEWER_STDOUT_WORLD);
    MatDestroy(&C);
    return 0;
}
*/



PetscErrorCode SimrankEstimator::MFFD_Matvec(Mat MFFD, Vec x, Vec f) {
    void* to_mffd;
    MatShellGetContext(MFFD, &to_mffd);
    Mat     **m = (Mat**)to_mffd;
    Mat      W = *m[0], WT = *m[1], BUF = *m[2];
    PetscInt N;
    PetscInt num_iter = *(int*)m[3];

    PetscInt           i1, i2, ncols_buf, ncols_wt;
    const PetscScalar *vals_buf, *vals_wt;
    const PetscInt    *ids_buf,  *ids_wt;

    MatGetSize(BUF, &N, &N);
    
    MatGetOwnershipRange(BUF, &i1, &i2);
    PetscScalar *vals_f = new PetscScalar[i2 - i1];
    PetscInt    *ids_f  = new PetscInt   [i2 - i1];
    for (int i = 0; i < i2 - i1; i++)
        ids_f[i] = i1 + i;

    VecCopy(x, f);
    for (int iter = 0; iter < num_iter; iter++) {
        MatCopy(WT, BUF, SAME_NONZERO_PATTERN);
        MatDiagonalScale(BUF, NULL, f);

        for (int i = i1; i < i2; i++) {
            MatGetRow(BUF, i, &ncols_buf, &ids_buf, &vals_buf);
            MatGetRow(WT,  i, &ncols_wt,  &ids_wt,  &vals_wt);

            vals_f[i] = SparseVecDot(
                ids_buf, vals_buf, ncols_buf, 
                ids_wt,  vals_wt,  ncols_wt
            );
            MatRestoreRow(BUF, i, &ncols_buf, &ids_buf, &vals_buf);
            MatRestoreRow(W,  i, &ncols_wt,  &ids_wt,  &vals_wt);
        }
        VecSetValues(f, i2 - i1, ids_f, vals_f, ADD_VALUES);
    }

    VecAssemblyBegin(f);
    VecAssemblyEnd(f);

    delete[] ids_f;
    delete[] vals_f;
    return 0;
}



void SimrankEstimator::solveD(PetscScalar tol) {
    Mat MFFD;
    Vec tmp, b;
    Mat* to_mffd[] = {&W, &WT, &BUF, (Mat*)&num_iter};
    KSP solver;

    // PetscOptionsInsertString(NULL, "-ksp_view -ksp_converged_reason -ksp_monitor -pc_type -ksp_view_mat_explicit");
    PetscOptionsInsertString(NULL, "-ksp_monitor -pc_type -ksp_guess_type");

    PetscOptionsSetValue(NULL, "-pc_type", "none");
    PetscOptionsSetValue(NULL, "-ksp_guess_type", "pod");

    VecCreate  (PETSC_COMM_WORLD, &tmp);
    VecCreate  (PETSC_COMM_WORLD, &b);
    VecSetSizes(tmp, PETSC_DECIDE, N);
    VecSetSizes(b,   PETSC_DECIDE, N);
    VecSetSizes(x,   PETSC_DECIDE, N);
    VecSetType (tmp, VECMPI);
    VecSetType (b,   VECMPI);
    VecSet     (b,   1.0);
    // VecSet     (x,   0.0);
    
    MatCreateShell       (PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, to_mffd, &MFFD);
    MatShellSetOperation (MFFD, MATOP_MULT, (void(*)(void))MFFD_Matvec);

    KSPCreate        (PETSC_COMM_WORLD, &solver);
    KSPSetOperators  (solver, MFFD, MFFD);
    KSPSetType       (solver, KSPGMRES);
    KSPSetTolerances(solver, 0, 0.00001, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetInitialGuessNonzero(solver, PETSC_TRUE);
    KSPSetFromOptions(solver);

    KSPSolve(solver, b, x);
    
    MatDestroy(&MFFD);
    VecDestroy(&tmp);
    VecDestroy(&b);
    KSPDestroy(&solver);
}


const Vec SimrankEstimator::getD() {
    return x;
}