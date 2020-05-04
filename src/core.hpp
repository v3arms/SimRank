#include <petscksp.h>
#include <petscviewer.h>


class SimrankEstimator {
    public :
        SimrankEstimator();
        ~SimrankEstimator();

        void matLoadPetsc(const char fname[]);
        void solve(PetscScalar tol = 0.00001);
        void exportD() = delete;
        const Mat getWT();
        const Vec getD();

    private : 
        static PetscErrorCode MF_Matvec(Mat, Vec x, Vec y);
        void* MF_Matvec_Preallocate();
        void  MF_Matvec_Free(void* ctx);
        
        Mat         WT;
        Vec         x;
        PetscScalar tol, c;
        PetscInt    num_iter, N, argc, max_sp_rate;
        char        **argv;

};


SimrankEstimator::SimrankEstimator()
: tol(0.00001)
, c(0.4)
, num_iter(10)
, max_sp_rate(1000) {
    MatCreate(PETSC_COMM_WORLD, &WT);
    MatSetType(WT,  MATMPIAIJ);
    VecCreate (PETSC_COMM_WORLD, &x);
    VecSetType(x,   VECMPI);
}


SimrankEstimator::~SimrankEstimator() {
    MatDestroy(&WT);
    VecDestroy(&x);
}


void SimrankEstimator::matLoadPetsc(const char fname[]) {
    PetscViewer vw;
    Mat TMP;
    
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname, FILE_MODE_READ, &vw);
    MatCreate(PETSC_COMM_WORLD, &TMP);
    MatLoad(TMP, vw);
    PetscViewerDestroy(&vw);

    MatTranspose(TMP, MAT_INPLACE_MATRIX, &TMP);
    MatGetSize  (TMP, &N, &N);
    MatDuplicate(TMP, MAT_COPY_VALUES, &WT);

    int i1, i2, ncols;
    const PetscScalar *vals;
    const PetscInt *ids;
    MatInfo info;

    MatGetOwnershipRange(TMP, &i1, &i2);
    for (int i = i1, sum = 0; i < i2; i++, sum = 0) {
        MatGetRow    (TMP, i, &ncols, &ids, &vals);
        PetscScalar *new_vals = new PetscScalar[ncols], sqrtc = sqrt(c);
        for (int j = 0; j < ncols; j++)
            sum += vals[j];
        for (int j = 0; j < ncols; j++)
            new_vals[j] = sqrtc * vals[j] / sum;
        MatSetValues (WT, 1, &i, ncols, ids, new_vals, INSERT_VALUES);
        MatRestoreRow(TMP, i, &ncols, &ids, &vals);
    }
    MatAssemblyBegin(WT, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (WT, MAT_FINAL_ASSEMBLY);
    MatGetInfo(WT, MAT_GLOBAL_SUM, &info);

    PetscPrintf(PETSC_COMM_WORLD, "Total matrix memory : %lf MB\n", info.memory / (1024*1024));

    MatDestroy(&TMP);
}


const Mat SimrankEstimator::getWT() {
    return WT;
}


const Vec SimrankEstimator::getD() {
    return x;
}



PetscScalar SparseVecDot(
    const PetscInt    *ids1, 
    const PetscScalar *vals1, 
    const PetscInt     ncols1,
    const PetscInt    *ids2, 
    const PetscScalar *vals2,
    const PetscInt     ncols2
) {
    PetscScalar s = 0;
    for (int i1 = 0, i2 = 0; i1 < ncols1 && i2 < ncols2;) {
        if (ids1[i1] == ids2[i2]) {
            s += vals1[i1] * vals2[i2];
            i1++, i2++;
        } else
            ids1[i1] < ids2[i2] ? i1++ : i2++;
    }
    return s;
}


void* SimrankEstimator::MF_Matvec_Preallocate() {
    PetscInt N, i1, i2, nproc;
    MatGetSize(WT, &N, &N);
    MatGetOwnershipRange(WT, &i1, &i2);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    auto *f_all  = new PetscScalar[N],
         *vals_f = new PetscScalar[i2 - i1];
    auto *ids_f  = new PetscInt[i2 - i1],
         *recvc  = new PetscInt[nproc]; 
    const PetscInt *recvr;
    MatGetOwnershipRanges(WT, &recvr);
    for (int i = 1; i <= nproc; i++)
        recvc[i - 1] = recvr[i] - recvr[i - 1];
    for (int i = 0; i < i2 - i1; i++)
        ids_f[i] = i1 + i;
    
    Mat** mf_ctx = new Mat*[10];
    mf_ctx[0] = &WT;
    mf_ctx[1] = (Mat*)&num_iter;
    mf_ctx[2] = (Mat*)f_all;
    mf_ctx[3] = (Mat*)vals_f;
    mf_ctx[4] = (Mat*)ids_f;
    mf_ctx[5] = (Mat*)recvc;
    mf_ctx[6] = (Mat*)recvr;
    return mf_ctx;
}


void SimrankEstimator::MF_Matvec_Free(void* ctx) {
    Mat** ct = (Mat**)ctx;
    delete[] ct[2];
    delete[] ct[3];
    delete[] ct[4];
    delete[] ct[5];
    // delete[] ct[6];
    delete[] ct;
}



PetscErrorCode SimrankEstimator::MF_Matvec(Mat MF, Vec x, Vec f) {
    void* to_mffd;
    MatShellGetContext(MF, &to_mffd);
    Mat **m = (Mat**)to_mffd;

    Mat         WT       = *m[0];
    PetscInt    num_iter = *(PetscInt*)m[1];
    PetscScalar *f_all   = (PetscScalar*)m[2],
                *vals_f  = (PetscScalar*)m[3];
    PetscInt    *ids_f   = (PetscInt*)m[4],
                *recvc   = (PetscInt*)m[5],
                *recvr   = (PetscInt*)m[6];


    PetscInt           i1, i2, ncols_wt;
    const PetscScalar *vals_wt;
    const PetscInt    *ids_wt;

    PetscScalar *cur_f_vals;
    MatGetOwnershipRange(WT, &i1, &i2);
    
    VecCopy(x, f);
    for (int iter = 0; iter < num_iter; iter++) {
        VecGetArray(f, &cur_f_vals);
        MPI_Allgatherv(cur_f_vals, i2 - i1, MPI_DOUBLE, f_all, recvc, recvr, MPI_DOUBLE, MPI_COMM_WORLD);
        
        for (int i = i1; i < i2; i++) {
            MatGetRow(WT,  i, &ncols_wt,  &ids_wt,  &vals_wt);
            vals_f[i - i1] = 0;
            for (int j = 0; j < ncols_wt; j++) {
                vals_f[i - i1] += vals_wt[j] * vals_wt[j] * f_all[ids_wt[j]];                
            }
            MatRestoreRow(WT,  i, &ncols_wt,  &ids_wt,  &vals_wt);
        }
        VecRestoreArray(f, &cur_f_vals);
        VecSetValues(f, i2 - i1, ids_f, vals_f, ADD_VALUES);
    }
    VecAssemblyBegin(f);
    VecAssemblyEnd(f);

    return 0;
}


void SimrankEstimator::solve(PetscScalar tol) {
    Mat MF;
    Vec tmp, b;
    KSP solver;

    PetscOptionsInsertString(NULL, "-ksp_monitor -pc_type -ksp_gmres_restart");
    PetscOptionsInsertString(NULL, "-ksp_converged_reason");
    PetscOptionsSetValue(NULL, "-pc_type", "none");
    PetscOptionsSetValue(NULL, "-ksp_guess_type", "pod");
    PetscOptionsSetValue(NULL, "-ksp_gmres_cgs_refinement_type", "refine_ifneeded");
    PetscOptionsSetValue(NULL, "-ksp_gmres_restart", "25");

    VecCreate  (PETSC_COMM_WORLD, &tmp);
    VecCreate  (PETSC_COMM_WORLD, &b);
    VecSetSizes(tmp, PETSC_DECIDE, N);
    VecSetSizes(b,   PETSC_DECIDE, N);
    VecSetSizes(x,   PETSC_DECIDE, N);
    VecSetType (tmp, VECMPI);
    VecSetType (b,   VECMPI);
    VecSet     (b,   1.0);
    VecSet     (x,   1.0);
    
    void* to_mf = MF_Matvec_Preallocate();
    MatCreateShell       (PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, to_mf, &MF);
    MatShellSetOperation (MF, MATOP_MULT, (void(*)(void))MF_Matvec);

    KSPCreate        (PETSC_COMM_WORLD, &solver);
    KSPSetOperators  (solver, MF, MF);
    KSPSetType       (solver, KSPLGMRES);
    KSPSetTolerances(solver, 0, 0.00001, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetFromOptions(solver);
    KSPSetInitialGuessNonzero(solver, PETSC_TRUE);

    KSPSolve(solver, b, x);
    
    MF_Matvec_Free(to_mf);
    MatDestroy(&MF);
    VecDestroy(&tmp);
    VecDestroy(&b);
    KSPDestroy(&solver);
}
