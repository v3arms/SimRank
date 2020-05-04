#define CATCH_CONFIG_RUNNER

#include "catch.hpp"
#include "../src/core.hpp"


TEST_CASE("MatLoad") {
    auto sr = new SimrankEstimator();
    sr->matLoadPetsc("../data/amazon0505.petsc");
    // MatView(sr->getW(), PETSC_VIEWER_STDOUT_WORLD);
    delete sr;
}


TEST_CASE("solvesmp") {
    auto sr = new SimrankEstimator();
    sr->matLoadPetsc("../data/smp.petsc");
    MatView(sr->getWT(), PETSC_VIEWER_STDOUT_WORLD);
    sr->solve();
    VecView(sr->getD(), PETSC_VIEWER_STDOUT_WORLD);
}



TEST_CASE("sparsevecdot") {
    PetscInt    idx1[]  = {2, 4, 5, 6},
                idx2[]  = {5, 7},
                ncols1  = 4,
                ncols2  = 2;
    PetscScalar vals1[] = {1, 2, 3, 4},
                vals2[] = {2, 1};
    PetscInt    idx11[]  = {2},
                idx21[]  = {1, 3, 5, 6},
                ncols11  = 1,
                ncols21  = 4;
    PetscScalar vals11[] = {2},
                vals21[] = {1, 1, 1, 1};
    PetscInt    idx12[]  = {2, 4, 5, 6, 8, 10, 14, 18},
                idx22[]  = {3, 7, 8, 9, 14, 15, 17},
                ncols12  = 8,
                ncols22  = 7;
    PetscScalar vals12[] = {1, 2, 3, 4, 5, 6, 7, 8},
                vals22[] = {1, 2, 3, 4, 5, 6, 7};
    CHECK(SparseVecDot(idx1, vals1, ncols1, idx2, vals2, ncols2) == 6);
    CHECK(SparseVecDot(idx11, vals11, ncols11, idx21, vals21, ncols21) == 0);
    CHECK(SparseVecDot(idx12, vals12, ncols12, idx22, vals22, ncols22) == 50);
}


TEST_CASE("diagscale") {
    auto sr = new SimrankEstimator();
    sr->matLoadPetsc("../data/smp.petsc");

    Vec s;
    VecCreate(PETSC_COMM_WORLD, &s);
    VecSetType(s, VECMPI);
    VecSetSizes(s, PETSC_DECIDE, 4);
    VecSet(s, 2);
    Mat buf;
    MatDuplicate(sr->getWT(), MAT_COPY_VALUES, &buf);
    MatDiagonalScale(buf, NULL, s);

    MatView(sr->getWT(), PETSC_VIEWER_STDOUT_WORLD);
    MatView(buf, PETSC_VIEWER_STDOUT_WORLD);
}


TEST_CASE("solvewiki") {
    auto sr = new SimrankEstimator();
    sr->matLoadPetsc("../data/wiki.petsc");
    sr->solve();


}


TEST_CASE("solveam") {
    auto sr = new SimrankEstimator();
    sr->matLoadPetsc("../data/amazon0505.petsc");
    sr->solve();
}


TEST_CASE("solveexp") {
    auto sr = new SimrankEstimator();
    sr->matLoadPetsc("../data/experts.petsc");
    sr->solve();
}


TEST_CASE("mpi_seq_matmult") {
    Vec x;
    const PetscInt *ranges;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 10, &x);
    VecGetOwnershipRanges(x, &ranges);
    PetscInt size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 0; i < 2*size; i++)
        PetscPrintf(PETSC_COMM_WORLD, "%d ", ranges[i]);
    PetscPrintf(PETSC_COMM_WORLD, "\n");
}


TEST_CASE("allgather") {
    Vec x;
    PetscInt n, *recvcounts, i1, i2, *displs, rank;
    const PetscInt *vec_ranges;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 23, &x);
    VecSet(x, 2.0);
    VecGetOwnershipRanges(x, &vec_ranges);
    VecGetOwnershipRange(x, &i1, &i2);
    PetscScalar *vals, *globalvals = new double[24];

    MPI_Comm_size(MPI_COMM_WORLD, &n);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    recvcounts = new PetscInt[n];
    displs     = new PetscInt[n];
    
    for (int i = 1; i <= n; i++)
        recvcounts[i - 1] = vec_ranges[i] - vec_ranges[i - 1];

    VecGetArray(x, &vals);
    MPI_Allgatherv(vals, i2 - i1, MPI_DOUBLE, globalvals, recvcounts, vec_ranges, MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < 23; i++)
            PetscPrintf(PETSC_COMM_WORLD, "%lf ", globalvals[i]);
        PetscPrintf(PETSC_COMM_WORLD, "\n");
    }
    

    VecRestoreArray(x, &vals);
    delete[] recvcounts;
    delete[] globalvals;
    delete[] displs;
}



int main(int argc, char* argv[] ) {
    // PetscOptionsSetValue(NULL, "-ksp_view", NULL);
    // PetscOptionsSetValue()
    PetscInitialize(&argc, &argv, (char*)0, "");
    int result = Catch::Session().run( argc, argv );
    PetscFinalize();
    return result;
}
