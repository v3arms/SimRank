#define CATCH_CONFIG_RUNNER

#include "catch.hpp"
#include "../src/core.hpp"


TEST_CASE("MatLoad") {
    auto sr = new SimrankEstimator();
    sr->matLoadPetsc("../data/amazon0505.petsc");
    MatView(sr->getW(), PETSC_VIEWER_STDOUT_WORLD);
    delete sr;
}


TEST_CASE("solvesmp") {
    auto sr = new SimrankEstimator();
    sr->matLoadPetsc("../data/smp.petsc");
    MatView(sr->getW(), PETSC_VIEWER_STDOUT_WORLD);
    sr->solveD();
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
    CHECK(SimrankEstimator::SparseVecDot(idx1, vals1, ncols1, idx2, vals2, ncols2) == 6);
    CHECK(SimrankEstimator::SparseVecDot(idx11, vals11, ncols11, idx21, vals21, ncols21) == 0);
    CHECK(SimrankEstimator::SparseVecDot(idx12, vals12, ncols12, idx22, vals22, ncols22) == 50);
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
    MatDuplicate(sr->getW(), MAT_COPY_VALUES, &buf);
    MatDiagonalScale(buf, NULL, s);

    MatView(sr->getW(), PETSC_VIEWER_STDOUT_WORLD);
    MatView(buf, PETSC_VIEWER_STDOUT_WORLD);
}


TEST_CASE("solvewiki") {
    auto sr = new SimrankEstimator();
    sr->matLoadPetsc("../data/wiki.petsc");
    sr->solveD();


}


TEST_CASE("solveam") {
    auto sr = new SimrankEstimator();
    sr->matLoadPetsc("../data/amazon0505.petsc");
    sr->solveD();
}



int main(int argc, char* argv[] ) {
    // PetscOptionsSetValue(NULL, "-ksp_view", NULL);
    // PetscOptionsSetValue()
    PetscInitialize(&argc, &argv, (char*)0, "");
    int result = Catch::Session().run( argc, argv );
    PetscFinalize();
    return result;
}
