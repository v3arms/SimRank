#include "src/core.hpp"


int main(int argc, char *argv[]) {
    PetscInitialize(&argc, &argv, (char*)0, "");

    if (argc < 3) {
        PetscPrintf(PETSC_COMM_WORLD, "./singlesource [input_petsc_mat] [output_d]\n");
        exit(0);
    }
    const char *input = argv[1], *output = argv[2];
    auto sr = new SimrankEstimator();

    PetscPrintf(PETSC_COMM_WORLD, input);

    sr->matLoadPetsc(input);
    sr->solve();
    sr->exportD(output);

    delete sr;
    PetscFinalize();

    return 0;
}
