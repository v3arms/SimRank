#include <petscksp.h>
#include <petscviewer.h>
#include <fstream>


class SimrankEstimator {
    public :
        SimrankEstimator();
        ~SimrankEstimator();

        matLoadPetsc(const char fname[]);
        void solveD(PetscScalar tol = 0.00001);
        void exportD(const char* fname);
        const Mat getWT();
        const Vec getD();
        Vec SingleSource(int nodeid);

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
