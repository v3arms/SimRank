default: tests


include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
include ${PETSC_DIR}/lib/petsc/conf/variables


SRC_LIST_TEST 		  = 
SRC_LIST_MAIN 		  = ./src/glaplacian.cpp ./src/embedding.cpp ./main.cpp
SRC_LIST_TEST_PRECOMP = ./test/testmain.cpp 
CNPY_L 			  	  = -L${CNPY_DIR}/lib -lcnpy -lz
CNPY_D 			  	  = -I${CNPY_DIR}/include


tests: cln 
	if [ ! -f test/testmain.o ] ; then \
 	${CLINKER} ${SRC_LIST_TEST_PRECOMP} -c -o ./test/testmain.o ${SLEPC_EPS_LIB} ${SLEPC_INCLUDE} ${PETSC_CC_INCLUDES} ${CNPY_D} ${CNPY_L} ; \
	fi; \
	${CLINKER} ${SRC_LIST_TEST} -o tests ${SLEPC_EPS_LIB} ${SLEPC_INCLUDE} ${PETSC_CC_INCLUDES} ${CNPY_D} ${CNPY_L}

main: 
	${CLINKER} ${SRC_LIST_MAIN} -o embed ${SLEPC_EPS_LIB} ${SLEPC_INCLUDE} ${PETSC_CC_INCLUDES} ${CNPY_D} ${CNPY_L}

cln:
	rm -f tests

clear:
	rm -f test/testmain.o
	rm -f tests
