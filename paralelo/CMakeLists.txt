CMAKE_MINIMUM_REQUIRED (VERSION 3.2)
project(paralelo)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
	myruntime.h
	stemmer_paralelo.cpp
	kernel_stemmer.cu)

FIND_PACKAGE (CUDA QUIET REQUIRED)
SET (CUDA_VERBOSE_BUILD ON)
SET (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" " -arch=sm_30 --compiler-bindir /usr/bin/g++-5")

CUDA_ADD_EXECUTABLE (paralelo ${SOURCE_FILES})
TARGET_LINK_LIBRARIES (paralelo ${CUDA_LIBRARIES})






