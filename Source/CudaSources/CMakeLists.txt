aux_source_directory(./ CUDA_SRC)
set(CUDA_K_SRC 
    Blur.cuh
    Buffer.cuh
    Camera.cuh
    Core.cuh
    Core.cu
    Denoise.cuh
    Estimate.cuh
    Lighting.cuh
    Model.cuh
    MonteCarlo.cuh
    MultipleScattering.cuh
    NearestIntersection.cuh
    RayMarching.cuh
    RNG.cuh
    Sample.cuh
    Shader.cuh
    SingleScattering.cuh
    SpecularBloom.cuh
    ToneMap.cuh
    Transport.cuh
    Utilities.cuh
    Variance.cuh
    View.cuh
    Woodcock.cuh
    cutil_math.h
)

set(CUDA_C_SRC CudaUtilities.h CudaUtilities.cpp Scene.h Scene.cpp)

#gdb支持
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#QT5
#find_package(Qt5 COMPONENTS Core REQUIRED)

find_package(CUDA REQUIRED)

#QT4
# find_package(Qt4 REQUIRED)
# include(${QT_USE_FILE})
# set(CMAKE_AUTOMOC ON)
# SET(QT_USE_QTXML TRUE)

set(CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=compute_60;${CUDA_NVCC_FLAGS}")

include_directories(/usr/local/cuda/include)
include_directories(${PROJECT_SOURCE_DIR}/UtilSources)

#CUDA支持3：.cu .cuh kernel文件的编译，并生成so，注意是 cuda_add_library 而不是 add_library
cuda_add_library(CudaLibKernel ${CUDA_K_SRC})
add_library(CudaLib ${CUDA_C_SRC})
target_link_libraries(CudaLib CudaLibKernel UtilLib ${CUDA_LIBRARIES})

# add_library(CudaLib ${CUDA_SRC})
# target_link_libraries(CudaLib UtilLib ${CUDA_LIBRARIES})

#QT5
#target_link_libraries(CudaLib Qt5::Core)
#set_target_properties(UtilLib PROPERTIES AUTOMOC ON)
#set_target_properties(UtilLib PROPERTIES AUTOUIC ON)

#QT4
#target_link_libraries(CudaLib ${QT_LIBRARIES} Qt4::QtXml)

