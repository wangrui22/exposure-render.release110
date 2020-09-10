SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PROJECT_SOURCE_DIR}/UtilSources)
include_directories(${PROJECT_SOURCE_DIR}/PresetsSources)
include_directories(${PROJECT_SOURCE_DIR}/CudaSources)
include_directories(${PROJECT_SOURCE_DIR}/CameraSources)
include_directories(${PROJECT_SOURCE_DIR}/LightingSources)
include_directories(${PROJECT_SOURCE_DIR}/StatisticsSources)

include_directories(/usr/local/cuda/include)

#QT5
find_package(Qt5 COMPONENTS Core Widgets Gui Xml REQUIRED)

find_package(CUDA 10.2 REQUIRED)

# set(CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50;${CUDA_NVCC_FLAGS}")
# set(CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52;${CUDA_NVCC_FLAGS}")
# set(CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60;${CUDA_NVCC_FLAGS}")
# set(CUDA_NVCC_FLAGS "-gencode=arch=compute_61,code=sm_61;${CUDA_NVCC_FLAGS}")
# set(CUDA_NVCC_FLAGS "-gencode=arch=compute_61,code=compute_61;${CUDA_NVCC_FLAGS}")

#QT4
#find_package(Qt4 REQUIRED)
#include(${QT_USE_FILE})
#set(CMAKE_AUTOMOC ON)
#SET(QT_USE_QTXML TRUE)

find_package(VTK REQUIRED)

include( ${VTK_USE_FILE} )

aux_source_directory(./ AppearanceSrc)

add_executable(Appearance ${AppearanceSrc})

#QT5
target_link_libraries(Appearance UtilLib PresetsLib LightingLib CudaLib CameraLib StatisticLib Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml ${VTK_LIBRARIES} ${CUDA_LIBRARIES})
set_target_properties(Appearance PROPERTIES AUTOMOC ON)
set_target_properties(Appearance PROPERTIES AUTOUIC ON)

#QT4
#target_link_libraries(Appearance UtilLib PresetsLib LightingLib CudaLib CameraLib StatisticLib QVTK ${QT_LIBRARIES} ${VTK_LIBRARIES} Qt4::QtXml)