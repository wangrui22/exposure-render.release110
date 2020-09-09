SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PROJECT_SOURCE_DIR}/UtilSources)
include_directories(${PROJECT_SOURCE_DIR}/PresetsSources)
include_directories(${PROJECT_SOURCE_DIR}/CudaSources)
include_directories(${PROJECT_SOURCE_DIR}/CameraSources)
include_directories(${PROJECT_SOURCE_DIR}/LightingSources)

include_directories(/usr/local/cuda/include)

find_package(Qt5 COMPONENTS Core Widgets Gui Xml REQUIRED)
find_package(VTK REQUIRED)

include( ${VTK_USE_FILE} )

aux_source_directory(./ AppearanceSrc)

add_library(AppearanceLib ${AppearanceSrc})

target_link_libraries(AppearanceLib UtilLib PresetsLib LightingLib CudaLib CameraLib Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml VTKRendering VTKGraphics VTKIO VTKCommon VTKViews)

set_target_properties(AppearanceLib PROPERTIES AUTOMOC ON)
set_target_properties(AppearanceLib PROPERTIES AUTOUIC ON)