SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PROJECT_SOURCE_DIR}/UtilSources)
include_directories(${PROJECT_SOURCE_DIR}/PresetsSources)
include_directories(${PROJECT_SOURCE_DIR}/CudaSources)

include_directories(/usr/local/cuda/include)

#find_package(Qt5 COMPONENTS Core Widgets Gui Xml REQUIRED)

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
set(CMAKE_AUTOMOC ON)
SET(QT_USE_QTXML TRUE)

aux_source_directory(./ CameraSrc)

add_library(CameraLib ${CameraSrc})


target_link_libraries(CameraLib UtilLib PresetsLib CudaLib ${QT_LIBRARIES} Qt4::QtXml)

#target_link_libraries(CameraLib UtilLib PresetsLib CudaLib Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml)
#set_target_properties(CameraLib PROPERTIES AUTOMOC ON)
#set_target_properties(CameraLib PROPERTIES AUTOUIC ON)