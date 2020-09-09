SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PROJECT_SOURCE_DIR}/UtilSources)
include_directories(${PROJECT_SOURCE_DIR}/PresetsSources)
include_directories(${PROJECT_SOURCE_DIR}/CudaSources)
include_directories(/usr/local/cuda/include)

#QT5
find_package(Qt5 COMPONENTS Core Widgets Gui Xml REQUIRED)

#QT4
# find_package(Qt4 REQUIRED)
# include(${QT_USE_FILE})
# set(CMAKE_AUTOMOC ON)
# SET(QT_USE_QTXML TRUE)

aux_source_directory(./ LightingSrc)

add_library(LightingLib ${LightingSrc})

#QT5
target_link_libraries(LightingLib UtilLib PresetsLib Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml)
set_target_properties(LightingLib PROPERTIES AUTOMOC ON)
set_target_properties(LightingLib PROPERTIES AUTOUIC ON)

#QT4
#target_link_libraries(LightingLib UtilLib PresetsLib ${QT_LIBRARIES} Qt4::QtXml)