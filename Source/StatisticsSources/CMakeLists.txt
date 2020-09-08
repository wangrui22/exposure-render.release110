SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PROJECT_SOURCE_DIR}/UtilSources)
include_directories(/usr/local/cuda/include)

find_package(Qt5 COMPONENTS Core Widgets Gui Xml REQUIRED)

aux_source_directory(./ StatisticSrc)

add_library(StatisticLib ${StatisticSrc})

target_link_libraries(StatisticLib UtilLib Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml)

set_target_properties(StatisticLib PROPERTIES AUTOMOC ON)
set_target_properties(StatisticLib PROPERTIES AUTOUIC ON)