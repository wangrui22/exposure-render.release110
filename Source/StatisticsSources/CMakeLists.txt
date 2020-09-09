SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PROJECT_SOURCE_DIR}/UtilSources)
include_directories(/usr/local/cuda/include)

#QT5
find_package(Qt5 COMPONENTS Core Widgets Gui Xml REQUIRED)

#QT4
# find_package(Qt4 REQUIRED)
# include(${QT_USE_FILE})
# set(CMAKE_AUTOMOC ON)
# SET(QT_USE_QTXML TRUE)

aux_source_directory(./ StatisticSrc)

add_library(StatisticLib ${StatisticSrc})

#QT5
target_link_libraries(StatisticLib UtilLib Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml)
set_target_properties(StatisticLib PROPERTIES AUTOMOC ON)
set_target_properties(StatisticLib PROPERTIES AUTOUIC ON)

#QT4
#target_link_libraries(StatisticLib UtilLib ${QT_LIBRARIES} Qt4::QtXml)