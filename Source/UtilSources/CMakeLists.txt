SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#find_package(Qt5 COMPONENTS Core Widgets Gui Xml REQUIRED)
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
set(CMAKE_AUTOMOC ON)
SET(QT_USE_QTXML TRUE)

aux_source_directory(./ UtilSrc)

find_package(CUDA REQUIRED)

include_directories(/usr/local/cuda/include)

add_library(UtilLib ${UtilSrc})


target_link_libraries(UtilLib ${QT_LIBRARIES} Qt4::QtXml)

#target_link_libraries(UtilLib Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml)
#set_target_properties(UtilLib PROPERTIES AUTOMOC ON)
#set_target_properties(UtilLib PROPERTIES AUTOUIC ON)