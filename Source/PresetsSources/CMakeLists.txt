SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PROJECT_SOURCE_DIR}/UtilSources)
include_directories(/usr/local/cuda/include)

#find_package(Qt5 COMPONENTS Core Widgets Gui Xml REQUIRED)

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
set(CMAKE_AUTOMOC ON)
SET(QT_USE_QTXML TRUE)

aux_source_directory(./ PresetsSrc)

add_library(PresetsLib ${PresetsSrc})

target_link_libraries(PresetsLib UtilLib ${QT_LIBRARIES} Qt4::QtXml)

#target_link_libraries(PresetsLib UtilLib Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml)
#set_target_properties(PresetsLib PROPERTIES AUTOMOC ON)
#set_target_properties(PresetsLib PROPERTIES AUTOUIC ON)