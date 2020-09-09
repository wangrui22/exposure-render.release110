cmake_minimum_required(VERSION 3.10.0)

#project(ExposureRender)
project(ExposureRender LANGUAGES CXX CUDA)

add_subdirectory(./UtilSources)

add_subdirectory(./CudaSources)

add_subdirectory(./StatisticsSources)

add_subdirectory(./PresetsSources)

add_subdirectory(./LightingSources)

add_subdirectory(./CameraSources)

add_subdirectory(./AppearanceSources)