cmake_policy(SET CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

cmake_minimum_required(VERSION 3.10)
project(zimage-ncnn-vulkan)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

add_executable(zimage-ncnn-vulkan main.cpp zimage.cpp bpe_tokenizer.cpp)

include(${CMAKE_CURRENT_SOURCE_DIR}/deps_ncnn.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/deps_codec.cmake)

set(ZIMAGE_LINK_LIBRARIES ncnn ${DEPS_CODEC_LIBRARIES})

target_link_libraries(zimage-ncnn-vulkan ${ZIMAGE_LINK_LIBRARIES})

include(GNUInstallDirs)
install(TARGETS zimage-ncnn-vulkan RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
