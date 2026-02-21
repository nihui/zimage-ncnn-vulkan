cmake_policy(SET CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

cmake_minimum_required(VERSION 3.10)
project(zimage-ncnn-vulkan)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

macro(zimage_add_shader SHADER_SRC)
    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_COMP_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.comp.hex.h)

    add_custom_command(
        OUTPUT ${SHADER_COMP_HEADER}
        COMMAND ${CMAKE_COMMAND} -DSHADER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC} -DSHADER_COMP_HEADER=${SHADER_COMP_HEADER} -P "${CMAKE_CURRENT_SOURCE_DIR}/generate_shader_comp_header.cmake"
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC}
        COMMENT "Preprocessing shader source ${SHADER_SRC_NAME_WE}.comp"
        VERBATIM
    )
    set_source_files_properties(${SHADER_COMP_HEADER} PROPERTIES GENERATED TRUE)

    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_COMP_HEADER})
endmacro()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

zimage_add_shader(zimage_apply_cfg.comp)
zimage_add_shader(zimage_euler_step.comp)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

add_executable(zimage-ncnn-vulkan main.cpp zimage.cpp bpe_tokenizer.cpp)

add_dependencies(zimage-ncnn-vulkan generate-spirv)

include(${CMAKE_CURRENT_SOURCE_DIR}/deps_ncnn.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/deps_codec.cmake)

set(ZIMAGE_LINK_LIBRARIES ncnn ${DEPS_CODEC_LIBRARIES})

target_link_libraries(zimage-ncnn-vulkan ${ZIMAGE_LINK_LIBRARIES})

include(GNUInstallDirs)
install(TARGETS zimage-ncnn-vulkan RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
