cmake_minimum_required(VERSION 3.21)

foreach(required_var IN ITEMS LECORE_SOURCE_DIR LECORE_TEST_ROOT LECORE_CONSUMER_SOURCE)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "${required_var} is required")
    endif()
endforeach()

set(variant_build_dir "${LECORE_TEST_ROOT}/liblecore-build")
set(stage_dir "${LECORE_TEST_ROOT}/stage")
set(consumer_build_dir "${LECORE_TEST_ROOT}/consumer-build")
file(REMOVE_RECURSE "${LECORE_TEST_ROOT}")
file(MAKE_DIRECTORY "${LECORE_TEST_ROOT}")

if(NOT DEFINED LECORE_VARIANT_FORMAT)
    set(LECORE_VARIANT_FORMAT OFF)
endif()
if(NOT DEFINED LECORE_VARIANT_RADIX2)
    set(LECORE_VARIANT_RADIX2 ON)
endif()

set(generator_args)
if(DEFINED LECORE_TEST_GENERATOR AND NOT LECORE_TEST_GENERATOR STREQUAL "")
    list(APPEND generator_args -G "${LECORE_TEST_GENERATOR}")
endif()
if(DEFINED LECORE_TEST_GENERATOR_PLATFORM AND NOT LECORE_TEST_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND generator_args -A "${LECORE_TEST_GENERATOR_PLATFORM}")
endif()
if(DEFINED LECORE_TEST_GENERATOR_TOOLSET AND NOT LECORE_TEST_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND generator_args -T "${LECORE_TEST_GENERATOR_TOOLSET}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${LECORE_SOURCE_DIR}"
        -B "${variant_build_dir}"
        ${generator_args}
        "-DLECORE_ENABLE_FORMAT=${LECORE_VARIANT_FORMAT}"
        "-DLECORE_ENABLE_RADIX2=${LECORE_VARIANT_RADIX2}"
        -DLECORE_BUILD_SHARED=OFF
        -DLECORE_BUILD_TESTS=OFF
        -DLECORE_BUILD_EXAMPLES=OFF
        -DCMAKE_BUILD_TYPE=Release
    RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Configuring the format-off build failed: ${configure_result}")
endif()

set(config_args --config Release)
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${variant_build_dir}" ${config_args}
    RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Building the format-off variant failed: ${build_result}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${variant_build_dir}" --prefix "${stage_dir}" ${config_args}
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Installing the format-off variant failed: ${install_result}")
endif()
if(NOT LECORE_VARIANT_FORMAT AND EXISTS "${stage_dir}/include/lecore/lecore_format.h")
    message(FATAL_ERROR "A format-off installation unexpectedly contains lecore_format.h")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${LECORE_CONSUMER_SOURCE}"
        -B "${consumer_build_dir}"
        ${generator_args}
        "-DCMAKE_PREFIX_PATH=${stage_dir}"
        -DCMAKE_BUILD_TYPE=Release
    RESULT_VARIABLE consumer_configure_result
)
if(NOT consumer_configure_result EQUAL 0)
    message(FATAL_ERROR "Configuring a format-off consumer failed: ${consumer_configure_result}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_build_dir}" ${config_args}
    RESULT_VARIABLE consumer_build_result
)
if(NOT consumer_build_result EQUAL 0)
    message(FATAL_ERROR "Building a format-off consumer failed: ${consumer_build_result}")
endif()

set(consumer_executable "${consumer_build_dir}/liblecore_consumer${CMAKE_EXECUTABLE_SUFFIX}")
set(multiconfig_executable
    "${consumer_build_dir}/Release/liblecore_consumer${CMAKE_EXECUTABLE_SUFFIX}")
if(EXISTS "${multiconfig_executable}")
    set(consumer_executable "${multiconfig_executable}")
endif()
execute_process(COMMAND "${consumer_executable}" RESULT_VARIABLE run_result)
if(NOT run_result EQUAL 0)
    message(FATAL_ERROR "The format-off consumer failed at runtime: ${run_result}")
endif()
