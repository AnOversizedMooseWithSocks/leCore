cmake_minimum_required(VERSION 3.21)

foreach(required_var IN ITEMS LECORE_SOURCE_DIR LECORE_TEST_ROOT)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "${required_var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${LECORE_TEST_ROOT}")
file(MAKE_DIRECTORY "${LECORE_TEST_ROOT}")

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
        -B "${LECORE_TEST_ROOT}"
        ${generator_args}
        -DLECORE_BUILD_TESTS=OFF
        -DLECORE_BUILD_EXAMPLES=ON
        -DCMAKE_BUILD_TYPE=Release
    RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Configuring the examples-only build failed: ${configure_result}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${LECORE_TEST_ROOT}" --config Release
    RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Building the examples-only variant failed: ${build_result}")
endif()

foreach(example_name IN ITEMS liblecore_hrr_example liblecore_hrr_cpp_example)
    set(example_path
        "${LECORE_TEST_ROOT}/examples/${example_name}${CMAKE_EXECUTABLE_SUFFIX}")
    set(multiconfig_path
        "${LECORE_TEST_ROOT}/examples/Release/${example_name}${CMAKE_EXECUTABLE_SUFFIX}")
    if(EXISTS "${multiconfig_path}")
        set(example_path "${multiconfig_path}")
    endif()
    execute_process(COMMAND "${example_path}" RESULT_VARIABLE run_result)
    if(NOT run_result EQUAL 0)
        message(FATAL_ERROR "${example_name} failed at runtime: ${run_result}")
    endif()
endforeach()
