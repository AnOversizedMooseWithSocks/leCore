cmake_minimum_required(VERSION 3.15)

project(holoc VERSION 0.1.0 LANGUAGES C)

option(HOLO_USE_ACCELERATE "Use Apple's Accelerate framework" OFF)

set(HOLOC_SOURCES
    src/holo_core.c
    src/holo_trace.c
    src/holo_program.c
)

find_package(Threads REQUIRED)

function(holo_configure_target target)
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(${target} PUBLIC c_std_99)
    target_link_libraries(${target} PUBLIC Threads::Threads)

    if(HOLO_USE_ACCELERATE)
        if(NOT APPLE)
            message(FATAL_ERROR "HOLO_USE_ACCELERATE is only available on macOS")
        endif()
        target_compile_definitions(${target} PRIVATE HOLO_USE_ACCELERATE=1)
        target_link_libraries(${target} PUBLIC "-framework Accelerate")
    else()
        target_compile_definitions(${target} PRIVATE HOLO_USE_ACCELERATE=0)
        if(NOT MSVC)
            target_link_libraries(${target} PUBLIC m)
        endif()
    endif()

    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(holoc_static STATIC ${HOLOC_SOURCES})
if(WIN32)
    # Avoid colliding with the shared library's holoc.lib import library.
    set_target_properties(holoc_static PROPERTIES OUTPUT_NAME holoc_static)
else()
    set_target_properties(holoc_static PROPERTIES OUTPUT_NAME holoc)
endif()
holo_configure_target(holoc_static)

add_library(holoc SHARED ${HOLOC_SOURCES})
set_target_properties(holoc PROPERTIES
    OUTPUT_NAME holoc
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
holo_configure_target(holoc)

enable_testing()

foreach(test_name core trace program)
    add_executable(test_${test_name} tests/test_${test_name}.c)
    target_link_libraries(test_${test_name} PRIVATE holoc_static)
    add_test(NAME ${test_name} COMMAND test_${test_name})
    set_tests_properties(${test_name} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endforeach()
