function(lecore_add_c_test test_name source_file)
    add_executable("${test_name}" "${source_file}")
    target_compile_features("${test_name}" PRIVATE c_std_11)
    target_link_libraries("${test_name}" PRIVATE lecore::lecore)
    if(MSVC)
        target_compile_options("${test_name}" PRIVATE /W4 /permissive-)
    else()
        target_compile_options("${test_name}" PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
    if(WIN32 AND LECORE_BUILD_SHARED)
        add_custom_command(TARGET "${test_name}" POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:lecore>" "$<TARGET_FILE_DIR:${test_name}>"
            VERBATIM
        )
    endif()
    add_test(NAME "${test_name}" COMMAND "${test_name}")
endfunction()

lecore_add_c_test(liblecore_test_api c/test_api.c)
target_compile_definitions(liblecore_test_api PRIVATE
    LECORE_TEST_ISA_VERSION=${LECORE_ISA_VERSION}
    LECORE_TEST_VERSION="${LECORE_VERSION}"
)
lecore_add_c_test(liblecore_test_numeric c/test_numeric.c)
lecore_add_c_test(liblecore_test_edges c/test_edges.c)

if(LECORE_ENABLE_FORMAT)
    lecore_add_c_test(liblecore_test_format c/test_format.c)
endif()

add_executable(liblecore_test_cpp_header c/test_cpp_header.cpp)
target_compile_features(liblecore_test_cpp_header PRIVATE cxx_std_17)
target_link_libraries(liblecore_test_cpp_header PRIVATE lecore::lecore)
if(WIN32 AND LECORE_BUILD_SHARED)
    add_custom_command(TARGET liblecore_test_cpp_header POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:lecore>" "$<TARGET_FILE_DIR:liblecore_test_cpp_header>"
        VERBATIM
    )
endif()
add_test(NAME liblecore_test_cpp_header COMMAND liblecore_test_cpp_header)

if(NOT CMAKE_CROSSCOMPILING AND NOT LECORE_ENABLE_SANITIZERS)
    set(consumer_runner "${CMAKE_CURRENT_SOURCE_DIR}/consumers/run_installed_consumer.cmake")
    add_test(NAME liblecore_consumer_cmake
        COMMAND "${CMAKE_COMMAND}"
            "-DLECORE_BUILD_DIR=${PROJECT_BINARY_DIR}"
            "-DLECORE_CONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/consumers/cmake-consumer"
            "-DLECORE_TEST_ROOT=${PROJECT_BINARY_DIR}/_consumer_tests/cmake"
            "-DLECORE_TEST_CONFIG=$<CONFIG>"
            "-DLECORE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}"
            "-DLECORE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
            "-DLECORE_EXPECT_FORMAT=${LECORE_ENABLE_FORMAT}"
            "-DLECORE_EXPECT_SHARED=${LECORE_BUILD_SHARED}"
            -P "${consumer_runner}"
    )

    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        add_test(NAME liblecore_consumer_pkgconfig
            COMMAND "${CMAKE_COMMAND}"
                "-DLECORE_BUILD_DIR=${PROJECT_BINARY_DIR}"
                "-DLECORE_CONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/consumers/pkgconfig-consumer"
                "-DLECORE_TEST_ROOT=${PROJECT_BINARY_DIR}/_consumer_tests/pkgconfig"
                "-DLECORE_TEST_CONFIG=$<CONFIG>"
                "-DLECORE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}"
                "-DLECORE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
                "-DLECORE_EXPECT_FORMAT=${LECORE_ENABLE_FORMAT}"
                "-DLECORE_EXPECT_SHARED=${LECORE_BUILD_SHARED}"
                -DLECORE_USE_PKG_CONFIG=ON
                -P "${consumer_runner}"
        )
    endif()

    add_test(NAME liblecore_variant_format_off
        COMMAND "${CMAKE_COMMAND}"
            "-DLECORE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "-DLECORE_CONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/consumers/cmake-consumer"
            "-DLECORE_TEST_ROOT=${PROJECT_BINARY_DIR}/_consumer_tests/format-off"
            "-DLECORE_TEST_CONFIG=$<CONFIG>"
            "-DLECORE_TEST_GENERATOR=${CMAKE_GENERATOR}"
            "-DLECORE_TEST_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DLECORE_TEST_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            -DLECORE_VARIANT_FORMAT=OFF
            -DLECORE_VARIANT_RADIX2=ON
            -P "${CMAKE_CURRENT_SOURCE_DIR}/consumers/run_format_off_variant.cmake"
    )
    add_test(NAME liblecore_variant_radix2_off
        COMMAND "${CMAKE_COMMAND}"
            "-DLECORE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "-DLECORE_CONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/consumers/cmake-consumer"
            "-DLECORE_TEST_ROOT=${PROJECT_BINARY_DIR}/_consumer_tests/radix2-off"
            "-DLECORE_TEST_CONFIG=$<CONFIG>"
            "-DLECORE_TEST_GENERATOR=${CMAKE_GENERATOR}"
            "-DLECORE_TEST_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DLECORE_TEST_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            -DLECORE_VARIANT_FORMAT=ON
            -DLECORE_VARIANT_RADIX2=OFF
            -P "${CMAKE_CURRENT_SOURCE_DIR}/consumers/run_format_off_variant.cmake"
    )
    add_test(NAME liblecore_variant_examples_only
        COMMAND "${CMAKE_COMMAND}"
            "-DLECORE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "-DLECORE_TEST_ROOT=${PROJECT_BINARY_DIR}/_consumer_tests/examples-only"
            "-DLECORE_TEST_GENERATOR=${CMAKE_GENERATOR}"
            "-DLECORE_TEST_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DLECORE_TEST_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/consumers/run_examples_only.cmake"
    )
endif()

if(LECORE_BUILD_SHARED AND NOT CMAKE_CROSSCOMPILING AND NOT LECORE_ENABLE_SANITIZERS)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    set(python_conformance_script
        "${PROJECT_SOURCE_DIR}/../../tests/native/test_liblecore_conformance.py")
    if(Python3_Interpreter_FOUND AND EXISTS "${python_conformance_script}")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
            RESULT_VARIABLE numpy_import_result
            OUTPUT_QUIET
            ERROR_QUIET
        )
    else()
        set(numpy_import_result 1)
    endif()
    if(Python3_Interpreter_FOUND AND EXISTS "${python_conformance_script}"
       AND numpy_import_result EQUAL 0)
        add_test(NAME liblecore_python_conformance
            COMMAND "${Python3_EXECUTABLE}"
                "${python_conformance_script}"
                --library "$<TARGET_FILE:lecore>"
        )
    else()
        message(STATUS
            "Skipping Python conformance test (Python 3, NumPy, or harness unavailable)")
    endif()
endif()
