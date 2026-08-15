cmake_minimum_required(VERSION 3.21)

foreach(required_var IN ITEMS
        LECORE_BUILD_DIR
        LECORE_CONSUMER_SOURCE
        LECORE_INSTALL_BINDIR
        LECORE_INSTALL_LIBDIR
        LECORE_TEST_ROOT)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "${required_var} is required")
    endif()
endforeach()

set(stage_dir "${LECORE_TEST_ROOT}/stage")
set(consumer_build_dir "${LECORE_TEST_ROOT}/build")
file(REMOVE_RECURSE "${LECORE_TEST_ROOT}")
file(MAKE_DIRECTORY "${LECORE_TEST_ROOT}")

set(config_args)
if(DEFINED LECORE_TEST_CONFIG AND NOT LECORE_TEST_CONFIG STREQUAL "")
    list(APPEND config_args --config "${LECORE_TEST_CONFIG}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${LECORE_BUILD_DIR}" --prefix "${stage_dir}" ${config_args}
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Installing liblecore for the consumer test failed: ${install_result}")
endif()

set(required_install_files
    "${stage_dir}/include/lecore/lecore.h"
    "${stage_dir}/${LECORE_INSTALL_LIBDIR}/cmake/lecore/lecoreConfig.cmake"
    "${stage_dir}/${LECORE_INSTALL_LIBDIR}/cmake/lecore/lecoreTargets.cmake"
    "${stage_dir}/${LECORE_INSTALL_LIBDIR}/pkgconfig/liblecore.pc"
    "${stage_dir}/share/liblecore/VERSION"
    "${stage_dir}/share/liblecore/ISA_VERSION"
    "${stage_dir}/share/doc/liblecore/README.md"
    "${stage_dir}/share/doc/liblecore/CHANGELOG.md"
    "${stage_dir}/share/doc/liblecore/PROVENANCE.md"
    "${stage_dir}/share/licenses/liblecore/LICENSE"
)
foreach(required_file IN LISTS required_install_files)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "The installation is missing ${required_file}")
    endif()
endforeach()

if(DEFINED LECORE_EXPECT_FORMAT AND LECORE_EXPECT_FORMAT)
    if(NOT EXISTS "${stage_dir}/include/lecore/lecore_format.h")
        message(FATAL_ERROR "A format-enabled installation is missing lecore_format.h")
    endif()
endif()

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${LECORE_CONSUMER_SOURCE}"
    -B "${consumer_build_dir}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_PREFIX_PATH=${stage_dir}"
)

if(DEFINED LECORE_USE_PKG_CONFIG AND LECORE_USE_PKG_CONFIG)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "PKG_CONFIG_PATH=${stage_dir}/${LECORE_INSTALL_LIBDIR}/pkgconfig"
            ${configure_command}
        RESULT_VARIABLE configure_result
    )
else()
    execute_process(COMMAND ${configure_command} RESULT_VARIABLE configure_result)
endif()
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Configuring the installed consumer failed: ${configure_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_build_dir}" ${config_args}
    RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Building the installed consumer failed: ${build_result}")
endif()

set(consumer_executable "${consumer_build_dir}/liblecore_consumer${CMAKE_EXECUTABLE_SUFFIX}")
if(DEFINED LECORE_TEST_CONFIG AND NOT LECORE_TEST_CONFIG STREQUAL "")
    set(multiconfig_executable
        "${consumer_build_dir}/${LECORE_TEST_CONFIG}/liblecore_consumer${CMAKE_EXECUTABLE_SUFFIX}")
    if(EXISTS "${multiconfig_executable}")
        set(consumer_executable "${multiconfig_executable}")
    endif()
endif()

if(WIN32 AND LECORE_EXPECT_SHARED)
    file(GLOB installed_runtime_dlls
        "${stage_dir}/${LECORE_INSTALL_BINDIR}/*lecore*.dll")
    list(LENGTH installed_runtime_dlls installed_runtime_count)
    if(NOT installed_runtime_count EQUAL 1)
        message(FATAL_ERROR
            "Expected one installed liblecore runtime DLL, found ${installed_runtime_count}")
    endif()
    get_filename_component(consumer_executable_dir
        "${consumer_executable}" DIRECTORY)
    file(COPY "${installed_runtime_dlls}" DESTINATION
        "${consumer_executable_dir}")
endif()

execute_process(COMMAND "${consumer_executable}" RESULT_VARIABLE run_result)
if(NOT run_result EQUAL 0)
    message(FATAL_ERROR "The installed consumer failed at runtime: ${run_result}")
endif()
