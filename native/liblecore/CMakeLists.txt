cmake_minimum_required(VERSION 3.21)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" LECORE_VERSION LIMIT_COUNT 1)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/ISA_VERSION" LECORE_ISA_VERSION LIMIT_COUNT 1)
if(NOT LECORE_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "VERSION must contain one MAJOR.MINOR.PATCH version")
endif()
if(NOT LECORE_ISA_VERSION MATCHES "^[0-9]+$")
    message(FATAL_ERROR "ISA_VERSION must contain one non-negative integer")
endif()

project(liblecore VERSION "${LECORE_VERSION}" DESCRIPTION "Portable leCore vector algebra kernel" LANGUAGES C)

include(CMakePackageConfigHelpers)
include(CheckCSourceRuns)
include(GNUInstallDirs)

option(LECORE_ENABLE_FORMAT "Build the optional interchange descriptor codec" ON)
option(LECORE_ENABLE_RADIX2 "Build the portable radix-2 HRR backend" ON)
option(LECORE_BUILD_SHARED "Build liblecore as a shared library instead of a static library" OFF)
option(LECORE_BUILD_TESTS "Build liblecore's tests" "${PROJECT_IS_TOP_LEVEL}")
option(LECORE_BUILD_EXAMPLES "Build liblecore's examples" "${PROJECT_IS_TOP_LEVEL}")
option(LECORE_BUILD_AMALGAMATION "Regenerate the checked-in amalgamation during the default build" OFF)
option(LECORE_BUILD_FUZZERS "Build bounded libFuzzer public-API targets" OFF)
option(LECORE_WARNINGS_AS_ERRORS "Treat warnings from liblecore sources as errors" OFF)
option(LECORE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LECORE_ASSUME_IEC_60559
    "Assert IEC 60559 NaN/Inf/evaluation behavior when cross-compiling" OFF)

if(LECORE_BUILD_FUZZERS AND NOT LECORE_ENABLE_SANITIZERS)
    message(FATAL_ERROR
        "LECORE_BUILD_FUZZERS requires LECORE_ENABLE_SANITIZERS=ON")
endif()
if(LECORE_BUILD_FUZZERS AND NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "LECORE_BUILD_FUZZERS currently requires Clang")
endif()

if(CMAKE_CROSSCOMPILING)
    if(NOT LECORE_ASSUME_IEC_60559)
        message(FATAL_ERROR
            "Cross builds must explicitly set LECORE_ASSUME_IEC_60559=ON after validating the target")
    endif()
else()
    set(LECORE_SAVED_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    set(LECORE_SAVED_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")
    if(MSVC)
        string(APPEND CMAKE_REQUIRED_FLAGS " /fp:strict")
    else()
        string(APPEND CMAKE_REQUIRED_FLAGS
            " -fno-fast-math -ffp-contract=off")
        if(NOT APPLE)
            list(APPEND CMAKE_REQUIRED_LIBRARIES m)
        endif()
    endif()
    check_c_source_runs([=[
        #include <float.h>
        #include <fenv.h>
        #include <math.h>

        int main(void) {
            volatile double one = 1.0;
            volatile double zero = 0.0;
            volatile float large = 16777216.0F;
            double infinity = one / zero;
            double not_a_number = zero / zero;
            float rounded = large + 1.0F;

            if (FLT_EVAL_METHOD != 0 || fegetround() != FE_TONEAREST) {
                return 1;
            }
            if (!isinf(infinity) || signbit(infinity) ||
                !isinf(-infinity) || !signbit(-infinity)) {
                return 2;
            }
            if (!isnan(not_a_number) || !isnan(not_a_number + one)) {
                return 3;
            }
            if (rounded != 16777216.0F) {
                return 4;
            }
            return 0;
        }
    ]=] LECORE_IEC_60559_RUNTIME_OK)
    set(CMAKE_REQUIRED_FLAGS "${LECORE_SAVED_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_LIBRARIES "${LECORE_SAVED_REQUIRED_LIBRARIES}")
    unset(LECORE_SAVED_REQUIRED_FLAGS)
    unset(LECORE_SAVED_REQUIRED_LIBRARIES)
    if(NOT LECORE_IEC_60559_RUNTIME_OK)
        message(FATAL_ERROR
            "The compiler/runtime failed liblecore's IEC 60559 and evaluation probe")
    endif()
endif()

set(LECORE_ABI_VERSION 0)
get_filename_component(LECORE_REPOSITORY_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(LECORE_AMALGAMATION_GENERATOR
    "${LECORE_REPOSITORY_ROOT}/tools/generate_liblecore_amalgamation.py")
set(LECORE_AMALGAMATION_HEADER
    "${CMAKE_CURRENT_SOURCE_DIR}/amalgamation/lecore.h")
set(LECORE_AMALGAMATION_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/amalgamation/lecore.c")

find_package(Python3 COMPONENTS Interpreter QUIET)
if(EXISTS "${LECORE_AMALGAMATION_GENERATOR}" AND Python3_Interpreter_FOUND)
    if(LECORE_BUILD_AMALGAMATION)
        add_custom_target(liblecore_amalgamation_generate ALL
            COMMAND "${Python3_EXECUTABLE}" "${LECORE_AMALGAMATION_GENERATOR}"
            WORKING_DIRECTORY "${LECORE_REPOSITORY_ROOT}"
            COMMENT "Regenerating the liblecore amalgamation"
            VERBATIM
        )
    else()
        add_custom_target(liblecore_amalgamation_generate
            COMMAND "${Python3_EXECUTABLE}" "${LECORE_AMALGAMATION_GENERATOR}"
            WORKING_DIRECTORY "${LECORE_REPOSITORY_ROOT}"
            COMMENT "Regenerating the liblecore amalgamation"
            VERBATIM
        )
    endif()
    add_custom_target(liblecore_amalgamation_check
        COMMAND "${Python3_EXECUTABLE}"
            "${LECORE_AMALGAMATION_GENERATOR}" --check
        WORKING_DIRECTORY "${LECORE_REPOSITORY_ROOT}"
        COMMENT "Checking the liblecore amalgamation for source drift"
        VERBATIM
    )
elseif(LECORE_BUILD_AMALGAMATION)
    message(FATAL_ERROR
        "LECORE_BUILD_AMALGAMATION requires Python 3 and the repository generator")
endif()

foreach(amalgamation_file IN ITEMS
        "${LECORE_AMALGAMATION_HEADER}" "${LECORE_AMALGAMATION_SOURCE}")
    if(NOT EXISTS "${amalgamation_file}")
        message(FATAL_ERROR
            "Missing checked-in amalgamation file: ${amalgamation_file}")
    endif()
endforeach()

set(LECORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cleanup_f32.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cleanup_f64.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/context.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hrr_direct_f32.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hrr_direct_f64.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/status.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vector_f32.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/vector_f64.c"
)
set(LECORE_FORMAT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/crc64.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/format.c"
)
set(LECORE_RADIX2_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hrr_radix2_f32.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hrr_radix2_f64.c"
)
if(LECORE_ENABLE_FORMAT)
    foreach(format_source IN LISTS LECORE_FORMAT_SOURCES)
        if(NOT EXISTS "${format_source}")
            message(FATAL_ERROR "LECORE_ENABLE_FORMAT requires ${format_source}")
        endif()
    endforeach()
    list(APPEND LECORE_SOURCES ${LECORE_FORMAT_SOURCES})
endif()
if(LECORE_ENABLE_RADIX2)
    list(APPEND LECORE_SOURCES ${LECORE_RADIX2_SOURCES})
endif()
foreach(lecore_source IN LISTS LECORE_SOURCES)
    if(NOT EXISTS "${lecore_source}")
        message(FATAL_ERROR "A required liblecore source is missing: ${lecore_source}")
    endif()
endforeach()

if(LECORE_BUILD_SHARED)
    set(LECORE_LIBRARY_TYPE SHARED)
else()
    set(LECORE_LIBRARY_TYPE STATIC)
endif()
add_library(lecore ${LECORE_LIBRARY_TYPE} ${LECORE_SOURCES})
add_library(lecore::lecore ALIAS lecore)

target_compile_features(lecore PUBLIC c_std_11)
set_target_properties(lecore PROPERTIES
    C_EXTENSIONS OFF
    C_VISIBILITY_PRESET hidden
    EXPORT_NAME lecore
    OUTPUT_NAME lecore
    POSITION_INDEPENDENT_CODE ON
    SOVERSION "${LECORE_ABI_VERSION}"
    VERSION "${PROJECT_VERSION}"
    VISIBILITY_INLINES_HIDDEN YES
)
target_include_directories(lecore
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_definitions(lecore
    PUBLIC
        LECORE_ENABLE_FORMAT=$<BOOL:${LECORE_ENABLE_FORMAT}>
        LECORE_ENABLE_RADIX2=$<BOOL:${LECORE_ENABLE_RADIX2}>
        LECORE_SHARED_LIBRARY=$<BOOL:${LECORE_BUILD_SHARED}>
        $<$<BOOL:${LECORE_BUILD_SHARED}>:LECORE_SHARED=1>
    PRIVATE
        LECORE_BUILDING_LIBRARY=1
        LECORE_ABI_VERSION_VALUE=${LECORE_ABI_VERSION}
        LECORE_ISA_VERSION_VALUE=${LECORE_ISA_VERSION}
        LECORE_VERSION_STRING_VALUE="${LECORE_VERSION}"
)

if(MSVC)
    target_compile_options(lecore PRIVATE /W4 /permissive- /fp:strict)
    if(LECORE_WARNINGS_AS_ERRORS)
        target_compile_options(lecore PRIVATE /WX)
    endif()
else()
    target_compile_options(lecore PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        -fno-fast-math -ffp-contract=off
    )
    if(LECORE_WARNINGS_AS_ERRORS)
        target_compile_options(lecore PRIVATE -Werror)
    endif()
    if(NOT APPLE)
        target_link_libraries(lecore PUBLIC m)
    endif()
endif()

if(LECORE_ENABLE_SANITIZERS)
    if(MSVC)
        message(FATAL_ERROR "LECORE_ENABLE_SANITIZERS currently requires Clang or GCC")
    endif()
    target_compile_options(lecore PUBLIC
        "$<BUILD_INTERFACE:-fsanitize=address,undefined>"
        "$<BUILD_INTERFACE:-fno-omit-frame-pointer>"
    )
    target_link_options(lecore PUBLIC
        "$<BUILD_INTERFACE:-fsanitize=address,undefined>"
        "$<BUILD_INTERFACE:-fno-omit-frame-pointer>"
    )
endif()

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/lecoreConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lecoreConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/lecore"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/lecoreConfigVersion.cmake"
    VERSION "${LECORE_VERSION}"
    # ABI-0 may break between 0.x minors; do not let 0.2 satisfy a 0.1 request.
    COMPATIBILITY SameMinorVersion
)

if(MSVC)
    set(LECORE_PC_PUBLIC_LIBS "")
    set(LECORE_PC_PRIVATE_LIBS "")
elseif(LECORE_BUILD_SHARED)
    set(LECORE_PC_PUBLIC_LIBS "")
    set(LECORE_PC_PRIVATE_LIBS "-lm")
else()
    set(LECORE_PC_PUBLIC_LIBS "-lm")
    set(LECORE_PC_PRIVATE_LIBS "")
endif()
if(LECORE_ENABLE_FORMAT)
    set(LECORE_PC_CFLAGS "-DLECORE_ENABLE_FORMAT=1")
else()
    set(LECORE_PC_CFLAGS "-DLECORE_ENABLE_FORMAT=0")
endif()
if(LECORE_ENABLE_RADIX2)
    string(APPEND LECORE_PC_CFLAGS " -DLECORE_ENABLE_RADIX2=1")
else()
    string(APPEND LECORE_PC_CFLAGS " -DLECORE_ENABLE_RADIX2=0")
endif()
if(LECORE_BUILD_SHARED)
    string(APPEND LECORE_PC_CFLAGS " -DLECORE_SHARED_LIBRARY=1 -DLECORE_SHARED=1")
else()
    string(APPEND LECORE_PC_CFLAGS " -DLECORE_SHARED_LIBRARY=0")
endif()
if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
    set(LECORE_PC_PREFIX "${CMAKE_INSTALL_PREFIX}")
    set(LECORE_PC_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
else()
    file(RELATIVE_PATH LECORE_PC_PREFIX_RELATIVE
        "/${CMAKE_INSTALL_LIBDIR}/pkgconfig" "/")
    set(LECORE_PC_PREFIX "\${pcfiledir}/${LECORE_PC_PREFIX_RELATIVE}")
    set(LECORE_PC_LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
endif()
if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
    set(LECORE_PC_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
else()
    set(LECORE_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
endif()
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/pkgconfig/liblecore.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/liblecore.pc"
    @ONLY
)

install(TARGETS lecore
    EXPORT liblecoreTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/lecore/lecore.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lecore"
)
if(LECORE_ENABLE_FORMAT)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/lecore/lecore_format.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lecore"
    )
endif()
install(EXPORT liblecoreTargets
    FILE lecoreTargets.cmake
    NAMESPACE lecore::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/lecore"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/lecoreConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lecoreConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/lecore"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/liblecore.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
install(FILES VERSION ISA_VERSION ABI0_SYMBOLS.txt
    DESTINATION "${CMAKE_INSTALL_DATADIR}/liblecore")
install(FILES
    "${LECORE_AMALGAMATION_HEADER}"
    "${LECORE_AMALGAMATION_SOURCE}"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/liblecore/source/amalgamation"
)
set(LECORE_DOCUMENTATION_FILES)
foreach(documentation_file IN ITEMS README.md CHANGELOG.md PROVENANCE.md BENCHMARKS.md)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${documentation_file}")
        list(APPEND LECORE_DOCUMENTATION_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/${documentation_file}")
    endif()
endforeach()
if(LECORE_DOCUMENTATION_FILES)
    install(FILES ${LECORE_DOCUMENTATION_FILES} DESTINATION "${CMAKE_INSTALL_DOCDIR}")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../LICENSE")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../LICENSE"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/liblecore"
    )
endif()

if(LECORE_BUILD_EXAMPLES OR LECORE_BUILD_TESTS)
    enable_language(CXX)
endif()

if(LECORE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(LECORE_BUILD_TESTS)
    include(CTest)
    add_subdirectory(tests)
    set(LECORE_AMALGAMATION_CHECK_SCRIPT
        "${LECORE_REPOSITORY_ROOT}/tests/native/check_liblecore_amalgamation.py")
    if(Python3_Interpreter_FOUND AND EXISTS "${LECORE_AMALGAMATION_CHECK_SCRIPT}"
       AND NOT CMAKE_CROSSCOMPILING AND NOT MSVC)
        add_test(NAME liblecore_amalgamation_standalone
            COMMAND "${CMAKE_COMMAND}" -E env
                "CC=${CMAKE_C_COMPILER}"
                "CXX=${CMAKE_CXX_COMPILER}"
                "${Python3_EXECUTABLE}"
                "${LECORE_AMALGAMATION_CHECK_SCRIPT}"
        )
    endif()
    set(LECORE_SYMBOL_CHECK_SCRIPT
        "${LECORE_REPOSITORY_ROOT}/tests/native/check_liblecore_symbols.py")
    if(LECORE_BUILD_SHARED AND UNIX AND NOT WIN32 AND NOT CMAKE_CROSSCOMPILING
       AND Python3_Interpreter_FOUND AND CMAKE_NM
       AND EXISTS "${LECORE_SYMBOL_CHECK_SCRIPT}")
        if(LECORE_ENABLE_FORMAT)
            set(LECORE_SYMBOL_CHECK_FORMAT on)
        else()
            set(LECORE_SYMBOL_CHECK_FORMAT off)
        endif()
        add_test(NAME liblecore_abi0_symbols
            COMMAND "${Python3_EXECUTABLE}"
                "${LECORE_SYMBOL_CHECK_SCRIPT}"
                --library "$<TARGET_FILE:lecore>"
                --nm "${CMAKE_NM}"
                --format "${LECORE_SYMBOL_CHECK_FORMAT}"
        )
    endif()
endif()

if(LECORE_BUILD_FUZZERS)
    add_subdirectory(tests/fuzz)
endif()
