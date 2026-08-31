cmake_minimum_required(VERSION 3.21)
project(liblecore_pkgconfig_consumer LANGUAGES C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(liblecore REQUIRED IMPORTED_TARGET liblecore)

add_executable(liblecore_consumer main.c)
target_compile_features(liblecore_consumer PRIVATE c_std_11)
target_link_libraries(liblecore_consumer PRIVATE PkgConfig::liblecore)
