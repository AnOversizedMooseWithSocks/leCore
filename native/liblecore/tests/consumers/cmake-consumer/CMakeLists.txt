cmake_minimum_required(VERSION 3.21)
project(liblecore_cmake_consumer LANGUAGES C CXX)

find_package(lecore CONFIG REQUIRED)

add_executable(liblecore_consumer main.cpp)
target_compile_features(liblecore_consumer PRIVATE cxx_std_17)
target_link_libraries(liblecore_consumer PRIVATE lecore::lecore)
