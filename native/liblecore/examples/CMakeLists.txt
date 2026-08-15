add_executable(liblecore_hrr_example hrr_example.c)
target_compile_features(liblecore_hrr_example PRIVATE c_std_11)
target_link_libraries(liblecore_hrr_example PRIVATE lecore::lecore)

add_executable(liblecore_hrr_cpp_example hrr_example.cpp)
target_compile_features(liblecore_hrr_cpp_example PRIVATE cxx_std_17)
target_link_libraries(liblecore_hrr_cpp_example PRIVATE lecore::lecore)

if(WIN32 AND LECORE_BUILD_SHARED)
    foreach(example_target IN ITEMS
            liblecore_hrr_example liblecore_hrr_cpp_example)
        add_custom_command(TARGET "${example_target}" POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:lecore>" "$<TARGET_FILE_DIR:${example_target}>"
            VERBATIM
        )
    endforeach()
endif()
