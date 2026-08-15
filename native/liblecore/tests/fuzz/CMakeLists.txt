function(lecore_add_fuzzer target_name source_file)
    add_executable("${target_name}" "${source_file}")
    target_compile_features("${target_name}" PRIVATE c_std_11)
    target_link_libraries("${target_name}" PRIVATE lecore::lecore)
    target_compile_options("${target_name}" PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
        -fno-fast-math -ffp-contract=off
        -fsanitize=fuzzer,address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options("${target_name}" PRIVATE
        -fsanitize=fuzzer,address,undefined
        -fno-omit-frame-pointer
    )
endfunction()

lecore_add_fuzzer(liblecore_fuzz_public_api fuzz_public_api.c)
if(LECORE_ENABLE_FORMAT)
    lecore_add_fuzzer(liblecore_fuzz_format fuzz_format.c)
endif()
