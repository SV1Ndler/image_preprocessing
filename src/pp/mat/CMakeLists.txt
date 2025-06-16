target_sources(
  ${target_name}
  PRIVATE
    mat.cpp
)

# #TEST
# set(test_target_name "${target_name}_cipher_test")

# add_executable(${test_target_name})

# target_sources(
#   ${test_target_name}
#   PRIVATE
#     cipher.test.cpp
# )

# target_link_libraries(
#   ${test_target_name}
#   PRIVATE
#     ${target_name}
#     gtest
#     gtest_main
# )

# set_compile_options(${target_name})

# add_test(
#   NAME ${test_target_name}
#   COMMAND ${test_target_name}
# )