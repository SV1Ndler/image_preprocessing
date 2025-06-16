include(CompileOptions)

find_package(OpenMP REQUIRED)

set(target_name pp)

add_library(
  ${target_name}
  STATIC
)

target_link_libraries(
  ${target_name}
  PUBLIC
  OpenMP::OpenMP_CXX
)

target_include_directories(
  ${target_name}
  PUBLIC
    "${CMAKE_SOURCE_DIR}/src"
)

set_compile_options(${target_name})

add_subdirectory(mat)
add_subdirectory(pixel)
add_subdirectory(transformation)