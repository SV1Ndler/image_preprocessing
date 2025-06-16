include(CompileOptions)

set(target_name configuration)

add_library(
  ${target_name}
  STATIC
)

target_include_directories(
  ${target_name}
  PUBLIC
    "${CMAKE_SOURCE_DIR}/src"
)

target_link_libraries(
  ${target_name}
  PUBLIC
  nlohmann_json::nlohmann_json
  pp
)

set_compile_options(${target_name})

add_subdirectory(parser)
add_subdirectory(filter)