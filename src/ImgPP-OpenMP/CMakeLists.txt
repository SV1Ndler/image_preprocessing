include(CompileOptions)

find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "OpenMP found, version: ${OpenMP_VERSION}")
    add_definitions(-DUSE_OPENMP)
else()
    message(WARNING "OpenMP not found - building without parallelization")
endif()

set(target_name imgpp_openmp)

add_executable(${target_name})

target_sources(
  ${target_name}
  PRIVATE
    imgpp.cpp
)

target_link_libraries(
  ${target_name}
  PRIVATE
  OpenMP::OpenMP_CXX
  pp
)

set_compile_options(${target_name})