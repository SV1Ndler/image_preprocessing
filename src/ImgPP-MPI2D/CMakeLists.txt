include(CompileOptions)

find_package(MPI REQUIRED)

# if(OpenMP_FOUND)
#     message(STATUS "OpenMP found, version: ${OpenMP_VERSION}")
#     add_definitions(-DUSE_OPENMP)
# else()
#     message(WARNING "OpenMP not found - building without parallelization")
# endif()

set(target_name imgpp_mpi_2d)

add_executable(${target_name})

target_sources(
  ${target_name}
  PRIVATE
    imgpp.cpp
)

target_link_libraries(
  ${target_name}
  PRIVATE
  MPI::MPI_CXX
  pp
)

# ${MPI_CXX_LIBRARIES}

# target_include_directories(
#   ${target_name}
#   PRIVATE
#   ${MPI_CXX_INCLUDE_PATH}
# )

# target_compile_options(
#   ${target_name}
#   PRIVATE
#   ${MPI_CXX_COMPILE_FLAGS}
# )

# CLI11::CLI11

set_compile_options(${target_name})