include(CompileOptions)

find_package( OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "OpenMP found, version: ${OpenMP_VERSION}")
    add_definitions(-DUSE_OPENMP)
else()
    message(WARNING "OpenMP not found - building without parallelization")
endif()

set(target_name imgpp_openmp)

add_executable(${target_name})

# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#     target_compile_options(${target_name} PRIVATE -fopenmp)
# endif()
# target_compile_options(${target_name} PRIVATE -fopenmp)

target_sources(
  ${target_name}
  PRIVATE
    imgpp.cpp
)

# include_directories(  )

target_link_libraries(
  ${target_name}
  PRIVATE
  OpenMP::OpenMP_CXX
  pp
)


# "${CMAKE_SOURCE_DIR}/src"

# CLI11::CLI11

set_compile_options(${target_name})