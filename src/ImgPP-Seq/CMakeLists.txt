include(CompileOptions)

find_package( OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "OpenMP found, version: ${OpenMP_VERSION}")
    add_definitions(-DUSE_OPENMP)
else()
    message(WARNING "OpenMP not found - building without parallelization")
endif()

set(target_name imgpp)

add_executable(${target_name})

# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#     target_compile_options(${target_name} PRIVATE -fopenmp)
# endif()
target_compile_options(${target_name} PRIVATE -fopenmp)

target_sources(
  ${target_name}
  PRIVATE
    imgpp.cpp
)

# include_directories(  )

target_link_libraries(
  ${target_name}
  PRIVATE
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
  pp
  configuration
)

target_include_directories(
  ${target_name}
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
# "${CMAKE_SOURCE_DIR}/src"

# CLI11::CLI11

set_compile_options(${target_name})