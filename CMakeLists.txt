CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
project(hanalearn)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

execute_process(
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/get_pybind_flags.py
  OUTPUT_VARIABLE PYBIND_FLAGS
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYBIND11_COMPILER_TYPE=\\\"_gcc\\\" -DPYBIND11_STDLIB=\\\"_libstdcpp\\\" -DPYBIND11_BUILD_ABI=\\\"_cxxabi1011\\\"")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/rela)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi-learning-environment)

find_package(Torch REQUIRED)

pybind11_add_module(
  hanalearn
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/clone_data_generator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/r2d2_actor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/pybind.cc
)
target_link_libraries(hanalearn PUBLIC hanabi)
target_link_libraries(hanalearn PUBLIC rela_lib)
target_include_directories(hanalearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
