cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")


# get and append paths for finding dep
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})


# find packages & third_party
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)
find_package(Torch REQUIRED)

# Temp fix for PyTorch 1.5.
set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python.so")

# message(${CMAKE_CURRENT_SOURCE_DIR}/../)
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11 third_party/pybind11
)


# lib for other c++ programs
add_library(rela_lib
  transition.cc
  batcher.cc
  batch_runner.cc
  context.cc
)
target_include_directories(rela_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(rela_lib PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela_lib PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela_lib PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES} )


# python lib
pybind11_add_module(rela pybind.cc)
target_include_directories(rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(rela PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela PUBLIC rela_lib ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES} )
