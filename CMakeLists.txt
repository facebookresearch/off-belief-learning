CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
project(hanalearn)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -Wno-attributes -fPIC -march=native -Wfatal-errors")

execute_process(
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/get_pybind_flags.py
  OUTPUT_VARIABLE PYBIND_FLAGS
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND_FLAGS}")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi-learning-environment)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)

# find pytorch
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)
# find_package(PythonInterp 3.7 REQUIRED)
# find_package(PythonLibs 3.7 REQUIRED)
set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python.so")

message("---------------------")
message(${TorchPath})
message(${PYBIND_FLAGS})
message(${CMAKE_CXX_FLAGS})
message("---------------------")

# rela lib
add_library(rela_lib
  rela/transition.cc
  rela/batcher.cc
  rela/batch_runner.cc
  rela/context.cc
)
target_include_directories(rela_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rela_lib PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela_lib PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela_lib PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})

pybind11_add_module(rela rela/pybind.cc)
target_include_directories(rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rela PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rela PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela PUBLIC rela_lib ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})

# final hanalearn lib
pybind11_add_module(
  hanalearn
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/clone_data_generator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/r2d2_actor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/searchcc/hand_dist.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/searchcc/hybrid_model.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/searchcc/rl_search.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/searchcc/sim_actor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/searchcc/sparta.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/searchcc/thread_loop.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/searchcc/finesse.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/rlcc/pybind.cc
)
target_link_libraries(hanalearn PUBLIC hanabi)
target_link_libraries(hanalearn PUBLIC rela_lib)
target_include_directories(hanalearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
