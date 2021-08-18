cmake_minimum_required (VERSION 2.8.11)
project (hanabi_learning_environment)

set(CMAKE_C_FLAGS "-O3 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -Wall -Wextra -fPIC -Wno-sign-compare")

add_subdirectory (hanabi_lib)

add_library (pyhanabi SHARED pyhanabi.cc)
target_link_libraries (pyhanabi LINK_PUBLIC hanabi)

add_executable (game_example game_example.cc)
target_link_libraries (game_example LINK_PUBLIC hanabi)
