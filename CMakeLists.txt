cmake_minimum_required(VERSION 3.15)
project(eiko_skelfm C)
set(CMAKE_C_STANDARD 99)

add_library(eiko_skelfm STATIC
    src/common.c src/msfm2d.c src/msfm3d.c
)
target_include_directories(eiko_skelfm PUBLIC src/)
if(UNIX)
    target_link_libraries(eiko_skelfm m)  # libm on Linux/macOS; not needed on Windows
endif()
