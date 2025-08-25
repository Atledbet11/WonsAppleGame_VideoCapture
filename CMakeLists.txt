cmake_minimum_required(VERSION 3.10.0)
project(WonsAppleGame_VideoCapture VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include(CTest)
# enable_testing()

option(USE_LOCAL_OPENCV "Use local OpenCV (CUDA) on this machine" OFF)

if(USE_LOCAL_OPENCV)
    # Personal OpenCV build with Cuda
    set(OpenCV_DIR "F:/ComputerScience/opencv/build/install/x64/vc16/lib")
else()
    # Redistributable CPU bound build of OpenCV
    set(OpenCV_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/opencv/runtime/x64/vc16/lib")
endif()

if(MSVC)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(WonsAppleGame_VideoCapture main.cpp)

target_link_libraries(WonsAppleGame_VideoCapture ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Helper script path
get_filename_component(_SCRIPT_COPY_DLLS
  "${CMAKE_CURRENT_LIST_DIR}/cmake/copy_dlls.cmake" ABSOLUTE)

# Resolve OpenCV bin robustly from OpenCV_DIR:
#   .../lib               -> .../bin (../bin)
#   .../lib/cmake/opencv4 -> .../bin (../../../bin)
set(_OPENCV_BIN_DIR "")
if(EXISTS "${OpenCV_DIR}/../bin")
  get_filename_component(_OPENCV_BIN_DIR "${OpenCV_DIR}/../bin" ABSOLUTE)
elseif(EXISTS "${OpenCV_DIR}/../../../bin")
  get_filename_component(_OPENCV_BIN_DIR "${OpenCV_DIR}/../../../bin" ABSOLUTE)
else()
  message(WARNING "Could not infer OpenCV bin dir from OpenCV_DIR='${OpenCV_DIR}'")
endif()

# CUDA bin dir (don’t require modern CMake)
set(_CUDA_BIN_DIR "")
# If you also want the modern way and have CMake >= 3.17, you can uncomment:
# if(USE_LOCAL_OPENCV)
#   find_package(CUDAToolkit QUIET)  # fills CUDAToolkit_BIN_DIR on CUDA 11+
# endif()
if(DEFINED CUDAToolkit_BIN_DIR AND EXISTS "${CUDAToolkit_BIN_DIR}")
  set(_CUDA_BIN_DIR "${CUDAToolkit_BIN_DIR}")
elseif(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin")
  set(_CUDA_BIN_DIR "$ENV{CUDA_PATH}/bin")
endif()

message(STATUS "OpenCV bin: ${_OPENCV_BIN_DIR}")
if(USE_LOCAL_OPENCV)
  message(STATUS "CUDA bin:   ${_CUDA_BIN_DIR}")
endif()

# Copy *.dll after building (per preset)
if(NOT USE_LOCAL_OPENCV)
  if(_OPENCV_BIN_DIR)
    add_custom_command(TARGET WonsAppleGame_VideoCapture POST_BUILD
      COMMAND ${CMAKE_COMMAND} -Dsrc=${_OPENCV_BIN_DIR}
                                 -Ddst=$<TARGET_FILE_DIR:WonsAppleGame_VideoCapture>
                                 -P ${_SCRIPT_COPY_DLLS}
      COMMENT "Copying CPU OpenCV DLLs"
      VERBATIM)
  endif()
else()
  if(_OPENCV_BIN_DIR)
    add_custom_command(TARGET WonsAppleGame_VideoCapture POST_BUILD
      COMMAND ${CMAKE_COMMAND} -Dsrc=${_OPENCV_BIN_DIR}
                                 -Ddst=$<TARGET_FILE_DIR:WonsAppleGame_VideoCapture>
                                 -P ${_SCRIPT_COPY_DLLS}
      COMMENT "Copying CUDA OpenCV DLLs"
      VERBATIM)
  endif()
  if(_CUDA_BIN_DIR)
    add_custom_command(TARGET WonsAppleGame_VideoCapture POST_BUILD
      COMMAND ${CMAKE_COMMAND} -Dsrc=${_CUDA_BIN_DIR}
                                 -Ddst=$<TARGET_FILE_DIR:WonsAppleGame_VideoCapture>
                                 -P ${_SCRIPT_COPY_DLLS}
      COMMENT "Copying NVIDIA CUDA runtime DLLs"
      VERBATIM)
  else()
    message(WARNING "CUDA bin folder not found; CUDA DLLs will not be copied.")
  endif()
endif()
