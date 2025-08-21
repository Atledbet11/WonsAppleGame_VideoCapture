# cmake/copy_dlls.cmake
if(NOT DEFINED src OR NOT DEFINED dst)
  message(FATAL_ERROR "copy_dlls.cmake: 'src' and 'dst' must be defined")
endif()

file(GLOB _dlls "${src}/*.dll")
if(NOT _dlls)
  message(WARNING "copy_dlls.cmake: no DLLs found in: ${src}")
endif()

file(MAKE_DIRECTORY "${dst}")
foreach(_f IN LISTS _dlls)
  message(STATUS "Copying: ${_f} -> ${dst}")
  file(COPY "${_f}" DESTINATION "${dst}")
endforeach()