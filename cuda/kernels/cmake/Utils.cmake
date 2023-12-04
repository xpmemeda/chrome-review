function(get_torch_cmake_prefix_path var_name version_required)
  if (DEFINED CACHE{${var_name}})
    return()
  endif()
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE OUT_TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE RESULT_VARIABLE_TORCH_CMAKE_PREFIX_PATH
    ERROR_QUIET
  )
  if (NOT ${RESULT_VARIABLE_TORCH_CMAKE_PREFIX_PATH} EQUAL 0)
    message(WARNING "Failed to get torch cmake prefix path")
    return()
  endif()
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__version__)"
    OUTPUT_VARIABLE TORCH_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE RESULT_VARIABLE_TORCH_VERSION
    ERROR_QUIET
  )
  if (NOT ${RESULT_VARIABLE_TORCH_VERSION} EQUAL 0)
    message(WARNING "Failed to get torch version")
    return()
  endif()
  if (${TORCH_VERSION} VERSION_LESS ${version_required})
    message(WARNING "Torch version ${TORCH_VERSION} is less than ${version_required}")
    return()
  endif()
  set(${var_name} ${OUT_TORCH_CMAKE_PREFIX_PATH} CACHE PATH "Torch cmake prefix path" FORCE)
endfunction()
