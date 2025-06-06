# reference to https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/dialects/torch/importer/jit_ir/cmake/modules/TorchMLIRPyTorch.cmake

# PNNXProbeForPyTorchInstall
# Attempts to find a Torch installation and set the Torch_ROOT variable
# based on introspecting the python environment. This allows a subsequent
# call to find_package(Torch) to work.
function(PNNXProbeForPyTorchInstall)
  if(Torch_ROOT)
    message(STATUS "Using cached Torch root = ${Torch_ROOT}")
  elseif(Torch_INSTALL_DIR)
    message(STATUS "Using cached Torch install dir = ${Torch_INSTALL_DIR}")
    set(Torch_DIR "${Torch_INSTALL_DIR}/share/cmake/Torch" CACHE STRING "Torch dir" FORCE)
  else()
    #find_package (Python3 COMPONENTS Interpreter Development)
    find_package (Python3)
    message(STATUS "Checking for PyTorch using ${Python3_EXECUTABLE} ...")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
      -c "import os;import torch;print(torch.utils.cmake_prefix_path, end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE PYTORCH_STATUS
      OUTPUT_VARIABLE PYTORCH_PACKAGE_DIR)
    if(NOT PYTORCH_STATUS EQUAL "0")
      message(STATUS "Unable to 'import torch' with ${Python3_EXECUTABLE} (fallback to explicit config)")
      return()
    endif()
    message(STATUS "Found PyTorch installation at ${PYTORCH_PACKAGE_DIR}")

    set(Torch_ROOT "${PYTORCH_PACKAGE_DIR}" CACHE STRING
        "Torch configure directory" FORCE)
  endif()
endfunction()
