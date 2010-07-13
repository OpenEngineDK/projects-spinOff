# Project name. Should be the same as the project directory name and
# will become the name of the binary target.
SET( PROJECT_NAME "spinOff")
SET( CUDA_KERNELS "CUDA_KERNELS")

CUDA_ADD_LIBRARY(${CUDA_KERNELS}
  FFT_wrap/fft.cu
  FFT_wrap/batchfft.cu
)


# Project source code list
SET( PROJECT_SOURCES
  # Add all the cpp source files here
  main.cpp
  Science/MRIModule.h
  Science/MRIModule.cpp

)

# Include needed to use SDL under Mac OS X
IF(APPLE)
  SET(PROJECT_SOURCES ${PROJECT_SOURCES}  ${SDL_MAIN_FOR_MAC})
ENDIF(APPLE)

# Project executable
ADD_EXECUTABLE(${PROJECT_NAME}
  ${PROJECT_SOURCES}
)

CUDA_ADD_CUFFT_TO_TARGET(${PROJECT_NAME})
CUDA_ADD_CUBLAS_TO_TARGET(${PROJECT_NAME})
# Project dependencies
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
  # Core library dependencies
  OpenEngine_Core
  OpenEngine_Logging
  OpenEngine_Display
  OpenEngine_Devices
  OpenEngine_Scene
  # Extension dependencies
  Extensions_SDL
  Extensions_SetupHelpers  
  Extensions_SpinOff  
  Extensions_GenericHandlers
  ${CUDA_KERNELS}
)
