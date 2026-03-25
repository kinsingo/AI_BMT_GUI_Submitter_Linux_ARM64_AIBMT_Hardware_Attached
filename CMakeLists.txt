cmake_minimum_required(VERSION 3.14)

project(AI_BMT_GUI_Submitter_App VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_SOURCES main.cpp)

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
# Headers from the official 1.20.0 ARM64 release package (third_party/).
# The .so files are symlinked to the GPU-enabled library in the ai-bmt conda env.
set(ONNXRUNTIME_DIR      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)
set(ONNXRUNTIME_INCLUDE  ${ONNXRUNTIME_DIR}/include)
set(ONNXRUNTIME_LIB_DIR  ${ONNXRUNTIME_DIR}/lib)
# Runtime provider plugin directory (CUDA / TensorRT .so files reside here)
set(ORT_PROVIDER_DIR
    "/home/jhyi/miniconda3/envs/ai-bmt/lib/python3.10/site-packages/onnxruntime/capi")

# ── OpenCV ────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(AI_BMT_GUI_Submitter ${PROJECT_SOURCES})

# Include directories
target_include_directories(AI_BMT_GUI_Submitter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(AI_BMT_GUI_Submitter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/build/libAI_BMT_GUI_Library.so
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
    ${OpenCV_LIBS}
)

# Set RPATH so the binary can locate onnxruntime and its provider plugins at runtime
set_target_properties(AI_BMT_GUI_Submitter PROPERTIES
    BUILD_RPATH   "${CMAKE_BINARY_DIR}/lib;${ONNXRUNTIME_LIB_DIR};${ORT_PROVIDER_DIR}"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${ONNXRUNTIME_LIB_DIR};${ORT_PROVIDER_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Ensure RPATH is always used (not stripped out)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
