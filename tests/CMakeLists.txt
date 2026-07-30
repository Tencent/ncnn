
if(MSVC)
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4996)
endif()

if(NCNN_WEBGPU)
    if(NCNN_WEBGPU_NODE_EXECUTABLE)
        set(ncnn_webgpu_node_command ${NCNN_WEBGPU_NODE_EXECUTABLE})
    elseif(CMAKE_CROSSCOMPILING_EMULATOR)
        set(ncnn_webgpu_node_command ${CMAKE_CROSSCOMPILING_EMULATOR})
    else()
        find_program(ncnn_webgpu_node_command node REQUIRED)
    endif()

    add_custom_target(webgpu_tests)
    set_property(TARGET webgpu_tests PROPERTY FOLDER "tests/webgpu")

    function(ncnn_add_webgpu_browser_test target)
        set(require_webgpu_argument)
        if(ARGC GREATER 1 AND "${ARGV1}" STREQUAL "REQUIRE_WEBGPU")
            list(APPEND require_webgpu_argument --require-webgpu)
        endif()

        target_link_options(${target} PRIVATE
            "--no-entry"
            "-sMODULARIZE=1"
            "-sEXPORT_ES6=1"
            "-sENVIRONMENT=worker"
            "-sEXPORTED_FUNCTIONS=['_main']"
            "-sEXPORTED_RUNTIME_METHODS=['callMain']")
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/webgpu"
            SUFFIX ".js")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/webgpu/webgpu_test.html"
                "$<TARGET_FILE_DIR:${target}>/webgpu_test.html"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/webgpu/webgpu_test_worker.mjs"
                "$<TARGET_FILE_DIR:${target}>/webgpu_test_worker.mjs")
        add_dependencies(webgpu_tests ${target})

        if(NCNN_TEST_CHROME)
            set(extra_arguments)
            if(NCNN_TEST_WEBGPU_ADAPTER STREQUAL "swiftshader")
                list(APPEND extra_arguments
                    --expected-adapter swiftshader
                    --swiftshader
                    --no-sandbox)
            else()
                list(APPEND extra_arguments --hardware)
            endif()

            add_test(
                NAME ${target}
                COMMAND ${ncnn_webgpu_node_command}
                    "${CMAKE_CURRENT_SOURCE_DIR}/webgpu/run_webgpu_test.mjs"
                    --chrome "${NCNN_TEST_CHROME}"
                    --page "$<TARGET_FILE_DIR:${target}>/webgpu_test.html"
                    --module "$<TARGET_FILE_NAME:${target}>"
                    --test-name ${target}
                    --expected-version "${NCNN_WEBGPU_CHROME_VERSION}"
                    ${require_webgpu_argument}
                    ${extra_arguments})
            set_tests_properties(${target} PROPERTIES
                LABELS webgpu
                TIMEOUT 600)
        endif()
    endfunction()
endif()

add_library(ncnntestutil STATIC testutil.cpp)
target_link_libraries(ncnntestutil PUBLIC ncnn)

macro(ncnn_add_test name)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE ncnntestutil ncnn)

    if(NCNN_WEBGPU)
        if("${name}" STREQUAL "command" OR "${name}" STREQUAL "squeezenet")
            ncnn_add_webgpu_browser_test(test_${name} REQUIRE_WEBGPU)
        endif()
    else()
        add_test(NAME test_${name} COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:test_${name}> -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/run_test.cmake)
    endif()

    # add test to a virtual project group
    set_property(TARGET test_${name} PROPERTY FOLDER "tests")
endmacro()

macro(ncnn_add_layer_test class)
    string(TOLOWER ${class} name)

    # enable if WITH_LAYER_xxx option ON
    if(${WITH_LAYER_${name}})
        file(GLOB test_${name}_SRCS "test_${name}.cpp" "test_${name}_*.cpp" LIST_DIRECTORIES FALSE)

        foreach(test_file ${test_${name}_SRCS})
            get_filename_component(test_filename ${test_file} NAME_WE)
            add_executable(${test_filename} ${test_file})
            target_link_libraries(${test_filename} PRIVATE ncnntestutil ncnn)

            if(NCNN_WEBGPU)
                ncnn_add_webgpu_browser_test(${test_filename})
            else()
                add_test(NAME ${test_filename} COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:${test_filename}> -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/run_test.cmake)
            endif()

            # add test to a virtual project group
            set_property(TARGET ${test_filename} PROPERTY FOLDER "tests")
        endforeach()
    endif()
endmacro()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/layer)

if(NCNN_WEBGPU)
    add_executable(webgpu_header_abi_probe webgpu/header_abi_probe.cpp)
    target_link_libraries(webgpu_header_abi_probe PRIVATE ncnn)
    set_property(TARGET webgpu_header_abi_probe PROPERTY FOLDER "tests/webgpu")

    if(NOT NCNN_TEST_CHROME)
        message(STATUS "ncnn WebGPU browser tests are not registered; set NCNN_TEST_CHROME to the frozen Chrome executable")
    endif()
endif()

if(NCNN_PIXEL_AFFINE)
    ncnn_add_test(mat_pixel_affine)
endif()

if(NCNN_PIXEL_DRAWING)
    ncnn_add_test(mat_pixel_drawing)
endif()

if(NCNN_PIXEL_ROTATE)
    ncnn_add_test(mat_pixel_rotate)
endif()

if(NCNN_PIXEL)
    ncnn_add_test(mat_pixel_resize)
    ncnn_add_test(mat_pixel)
    ncnn_add_test(squeezenet)
endif()

ncnn_add_test(c_api)
ncnn_add_test(cpu)
ncnn_add_test(expression)
ncnn_add_test(paramdict)
if(NCNN_BATCH)
    ncnn_add_test(mat_batch)
endif()

if(NCNN_VULKAN)
    ncnn_add_test(command)
    ncnn_add_test(pipeline_cache)
endif()

if(NCNN_WEBGPU)
    ncnn_add_test(command)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT NCNN_WEBGPU AND TARGET test_squeezenet)
    target_link_libraries(test_squeezenet PRIVATE nodefs.js)
endif()

if(NCNN_WEBGPU AND TARGET test_squeezenet)
    set_property(TARGET test_squeezenet APPEND PROPERTY
        LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/../examples/squeezenet_v1.1.param"
            "${CMAKE_CURRENT_SOURCE_DIR}/../examples/squeezenet_v1.1.param.bin"
            "${CMAKE_CURRENT_SOURCE_DIR}/../examples/squeezenet_v1.1.bin")
    target_link_options(test_squeezenet PRIVATE
        "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/../examples/squeezenet_v1.1.param@/working/squeezenet_v1.1.param"
        "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/../examples/squeezenet_v1.1.param.bin@/working/squeezenet_v1.1.param.bin"
        "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/../examples/squeezenet_v1.1.bin@/working/squeezenet_v1.1.bin")
endif()

ncnn_add_layer_test(AbsVal)
ncnn_add_layer_test(BatchNorm)
ncnn_add_layer_test(Bias)
ncnn_add_layer_test(BinaryOp)
ncnn_add_layer_test(BNLL)
ncnn_add_layer_test(Cast)
ncnn_add_layer_test(CELU)
ncnn_add_layer_test(Clip)
ncnn_add_layer_test(Concat)
ncnn_add_layer_test(Convolution)
ncnn_add_layer_test(Convolution1D)
ncnn_add_layer_test(Convolution3D)
ncnn_add_layer_test(ConvolutionDepthWise)
ncnn_add_layer_test(ConvolutionDepthWise1D)
ncnn_add_layer_test(ConvolutionDepthWise3D)
ncnn_add_layer_test(CopyTo)
ncnn_add_layer_test(Crop)
ncnn_add_layer_test(CumulativeSum)
ncnn_add_layer_test(Deconvolution)
ncnn_add_layer_test(Deconvolution1D)
ncnn_add_layer_test(Deconvolution3D)
ncnn_add_layer_test(DeconvolutionDepthWise)
ncnn_add_layer_test(DeconvolutionDepthWise1D)
ncnn_add_layer_test(DeconvolutionDepthWise3D)
ncnn_add_layer_test(DeepCopy)
ncnn_add_layer_test(DeformableConv2D)
ncnn_add_layer_test(Dequantize)
ncnn_add_layer_test(Diag)
ncnn_add_layer_test(Dropout)
ncnn_add_layer_test(Einsum)
ncnn_add_layer_test(Eltwise)
ncnn_add_layer_test(ELU)
ncnn_add_layer_test(Embed)
ncnn_add_layer_test(Erf)
ncnn_add_layer_test(Exp)
ncnn_add_layer_test(ExpandDims)
ncnn_add_layer_test(Flatten)
ncnn_add_layer_test(Flip)
ncnn_add_layer_test(Fold)
ncnn_add_layer_test(GELU)
ncnn_add_layer_test(GLU)
ncnn_add_layer_test(Gemm)
ncnn_add_layer_test(GridSample)
ncnn_add_layer_test(GroupNorm)
ncnn_add_layer_test(GRU)
ncnn_add_layer_test(HardSigmoid)
ncnn_add_layer_test(HardSwish)
ncnn_add_layer_test(InnerProduct)
ncnn_add_layer_test(Input)
ncnn_add_layer_test(InstanceNorm)
ncnn_add_layer_test(Interp)
ncnn_add_layer_test(InverseSpectrogram)
ncnn_add_layer_test(LayerNorm)
ncnn_add_layer_test(Log)
ncnn_add_layer_test(LRN)
ncnn_add_layer_test(LSTM)
ncnn_add_layer_test(MatMul)
ncnn_add_layer_test(MemoryData)
ncnn_add_layer_test(Mish)
ncnn_add_layer_test(MultiHeadAttention)
ncnn_add_layer_test(MVN)
ncnn_add_layer_test(Noop)
ncnn_add_layer_test(Normalize)
ncnn_add_layer_test(Packing)
ncnn_add_layer_test(Padding)
ncnn_add_layer_test(Permute)
ncnn_add_layer_test(PixelShuffle)
ncnn_add_layer_test(Pooling)
ncnn_add_layer_test(Pooling1D)
ncnn_add_layer_test(Pooling3D)
ncnn_add_layer_test(Power)
ncnn_add_layer_test(PReLU)
ncnn_add_layer_test(PriorBox)
ncnn_add_layer_test(Quantize)
ncnn_add_layer_test(Reduction)
ncnn_add_layer_test(ReLU)
ncnn_add_layer_test(Reorg)
ncnn_add_layer_test(Requantize)
ncnn_add_layer_test(Reshape)
ncnn_add_layer_test(RMSNorm)
ncnn_add_layer_test(RNN)
ncnn_add_layer_test(ROIPooling)
ncnn_add_layer_test(ROIAlign)
ncnn_add_layer_test(RotaryEmbed)
ncnn_add_layer_test(Scale)
ncnn_add_layer_test(SDPA)
ncnn_add_layer_test(SELU)
ncnn_add_layer_test(Shrink)
ncnn_add_layer_test(ShuffleChannel)
ncnn_add_layer_test(Sigmoid)
ncnn_add_layer_test(Slice)
ncnn_add_layer_test(Softmax)
ncnn_add_layer_test(Split)
ncnn_add_layer_test(Softplus)
ncnn_add_layer_test(Spectrogram)
ncnn_add_layer_test(StatisticsPooling)
ncnn_add_layer_test(Squeeze)
ncnn_add_layer_test(Swish)
ncnn_add_layer_test(TanH)
ncnn_add_layer_test(Threshold)
ncnn_add_layer_test(Tile)
ncnn_add_layer_test(UnaryOp)
ncnn_add_layer_test(Unfold)
ncnn_add_layer_test(Yolov3DetectionOutput)
