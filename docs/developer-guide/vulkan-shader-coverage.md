# Vulkan shader line coverage

`NCNN_COVERAGE=ON` enables C/C++ coverage and, when `NCNN_VULKAN=ON`, line coverage for builtin Vulkan shaders. It requires stdio and the standard C++ library. Coverage builds retain the original shader comments and blank lines; normal builds still compact embedded shader sources.

```sh
cmake -S . -B build-coverage -DNCNN_VULKAN=ON -DNCNN_COVERAGE=ON -DNCNN_BUILD_TESTS=ON
cmake --build build-coverage -j 8
mkdir -p "$PWD/build-coverage/shader-coverage"
NCNN_SHADER_COVERAGE_DIR="$PWD/build-coverage/shader-coverage" ctest --test-dir build-coverage --output-on-failure -j 8
lcov --capture --directory build-coverage --output-file build-coverage/cpp.info
set -- -a build-coverage/cpp.info
for report in build-coverage/shader-coverage/shader-coverage-*.info; do
    set -- "$@" -a "$report"
done
lcov "$@" -o build-coverage/coverage.info
genhtml build-coverage/coverage.info -o build-coverage/html
```

The output directory must already exist. Without `NCNN_SHADER_COVERAGE_DIR`, reports are written to the current directory. A report is written when a `VulkanDevice` is destroyed (including destruction by `destroy_gpu_instance()` and normal process shutdown). Aborted processes cannot flush their coverage. Each report uses an exclusively created `shader-coverage-<pid>-<serial>.info` LCOV file, so parallel tests, multiple devices and repeated device lifetimes do not overwrite each other. Merge only after the test processes have finished. Processes without executable builtin shader lines do not write a report. Use an empty report directory for a new run and merge reports from the same source revision. No conversion tool or intermediate report format is needed.

CMake generates an internal source table, allocating consecutive bits for the original lines of each builtin source. All precision, packing and device variants share this mapping. Reports use absolute source paths from the checkout used at build time, independently of the test process's working directory.

glslang emits `OpLine` locations with explicit source names and line directives for the split source, generated options, shared GLSL definitions and activation include. The instrumenter inserts `OpAtomicOr` probes for each mapped line in each basic block. It preserves Phi/variable placement and merge/branch adjacency, and updates the entry-point interface for SPIR-V 1.4 and newer. The coverage SSBO uses descriptor set 1, binding 0; normal ncnn resources remain in set 0.

Each device owns a persistently mapped buffer. Dispatches atomically accumulate hit bits, and command buffers include a shader-write to host-read barrier. Destruction waits for the device, invalidates mapped memory and writes LCOV records using the executable-line mask and hit bits. Shader caches are separated from normal builds and include the source-map hash; cached probes also restore executable lines.

Each report contains `DA:<line>,0` or `DA:<line>,1`, along with per-source `LF` and `LH` totals. Merging with `lcov -a` adds the counts, so a merged value can exceed 1: it counts reports that hit the line, not shader executions. Line coverage still uses zero versus nonzero counts. Only lines with emitted probes in compiled variants are counted. Preprocessor-excluded code, shaders never compiled during the run and custom shaders supplied as strings or precompiled SPIR-V do not contribute to the denominator. Compile the variants you want measured during the test run, including pipelines that are not dispatched, to include their unexecuted lines. Lines in the shared GLSL sources are combined across callers. There is no branch coverage (`BRDA`) in this implementation.

Instrumentation adds atomics to shader execution and is intended for coverage runs, not performance measurement. It does not require `debugPrintfEXT` or a Vulkan validation layer.

Design references: [Slang shader coverage](https://github.com/shader-slang/slang/blob/master/docs/design/shader-coverage.md) for the source mapping and LCOV workflow, and [Vulkan GPU-assisted validation](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/gpu_validation.md) for SPIR-V instrumentation using dedicated descriptor resources.
