# Build and Test

## Basic Build (Linux)

```bash
cd ncnn
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

## Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `NCNN_VULKAN` | OFF | Enable Vulkan GPU support |
| `NCNN_OPENMP` | ON | Enable OpenMP multi-threading |
| `NCNN_BUILD_TESTS` | OFF | Build unit tests |
| `NCNN_BUILD_TOOLS` | ON* | Build converter tools |
| `NCNN_BUILD_EXAMPLES` | ON* | Build example programs |
| `NCNN_BUILD_BENCHMARK` | ON | Build benchmark tool |
| `NCNN_SHARED_LIB` | OFF | Build shared library |
| `NCNN_RUNTIME_CPU` | ON | Runtime CPU feature detection & dispatch |
| `NCNN_SSE2` | ON | x86 SSE2 support |
| `NCNN_AVX` | ON | x86 AVX support |
| `NCNN_AVX2` | ON | x86 AVX2/FMA support |
| `NCNN_AVX512` | ON* | x86 AVX-512 support |
| `NCNN_ARM82` | ON | AArch64 fp16 (ARMv8.2) |
| `NCNN_ARM82DOT` | ON | AArch64 dot product |
| `NCNN_ARM84BF16` | ON | AArch64 BFloat16 |
| `NCNN_ARM84I8MM` | ON | AArch64 Int8 matrix multiply |
| `NCNN_ARM86SVE` | ON | AArch64 SVE |
| `NCNN_RVV` | ON | RISC-V Vector extension |
| `NCNN_SIMPLEMATH` | OFF | Use built-in math (no libm) |
| `NCNN_SIMPLESTL` | OFF | Use built-in STL (no libstdc++) |
| `WITH_LAYER_xxx` | ON | Enable/disable individual layers |

\* `NCNN_BUILD_TOOLS` and `NCNN_BUILD_EXAMPLES` default to OFF when cross-compiling or targeting Android/iOS. `NCNN_AVX512` defaults to ON only when the compiler supports it and `NCNN_AVX2` is ON.

## Build with Vulkan

```bash
cmake -DNCNN_VULKAN=ON ..
cmake --build . -j$(nproc)
```

Requires the Vulkan SDK. The bundled `glslang/` submodule compiles GLSL shaders to SPIR-V at build time.

## Build with Tests

```bash
cmake -DNCNN_BUILD_TESTS=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF ..
cmake --build . -j$(nproc)
ctest --output-on-failure -j$(nproc)
```

## Cross-Compilation

Toolchain files are in `toolchains/`. Example for AArch64:

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
      -DNCNN_BUILD_TESTS=ON ..
cmake --build . -j$(nproc)
```

Run tests with QEMU:

```bash
TESTS_EXECUTABLE_LOADER=qemu-aarch64-static \
TESTS_EXECUTABLE_LOADER_ARGUMENTS="-L;/usr/aarch64-linux-gnu" \
ctest --output-on-failure -j8
```

For RISC-V with RVV:

```bash
export RISCV_ROOT_PATH=/path/to/riscv-toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-unknown-linux-gnu.toolchain.cmake \
      -DNCNN_RVV=ON -DNCNN_BUILD_TESTS=ON ..
cmake --build . -j$(nproc)

# Test with QEMU (vlen=256)
TESTS_EXECUTABLE_LOADER=qemu-riscv64 \
TESTS_EXECUTABLE_LOADER_ARGUMENTS="-cpu;rv64,v=true,zfh=true,zvfh=true,vlen=256,elen=64,vext_spec=v1.0;-L;/path/to/sysroot" \
ctest --output-on-failure -j8
```

## Intel SDE for x86 ISA Testing

The CI uses Intel SDE to test advanced ISA extensions (AVX-512, AVX-VNNI, etc.) on machines that do not natively support them:

```bash
TESTS_EXECUTABLE_LOADER=/path/to/sde64 \
TESTS_EXECUTABLE_LOADER_ARGUMENTS="-spr;--" \
ctest --output-on-failure -j8
```

## Testing

Tests are in `tests/`. Each layer has a `test_<layername>.cpp` file.

### Test Pattern

Tests use `testutil.h` which provides `test_layer()` — it creates a layer with given `ParamDict` and weights, runs forward with random input using the naive (generic, non-optimized) layer implementation, then runs the same input through the CPU-optimized and Vulkan paths (when available), and compares the results with numerical tolerance checks.

```cpp
// tests/test_relu.cpp
#include "testutil.h"

static int test_relu(const ncnn::Mat& a, float slope)
{
    ncnn::ParamDict pd;
    pd.set(0, slope);
    std::vector<ncnn::Mat> weights(0);
    int ret = test_layer("ReLU", pd, weights, a);
    if (ret != 0)
        fprintf(stderr, "test_relu failed a.dims=%d a=(%d %d %d %d) slope=%f\n",
                a.dims, a.w, a.h, a.d, a.c, slope);
    return ret;
}

int main()
{
    SRAND(7767517);
    return test_relu(RandomMat(5, 6, 7, 24), 0.f)
        || test_relu(RandomMat(128), 0.1f);
}
```

### Adding a New Test

1. Create `tests/test_<layername>.cpp`
2. Add to `tests/CMakeLists.txt`: `ncnn_add_test(test_<layername>)`
3. Test all dimension ranks (1D, 2D, 3D, 4D) with various sizes, including:
   - Sizes divisible by common pack sizes (4, 8, 16)
   - Non-aligned sizes to test remainder loops
   - Multiple parameter combinations

## Code Coverage

CI runs code coverage on every push/PR (see `.github/workflows/test-coverage.yml`). It builds with `NCNN_COVERAGE=ON` which adds `-coverage -fprofile-arcs -ftest-coverage` flags and links `-lgcov`. After tests run, `lcov` collects the `.gcda` / `.gcno` data and uploads to Codecov.

When developing, you should measure coverage locally to ensure your new code is well tested:

```bash
# Build with coverage
mkdir build-coverage && cd build-coverage
cmake -DCMAKE_BUILD_TYPE=debug \
      -DNCNN_COVERAGE=ON \
      -DNCNN_RUNTIME_CPU=OFF \
      -DNCNN_OPENMP=OFF \
      -DNCNN_BUILD_TOOLS=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_TESTS=ON ..
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure -j$(nproc)

# Collect coverage
lcov -d ./src -c -o lcov.info
lcov -r lcov.info '/usr/*' -o lcov.info
lcov -r lcov.info '*/build-coverage/*' -o lcov.info
lcov --list lcov.info

# (Optional) Generate HTML report
genhtml lcov.info --output-directory coverage-html
# Open coverage-html/index.html in a browser
```

Aim for high coverage of your new or modified code paths. The CI coverage matrix tests multiple configurations — x86 ISA variants (none/sse2/avx/avx2/avx512/avx512vnni), cross-compiled architectures (ARM, RISC-V RVV, MIPS, LoongArch, PowerPC) via QEMU, Vulkan GPU (llvmpipe and SwiftShader), and OpenMP on/off — so make sure your tests exercise the relevant branches.
