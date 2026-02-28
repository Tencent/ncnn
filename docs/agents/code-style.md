# Code Style and Portability

## Formatting

The project uses **Allman brace style** with 4-space indentation, no tabs. Defined in `.clang-format` and `.astylerc`.

Key conventions:
- **Indentation**: 4 spaces, no tabs
- **Braces**: Allman style (opening brace on new line for functions, classes, control statements)
- **Namespaces**: No indentation inside `namespace ncnn { ... }`
- **Pointers**: Left-aligned (`float* ptr`, not `float *ptr`)
- **Column limit**: None (no line length limit)
- **Includes**: Not sorted by clang-format
- **Naming**: `snake_case` for variables/functions, `PascalCase` for class names, `UPPER_CASE` for macros
- **Comments**: `//` style, minimal — code is expected to be self-explanatory
- **Copyright header**: Every file starts with `// Copyright YYYY Tencent` and `// SPDX-License-Identifier: BSD-3-Clause`
- **SIMD code**: Uses `#if __SSE2__` / `#if __AVX__` / `#if __ARM_NEON` preprocessor guards, nested from wider to narrower
- **OpenMP**: `#pragma omp parallel for num_threads(opt.num_threads)` on the outer channel loop

Format code with:
```bash
./codeformat.sh  # runs clang-format + astyle twice for stable output
```

You do **not** need to run this locally before submitting. The GitHub CI workflow (`.github/workflows/code-format.yml`) automatically formats all C/C++ source files and GLSL shaders on every push/PR and commits the formatting changes back. Just write code following the conventions above, and CI will fix any minor formatting deviations.

## Code Portability (Core Library)

ncnn's core library (`src/`) is designed for maximum compiler and platform compatibility. Strict portability rules apply to all code under `src/`:

### Language Standard

- **C code**: C99
- **C++ code**: C++03 (`.clang-format` enforces `Standard: c++03`)
- **Do NOT use** C++11 or later features in `src/`: no `auto`, `nullptr`, range-based for loops, `constexpr`, `std::move`, lambda expressions, `override`/`final` keywords, uniform initialization `{}`, `<thread>`, `<mutex>`, `<atomic>`, etc.
- Use `0` instead of `nullptr`, explicit type declarations instead of `auto`, traditional for loops instead of range-for.

### STL Restrictions

ncnn provides its own minimal STL implementation in `src/simplestl.h` (enabled with `NCNN_SIMPLESTL=ON`) to support environments without a C++ standard library (bare-metal, some embedded systems). All core library code must be compatible with this subset:

- **Allowed**: `std::vector`, `std::string`, `std::pair`, `std::list`, `std::stack`, `std::swap`, `std::min`, `std::max`, `std::partial_sort`, `std::less`, `std::greater`
- **Not available in simplestl**: `std::map`, `std::set`, `std::unordered_map`, `std::shared_ptr`, `std::unique_ptr`, `<algorithm>` (beyond `partial_sort`), `<functional>`, `<iostream>`, streams, smart pointers, etc.
- When writing core library code, only use STL templates that are implemented in `simplestl.h`.

### Math Restrictions

ncnn also provides `src/simplemath.h` / `src/simplemath.cpp` (enabled with `NCNN_SIMPLEMATH=ON`) as a drop-in replacement for `<math.h>` / `<cmath>`, for platforms without a math library. Core code should stick to standard C99 math functions.

### OpenMP Restrictions

ncnn provides a minimal OpenMP runtime (`src/simpleomp.h` / `src/simpleomp.cpp`, enabled with `NCNN_SIMPLEOMP=ON`) that supports both the LLVM libomp ABI and the GCC libgomp ABI. Only the following OpenMP usage is allowed in the core library:

```cpp
#pragma omp parallel for num_threads(opt.num_threads)
```

Do not use any other OpenMP directives such as `critical`, `atomic`, `reduction`, `task`, `simd`, `sections`, or `barrier`. The `collapse(2)` clause is used in a few places but should be limited to simple cases.

### Tools and PNNX — No Restriction

Code outside the core library — specifically `tools/pnnx/`, `tools/caffe/`, `tools/onnx/`, `examples/`, `tests/`, `python/` — is **not** subject to these portability restrictions. PNNX in particular uses **C++17** (or C++14 for PyTorch < 2.1) and freely uses modern C++ features, the full standard library, protobuf, etc.
