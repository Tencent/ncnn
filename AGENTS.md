# ncnn - AI Agent Developer Guide

ncnn is a high-performance neural network inference framework optimized for mobile and embedded platforms (Tencent, BSD-3-Clause). Written in C/C++ with minimal dependencies. Supports x86, ARM, RISC-V, LoongArch, MIPS CPUs and Vulkan GPU. Includes PNNX for PyTorch/ONNX-to-ncnn conversion.

## Repository Layout

```
src/                    Core library (mat.h, net.h, layer.h, option.h, ...)
src/layer/              Generic layer implementations
src/layer/{x86,arm,riscv,loongarch,mips}/   Arch-optimized layers
src/layer/vulkan/       Vulkan GPU layers + shader/ (.comp GLSL shaders)
tools/pnnx/             PyTorch Neural Network eXchange converter
tools/{caffe,onnx}/     Legacy model converters
tests/                  Unit tests (test_<layername>.cpp)
cmake/                  Build modules (ncnn_add_layer.cmake)
toolchains/             Cross-compilation toolchain files
docs/                   Documentation
.clang-format           Code formatting (Allman, 4-space, C++03)
.github/workflows/      CI (build, test, coverage, format)
```

## Agent Documentation Index

Read these docs selectively based on the task at hand:

| Topic | Doc | When to read |
|---|---|---|
| Key data structures | [docs/agents/data-structures.md](docs/agents/data-structures.md) | Working with Mat, Layer, Net, Blob, ParamDict |
| Build and test | [docs/agents/build-and-test.md](docs/agents/build-and-test.md) | Building, testing, cross-compilation, coverage |
| Code style and portability | [docs/agents/code-style.md](docs/agents/code-style.md) | Writing code for src/ (C++03, simplestl, OpenMP rules) |
| CPU/GPU dispatch | [docs/agents/dispatch.md](docs/agents/dispatch.md) | Understanding layer registration, packing, Vulkan flow |
| PNNX architecture | [docs/agents/pnnx.md](docs/agents/pnnx.md) | Model conversion pipeline, IR, pass system |
| Task: Add ncnn operator | [docs/agents/task-add-operator.md](docs/agents/task-add-operator.md) | Adding a new layer to ncnn |
| Task: Add PNNX operator | [docs/agents/task-add-pnnx-operator.md](docs/agents/task-add-pnnx-operator.md) | Adding PyTorch op support to PNNX |
| Task: x86 SIMD optimization | [docs/agents/task-x86-optimization.md](docs/agents/task-x86-optimization.md) | SSE/AVX/AVX-512 layer optimization |
| Task: Vulkan optimization | [docs/agents/task-vulkan-optimization.md](docs/agents/task-vulkan-optimization.md) | GPU compute shader layer |
| Task: Cross-arch optimization | [docs/agents/task-cross-arch-optimization.md](docs/agents/task-cross-arch-optimization.md) | ARM NEON/SVE, RISC-V RVV, QEMU testing |

## Existing Project Documentation

- `docs/developer-guide/operation-param-weight-table.md` — all operator param/weight definitions
- `docs/developer-guide/param-and-model-file-structure.md` — .param and .bin file format
- `docs/developer-guide/element-packing.md` — element packing design
- `docs/developer-guide/how-to-implement-custom-layer-step-by-step.md` — custom layer guide
- `docs/developer-guide/glsl-extension.md` — ncnn Vulkan GLSL extensions
- `docs/developer-guide/layer-support-behavior.md` — layer support flags behavior
- `docs/how-to-build/how-to-build.md` — build instructions for all platforms
