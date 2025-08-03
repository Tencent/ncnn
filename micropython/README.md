# ncnn-mp

**ncnn-mp** 是一个在 Micropython 平台上使用 [ncnn](https://github.com/Tencent/ncnn) 的扩展模块，提供了 **ncnn C API** 的绑定，让你在嵌入式设备（如 ESP32、STM32 等）上直接运行 ncnn 推理。

## 特性
- 基于 **ncnn C API**，轻量封装，方便移植。
- 可在支持 Micropython 的平台直接调用 ncnn 推理。

## 依赖
- [Micropython](https://micropython.org/)
- [ncnn](https://github.com/Tencent/ncnn)
- CMake/GNU Make

## 构建
1. ncnn
```
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_SHARED_LIB=OFF -DNCNN_VULKAN=OFF -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=ON -DNCNN_SIMPLESTL=OFF -DNCNN_RUNTIME_CPU=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_PYTHON=OFF -DNCNN_STDIO=ON -DNCNN_STRING=ON -DNCNN_PIXEL=ON -DNCNN_DISABLE_RTTI=ON -DNCNN_DISABLE_EXCEPTION=ON ..
```
2. mpy
```bash
~/dev/ncnn_mp$ cd micropython/ports/unix/
~/dev/ncnn_mp/micropython/ports/unix$ make -C ../../mpy-cross
~/dev/ncnn_mp/micropython/ports/unix$ make submodules
~/dev/ncnn_mp/micropython/ports/unix$ make USER_C_MODULES=../../../modules
~/dev/ncnn_mp/micropython/ports/unix$ ./build-standard/micropython ../../../examples/main.py
```
