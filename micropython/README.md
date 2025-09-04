# ncnn_mp

[中文](./README_zh.md)

**ncnn_mp** is an external C module for MicroPython that brings Tencent's [ncnn](https://github.com/Tencent/ncnn) high-performance neural network inference framework to resource-constrained microcontrollers. It provides **ncnn C API** bindings, allowing you to run AI inference directly on embedded devices like ESP32, STM32 and more.

Although there exist Python bindings for `ncnn`, `ncnn_mp` enables developers to use `ncnn` features with an object-oriented and Pythonic API on embedded platforms.

## Features

- **Pythonic Object-Oriented API**: ncnn's C-style handles are transformed into Python objects (`Allocator`, `Option`, `Mat`, `Blob`, `ParamDict`, `DataReader`, `ModelBin`, `Layer`, `Net`, and `Extractor`).
- **Designed for MCUs**: Use a wrapper around the **ncnn C API**, making it ideal for devices with limited memory and processing power.
- **Great Developer Experience**: Includes a `.pyi` stub file, providing full **autocompletion**, **type hints**, and **inline documentation** in modern IDEs.
- **Cross-Platform**: Easily buildable for various MicroPython ports, with examples for Unix and ESP32-S3.

---

## Build

### Dependencies

- [ncnn](https://github.com/Tencent/ncnn)
- [MicroPython](https://github.com/micropython/micropython)
- CMake / GNU Make

### Get Ready

```bash
git submodule update --init --recursive
```

### Build for Linux (Unix Port, Make)

1.  **Build ncnn**

```bash
cd ncnn
mkdir build && cd build
# Example: a relatively feature-rich configuration
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=ON -DNCNN_VULKAN=ON -DNCNN_BUILD_BENCHMARK=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=./install ..
make -j4
make install
```

2.  **Build MicroPython Firmware**

```bash
cd micropython/ports/unix/
make -C ../../mpy-cross -j4
make submodules -j4
make USER_C_MODULES=../../../modules USE_VULKAN=1 -j4
./build-standard/micropython ../../../examples/main.py
```

> **For Debugging**: Change `CMAKE_BUILD_TYPE` to `DEBUG` in the ncnn build. Then, build MicroPython with `make USER_C_MODULES=../../../modules USE_VULKAN=1 NCNN_INSTALL_PREFIX=../../../ncnn/build-debug/install DEBUG=1 COPT=-O0 -j4`.

### Build for ESP32-S3 (Cross-compilation, CMake)

This example assumes you are building on a Linux host for an ESP32-S3 target.

0.  **Prepare the ESP-IDF Toolchain**

You can clone this repository wherever you want. When using `esp-idf` sdk, you must navigate into the repository's directory and then run `source ./export.sh` to configure the environment for your current terminal session.

```bash
git clone https://github.com/espressif/esp-idf
cd esp-idf
```

MicroPython requires a specific version of `esp-idf`. As of _2025.08_, the supported versions are `v5.2`, `v5.2.2`, `v5.3`, `v5.4`, `v5.4.1` and `v5.4.2`. Please check the [official MicroPython esp32 port documentation](https://github.com/micropython/micropython/blob/master/ports/esp32/README.md) for the latest version list.

```bash
git checkout v5.4.2
git submodule update --init --recursive
sh ./install.sh esp32s3
source ./export.sh
```

1.  **Build ncnn for ESP32-S3**

```bash
cd ncnn
mkdir build-esp32s3 && cd build-esp32s3
# You should add configs here to minimize your ncnn lib
cmake -DCMAKE_TOOLCHAIN_FILE=../../toolchains/esp32s3.toolchain.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=./install ..
make -j4
make install
```

2.  **Build MicroPython Firmware**

```bash
cd micropython/ports/esp32
make -C ../../mpy-cross -j4
make submodules BOARD=ESP32_GENERIC_S3 -j4
idf.py -D MICROPY_BOARD=ESP32_GENERIC_S3 -D USER_C_MODULES=../../../../modules/ncnn_mp/micropython.cmake -D NCNN_INSTALL_PREFIX=../../../../ncnn/build-esp32s3/install build
```

3.  **Flash to Device**

```bash
# [Optional]: Erase the entire flash chip
idf.py -p /dev/ttyACM0 erase-flash

# Flash the new firmware
idf.py -p /dev/ttyACM0 flash

# Monitor the device's output
# Same as: tio /dev/ttyACM0
idf.py -p /dev/ttyACM0 monitor
```

---

## References
- MicroPython doc for external C modules: [MicroPython external C modules](https://docs.micropython.org/en/latest/develop/cmodules.html)
- How to build minimal ncnn library: [build minimal library](https://github.com/Tencent/ncnn/wiki/build-minimal-library)
- ESP-IDF Official Guide for ESP32-S3: [Get Started](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/index.html)
- Using USB devices with WSL: [Connect USB devices](https://learn.microsoft.com/en-us/windows/wsl/connect-usb#attach-a-usb-device)