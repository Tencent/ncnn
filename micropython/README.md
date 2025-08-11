# NCNN MicroPython Module

This directory contains the MicroPython bindings for NCNN's C API, allowing you to run neural network inference directly in MicroPython.

## Prerequisites

On Debian, Ubuntu, or Raspberry Pi OS, you can install all required dependencies using:
```shell
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libomp-dev libopencv-dev
```
On Redhat or Centos, you can install all required dependencies using:
```shell
sudo yum install gcc gcc-c++ make git cmake protobuf-devel protobuf-compiler opencv-devel
```

## Build Instructions

### 1. Build NCNN Library

First, build the NCNN library with C API support:

```bash
mkdir -p ncnn/build_micropython
cd ncnn/build_micropython
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_C_API=ON \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_TOOLS=OFF \
      ..
make -j$(nproc)
make install
```

### 2. Build MicroPython with NCNN Module

First, clone the MicroPython repository if you haven't already:

```bash
cd ../..
git clone https://github.com/micropython/micropython.git
cd micropython

# Build mpy-cross first
cd mpy-cross
make -j$(nproc)

# Build MicroPython with NCNN module
cd ../ports/unix
make submodules
make clean
make USER_C_MODULES=/path/to/ncnn/micropython
```

## Usage

TODO