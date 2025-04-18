# ncnn
python wrapper of ncnn with [pybind11](https://github.com/pybind/pybind11), only support python3.x now.


Install from pip
==================

ncnn is available as wheel packages for macOS, Windows and Linux distributions, you can install with pip:

```
python -m pip install -U pip
python -m pip install -U ncnn
```

# Build from source

If you want to build ncnn with some options not as default, or just like to build everything yourself, it is not difficult to build ncnn from source.

## Prerequisites

**On Unix (Linux, OS X)**

* A compiler with C++11 support
* CMake >= 3.4

**On Mac**

* A compiler with C++11 support
* CMake >= 3.4

**On Windows**

* Visual Studio 2015 or higher
* CMake >= 3.4

##  Build & Install

1. clone ncnn and init submodule.

```bash
cd /pathto/ncnn
git submodule init && git submodule update
```

2. build and install.

```
python setup.py install
```

If you want to use a custom toolchain, you can install with the `CMAKE_TOOLCHAIN_FILE` environment variable, like this:

```
CMAKE_TOOLCHAIN_FILE="../../toolchains/power9le-linux-gnu-vsx.clang.toolchain.cmake" python setup.py install
```

if you want to enable the usage of vulkan, you can install as following:

```
python setup.py install --vulkan=on
```

> **Attention:**
>
> To enable Vulkan support, you must first install the Vulkan SDK.
>
> **For Windows or Linux Users:**
>
> Ensure that the `VULKAN_SDK` environment variable is set to the path of the Vulkan SDK.
>
> **For MacOS Users:**
>
> On MacOS, you will need to specify additional environment variables. For guidance on setting these variables, please refer to lines 279-286 in the following file: [ncnn/.github/workflows/release-python.yml at master · Tencent/ncnn](https://github.com/Tencent/ncnn/blob/master/.github/workflows/release-python.yml).

## Custom-build & Install

1. clone ncnn and init submodule.
```bash
cd /pathto/ncnn
git submodule init && git submodule update
```
2. build.
```bash
mkdir build
cd build
cmake -DNCNN_PYTHON=ON ..
make
```

3. install

```bash
cd /pathto/ncnn
pip install .
```

if you use conda or miniconda, you can also install as following:
```bash
cd /pathto/ncnn
python3 setup.py install
```

## Tests

**test**
```bash
cd /pathto/ncnn/python
python3 tests/test.py
```

**benchmark**

```bash
cd /pathto/ncnn/python
python3 tests/benchmark.py
```

## Numpy
**ncnn.Mat->numpy.array, with no memory copy**

```bash
mat = ncnn.Mat(...)
mat_np = np.array(mat)
```

**numpy.array->ncnn.Mat, with no memory copy**
```bash
mat_np = np.array(...)
mat = ncnn.Mat(mat_np)
```

# Model Zoo
install requirements
```bash
pip install -r requirements.txt
```
then you can import ncnn.model_zoo and get model list as follow:
```bash
import ncnn
import ncnn.model_zoo as model_zoo

print(model_zoo.get_model_list())
```
models now in model zoo are as list below:
```bash
mobilenet_yolov2
mobilenetv2_yolov3
yolov4_tiny
yolov4
yolov5s
yolact
mobilenet_ssd
squeezenet_ssd
mobilenetv2_ssdlite
mobilenetv3_ssdlite
squeezenet
faster_rcnn
peleenet_ssd
retinaface
rfcn
shufflenetv2
simplepose
nanodet
```
all model in model zoo has example in ncnn/python/examples folder

# Custom Layer

custom layer demo is in ncnn/python/ncnn/model_zoo/yolov5.py:23
