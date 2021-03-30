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

## Build
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

## Install
```bash
cd /pathto/ncnn/python
pip install .
```

if you use conda or miniconda, you can also install as following:
```bash
cd /pathto/ncnn/python
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
