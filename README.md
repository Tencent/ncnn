![NCNN](https://raw.githubusercontent.com/Tencent/ncnn/master/images/256-ncnn.png)

# ncnn

[![License][badge-license]](LICENSE.txt)
[![Download Total Count][badge-download-count]](https://github.com/Tencent/ncnn/releases)
[![codecov][badge-codecov]](https://codecov.io/gh/Tencent/ncnn)
[![Language grade: C/C++][badge-code-quality]](https://lgtm.com/projects/g/Tencent/ncnn/context:cpp)

[badge-license]: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
[badge-download-count]: https://img.shields.io/github/downloads/Tencent/ncnn/total.svg
[badge-codecov]: https://codecov.io/gh/Tencent/ncnn/branch/master/graph/badge.svg
[badge-code-quality]: https://img.shields.io/lgtm/grade/cpp/g/Tencent/ncnn.svg?logo=lgtm&logoWidth=18

ncnn is a high-performance neural network inference computing framework optimized for mobile platforms.
ncnn is deeply considerate about deployment and uses on mobile phones from the beginning of design.
ncnn does not have third party dependencies. It is cross-platform, and runs faster than all known open source frameworks on mobile phone cpu.
Developers can easily deploy deep learning algorithm models to the mobile platform by using efficient ncnn implementation,
create intelligent APPs, and bring the artificial intelligence to your fingertips.
ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu and so on.

ncnn 是一个为手机端极致优化的高性能神经网络前向计算框架。
ncnn 从设计之初深刻考虑手机端的部署和使用。
无第三方依赖，跨平台，手机端 cpu 的速度快于目前所有已知的开源框架。
基于 ncnn，开发者能够将深度学习算法轻松移植到手机端高效执行，
开发出人工智能 APP，将 AI 带到你的指尖。
ncnn 目前已在腾讯多款应用中使用，如：QQ，Qzone，微信，天天 P 图等。

---

## 技术交流 QQ 群：637093648 (超多大佬) 答案：卷卷卷卷卷 （已满）

## Pocky QQ 群（MLIR YES!）: 677104663(超多大佬) 答案：multi-level intermediate representation

## Telegram Group <https://t.me/ncnnyes>

## Discord Channel <https://discord.gg/YRsxgmF>

---

## Current building status matrix

| System            | CPU (32bit)                                                         | CPU (64bit)                                                                     | GPU (32bit)                                                     | GPU (64bit)                                                         |
| :---------------- | :------------------------------------------------------------------ | :------------------------------------------------------------------------------ | :-------------------------------------------------------------- | :------------------------------------------------------------------ |
| Linux (GCC)       | [![Build Status][pass-linux-x86-cpu-gcc]][ci-linux-x86-cpu-gcc]     | [![Build Status][pass-linux-x64-cpu-gcc]][ci-linux-x64-cpu-gcc]                 | —                                                               | [![Build Status][pass-linux-x64-gpu-gcc]][ci-linux-x64-gpu-gcc]     |
| Linux (Clang)     | [![Build Status][pass-linux-x86-cpu-clang]][ci-linux-x86-cpu-clang] | [![Build Status][pass-linux-x64-cpu-clang]][ci-linux-x64-cpu-clang]             | —                                                               | [![Build Status][pass-linux-x64-gpu-clang]][ci-linux-x64-gpu-clang] |
| Linux (ARM)       | [![Build Status][pass-linux-arm-cpu-gcc]][ci-linux-arm-cpu-gcc]     | [![Build Status][pass-linux-aarch64-cpu-gcc]][ci-linux-aarch64-cpu-gcc]         | —                                                               | —                                                                   |
| Linux (MIPS)      | [![Build Status][pass-linux-mips-cpu-gcc]][ci-linux-mips-cpu-gcc]   | [![Build Status][pass-linux-mips64-cpu-gcc]][ci-linux-mips64-cpu-gcc]           | —                                                               | —                                                                   |
| Linux (RISC-V)    | —                                                                   | [![Build Status][pass-linux-riscv64-cpu-gcc]][ci-linux-riscv64-cpu-gcc]         | —                                                               | —                                                                   |
| Linux (LoongArch) | —                                                                   | [![Build Status][pass-linux-loongarch64-cpu-gcc]][ci-linux-loongarch64-cpu-gcc] | —                                                               | —                                                                   |
| Windows           | [![Build Status][pass-windows-x86-cpu]][ci-windows-x86-cpu]         | [![Build Status][pass-windows-x64-cpu]][ci-windows-x64-cpu]                     | —                                                               | [![Build Status][pass-windows-x64-gpu]][ci-windows-x64-gpu]         |
| Windows (ARM)     | [![Build Status][pass-windows-arm-cpu]][ci-windows-arm-cpu]         | [![Build Status][pass-windows-arm64-cpu]][ci-windows-arm64-cpu]                 | —                                                               | —                                                                   |
| macOS             | —                                                                   | [![Build Status][pass-macos-x64-cpu]][ci-macos-x64-cpu]                         | —                                                               | [![Build Status][pass-macos-x64-gpu]][ci-macos-x64-gpu]             |
| macOS (ARM)       | —                                                                   | [![Build Status][pass-macos-arm64-cpu]][ci-macos-arm64-cpu]                     | —                                                               | [![Build Status][pass-macos-arm64-gpu]][ci-macos-arm64-gpu]         |
| Android           | [![Build Status][pass-android-armv7-cpu]][ci-android-armv7-cpu]     | [![Build Status][pass-android-armv8-cpu]][ci-android-armv8-cpu]                 | [![Build Status][pass-android-armv7-gpu]][ci-android-armv7-gpu] | [![Build Status][pass-android-armv8-gpu]][ci-android-armv8-gpu]     |
| Android-x86       | [![Build Status][pass-android-x86-cpu]][ci-android-x86-cpu]         | [![Build Status][pass-android-x64-cpu]][ci-android-x64-cpu]                     | [![Build Status][pass-android-x86-gpu]][ci-android-x86-gpu]     | [![Build Status][pass-android-x64-gpu]][ci-android-x64-gpu]         |
| iOS               | [![Build Status][pass-ios-cpu]][ci-ios-cpu]                         | [![Build Status][pass-ios-cpu]][ci-ios-cpu]                                     | —                                                               | [![Build Status][pass-ios-arm64-gpu]][ci-ios-arm64-gpu]             |
| iOS Simulator     | [![Build Status][pass-ios-simulator]][ci-ios-simulator]             | [![Build Status][pass-ios-simulator]][ci-ios-simulator]                         | —                                                               | [![Build Status][pass-ios-simulator-gpu]][ci-ios-simulator-gpu]     |
| WebAssembly       | [![Build Status][pass-web-assembly]][ci-web-assembly]               | —                                                                               | —                                                               | —                                                                   |
| RISC-V GCC/Newlib | [![Build Status][pass-elf-riscv32-cpu-gcc]][ci-elf-riscv32-cpu-gcc] | [![Build Status][pass-elf-riscv64-cpu-gcc]][ci-elf-riscv64-cpu-gcc]             | —                                                               | —                                                                   |

[pass-android-armv7-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/android-armv7-cpu.yml?branch=master
[pass-android-armv7-gpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/android-armv7-gpu.yml?branch=master
[pass-android-armv8-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/android-armv8-cpu.yml?branch=master
[pass-android-armv8-gpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/android-armv8-gpu.yml?branch=master
[pass-android-x64-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/android-x64-cpu.yml?branch=master
[pass-android-x64-gpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/android-x64-gpu.yml?branch=master
[pass-android-x86-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/android-x86-cpu.yml?branch=master
[pass-android-x86-gpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/android-x86-gpu.yml?branch=master
[pass-elf-riscv32-cpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/elf-riscv32-cpu-gcc.yml?branch=master
[pass-elf-riscv64-cpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/elf-riscv64-cpu-gcc.yml?branch=master
[pass-ios-arm64-gpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/ios-arm64-gpu.yml?branch=master
[pass-ios-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/ios-cpu.yml?branch=master
[pass-ios-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/ios-cpu.yml?branch=master
[pass-ios-simulator]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/ios-simulator.yml?branch=master
[pass-ios-simulator]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/ios-simulator.yml?branch=master
[pass-ios-simulator-gpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/ios-simulator-gpu.yml?branch=master
[pass-linux-aarch64-cpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-aarch64-cpu-gcc.yml?branch=master
[pass-linux-arm-cpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-arm-cpu-gcc.yml?branch=master
[pass-linux-loongarch64-cpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-loongarch64-cpu-gcc.yml?branch=master
[pass-linux-mips-cpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-mips-cpu-gcc.yml?branch=master
[pass-linux-mips64-cpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-mips64-cpu-gcc.yml?branch=master
[pass-linux-riscv64-cpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-riscv64-cpu-gcc.yml?branch=master
[pass-linux-x64-cpu-clang]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-x64-cpu-clang.yml?branch=master
[pass-linux-x64-cpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-x64-cpu-gcc.yml?branch=master
[pass-linux-x64-gpu-clang]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-x64-gpu-clang.yml?branch=master
[pass-linux-x64-gpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-x64-gpu-gcc.yml?branch=master
[pass-linux-x86-cpu-clang]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-x86-cpu-clang.yml?branch=master
[pass-linux-x86-cpu-gcc]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-x86-cpu-gcc.yml?branch=master
[pass-macos-arm64-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/macos-arm64-cpu.yml?branch=master
[pass-macos-arm64-gpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/macos-arm64-gpu.yml?branch=master
[pass-macos-x64-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/macos-x64-cpu.yml?branch=master
[pass-macos-x64-gpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/macos-x64-gpu.yml?branch=master
[pass-web-assembly]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/web-assembly.yml?branch=master
[pass-windows-arm-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/windows-arm-cpu.yml?branch=master
[pass-windows-arm64-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/windows-arm64-cpu.yml?branch=master
[pass-windows-x64-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/windows-x64-cpu.yml?branch=master
[pass-windows-x64-gpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/windows-x64-gpu.yml?branch=master
[pass-windows-x86-cpu]: https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/windows-x86-cpu.yml?branch=master
[ci-android-armv7-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aandroid-armv7-cpu
[ci-android-armv7-gpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aandroid-armv7-gpu
[ci-android-armv8-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aandroid-armv8-cpu
[ci-android-armv8-gpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aandroid-armv8-gpu
[ci-android-x64-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aandroid-x64-cpu
[ci-android-x64-gpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aandroid-x64-gpu
[ci-android-x86-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aandroid-x86-cpu
[ci-android-x86-gpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aandroid-x86-gpu
[ci-elf-riscv32-cpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aelf-riscv32-cpu-gcc
[ci-elf-riscv64-cpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aelf-riscv64-cpu-gcc
[ci-ios-arm64-gpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aios-arm64-gpu
[ci-ios-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aios-cpu
[ci-ios-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aios-cpu
[ci-ios-simulator]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aios-simulator
[ci-ios-simulator]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aios-simulator
[ci-ios-simulator-gpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aios-simulator-gpu
[ci-linux-aarch64-cpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-aarch64-cpu-gcc
[ci-linux-arm-cpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-arm-cpu-gcc
[ci-linux-loongarch64-cpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-loongarch64-cpu-gcc
[ci-linux-mips-cpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-mips-cpu-gcc
[ci-linux-mips64-cpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-mips64-cpu-gcc
[ci-linux-riscv64-cpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-riscv64-cpu-gcc
[ci-linux-x64-cpu-clang]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-x64-cpu-clang
[ci-linux-x64-cpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-x64-cpu-gcc
[ci-linux-x64-gpu-clang]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-x64-gpu-clang
[ci-linux-x64-gpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-x64-gpu-gcc
[ci-linux-x86-cpu-clang]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-x86-cpu-clang
[ci-linux-x86-cpu-gcc]: https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-x86-cpu-gcc
[ci-macos-arm64-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Amacos-arm64-cpu
[ci-macos-arm64-gpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Amacos-arm64-gpu
[ci-macos-x64-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Amacos-x64-cpu
[ci-macos-x64-gpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Amacos-x64-gpu
[ci-web-assembly]: https://github.com/Tencent/ncnn/actions?query=workflow%3Aweb-assembly
[ci-windows-arm-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Awindows-arm-cpu
[ci-windows-arm64-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Awindows-arm64-cpu
[ci-windows-x64-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Awindows-x64-cpu
[ci-windows-x64-gpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Awindows-x64-gpu
[ci-windows-x86-cpu]: https://github.com/Tencent/ncnn/actions?query=workflow%3Awindows-x86-cpu

---

## Support most commonly used CNN network

## 支持大部分常用的 CNN 网络

- Classical CNN:
  [VGG](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014)
  [AlexNet](https://github.com/BVLC/caffe/tree/9b891540183ddc834a02b2bd81b31afae71b2153/models/bvlc_alexnet)
  [GoogleNet](https://github.com/BVLC/caffe/tree/9b891540183ddc834a02b2bd81b31afae71b2153/models/bvlc_googlenet)
  Inception
  ...
- Practical CNN:
  [ResNet](https://github.com/tornadomeet/ResNet)
  [DenseNet](https://github.com/liuzhuang13/DenseNet)
  [SENet](https://github.com/hujie-frank/SENet)
  [FPN](https://github.com/unsky/FPN)
  ...
- Light-weight CNN:
  [SqueezeNet](https://github.com/forresti/SqueezeNet)
  [MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)
  [MobileNetV2/V3](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md)
  [ShuffleNetV1](https://github.com/farmingyard/ShuffleNet)
  [ShuffleNetV2](https://github.com/opconty/keras-shufflenetV2)
  [MNasNet](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet)
  ...
- Face Detection:
  [MTCNN](https://github.com/ipazc/mtcnn)
  [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
  [scrfd](https://github.com/nihui/ncnn-android-scrfd)
  ...
- Detection:
  [VGG-SSD](https://github.com/lzx1413/CAFFE_SSD)
  [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)
  [SqueezeNet-SSD](https://github.com/chuanqi305/SqueezeNet-SSD)
  [MobileNetV2-SSDLite](https://github.com/chuanqi305/MobileNetv2-SSDLite)
  [MobileNetV3-SSDLite](https://github.com/XiaoyuHuang96/MobilenetV3SSDLite-tfkeras)
  ...
- Detection:
  [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn)
  [R-FCN](https://github.com/daijifeng001/R-FCN)
  ...
- Detection:
  [YOLOv2](https://github.com/longcw/yolo2-pytorch)
  [YOLOv3](https://github.com/ultralytics/yolov3)
  [MobileNet-YOLOv3](https://github.com/eric612/MobileNet-YOLO)
  [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
  [YOLOv5](https://github.com/ultralytics/yolov5)
  [YOLOv7](https://github.com/WongKinYiu/yolov7)
  [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
  ...
- Detection:
  [NanoDet](https://github.com/RangiLyu/nanodet)
- Segmentation:
  [FCN](https://github.com/unsky/FPN)
  [PSPNet](https://github.com/hszhao/PSPNet)
  [UNet](https://github.com/zhixuhao/unet)
  [YOLACT](https://github.com/dbolya/yolact)
  ...
- Pose Estimation:
  [SimplePose](https://github.com/dog-qiuqiu/Ultralight-SimplePose)
  ...

---

## HowTo

**[how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build) on Linux / Windows / macOS / Raspberry Pi3, Pi4 / Android / NVIDIA Jetson / iOS / WebAssembly / AllWinner D1 / Loongson 2K1000**

- [Build for Linux / NVIDIA Jetson / Raspberry Pi3, Pi4 / POWER9](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)
- [Build for Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)
- [Build for macOS](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macos)
- [Build for ARM Cortex-A family with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-arm-cortex-a-family-with-cross-compiling)
- [Build for Hisilicon platform with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-hisilicon-platform-with-cross-compiling)
- [Build for Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)
- [Build for iOS on macOS with xcode](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-macos-with-xcode)
- [Build for WebAssembly](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-webassembly)
- [Build for AllWinner D1](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-allwinner-d1)
- [Build for Loongson 2K1000](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-loongson-2k1000)
- [Build for Termux on Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-termux-on-android)
- [Build for QNX](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-qnx)

**[download prebuild binary package for android and ios](https://github.com/Tencent/ncnn/releases)**

**[use ncnn with alexnet](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet) with detailed steps, recommended for beginners :)**

**[ncnn 组件使用指北 alexnet](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet.zh) 附带详细步骤，新人强烈推荐 :)**

**[use netron for ncnn model visualization](https://netron.app)**

**[out-of-the-box web model conversion](https://convertmodel.com/#outputFormat=ncnn)**

[ncnn low-level operation api](https://github.com/Tencent/ncnn/wiki/low-level-operation-api)

[ncnn param and model file spec](https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure)

[ncnn operation param weight table](https://github.com/Tencent/ncnn/wiki/operation-param-weight-table)

[how to implement custom layer step by step](https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step)

---

## FAQ

**[ncnn throw error](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-throw-error)**

**[ncnn produce wrong result](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result)**

**[ncnn vulkan](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-vulkan)**

---

## Features

- Supports convolutional neural networks, supports multiple input and multi-branch structure, can calculate part of the branch
- No third-party library dependencies, does not rely on BLAS / NNPACK or any other computing framework
- Pure C++ implementation, cross-platform, supports Android, iOS and so on
- ARM NEON assembly level of careful optimization, calculation speed is extremely high
- Sophisticated memory management and data structure design, very low memory footprint
- Supports multi-core parallel computing acceleration, ARM big.LITTLE CPU scheduling optimization
- Supports GPU acceleration via the next-generation low-overhead Vulkan API
- Extensible model design, supports 8bit quantization and half-precision floating point storage, can import caffe/pytorch/mxnet/onnx/darknet/keras/tensorflow(mlir) models
- Support direct memory zero copy reference load network model
- Can be registered with custom layer implementation and extended
- Well, it is strong, not afraid of being stuffed with 卷 QvQ

## 功能概述

- 支持卷积神经网络，支持多输入和多分支结构，可计算部分分支
- 无任何第三方库依赖，不依赖 BLAS/NNPACK 等计算框架
- 纯 C++ 实现，跨平台，支持 Android / iOS 等
- ARM Neon 汇编级良心优化，计算速度极快
- 精细的内存管理和数据结构设计，内存占用极低
- 支持多核并行计算加速，ARM big.LITTLE CPU 调度优化
- 支持基于全新低消耗的 Vulkan API GPU 加速
- 可扩展的模型设计，支持 8bit [量化](tools/quantize) 和半精度浮点存储，可导入 caffe/pytorch/mxnet/onnx/darknet/keras/tensorflow(mlir) 模型
- 支持直接内存零拷贝引用加载网络模型
- 可注册自定义层实现并扩展
- 恩，很强就是了，不怕被塞卷 QvQ

---

## supported platform matrix

- ✅ = known work and runs fast with good optimization
- ✔️ = known work, but speed may not be fast enough
- ❔ = shall work, not confirmed
- / = not applied

|            | Windows | Linux | Android | macOS | iOS |
| ---------- | ------- | ----- | ------- | ----- | --- |
| intel-cpu  | ✔️      | ✔️    | ❔      | ✔️    | /   |
| intel-gpu  | ✔️      | ✔️    | ❔      | ❔    | /   |
| amd-cpu    | ✔️      | ✔️    | ❔      | ✔️    | /   |
| amd-gpu    | ✔️      | ✔️    | ❔      | ❔    | /   |
| nvidia-gpu | ✔️      | ✔️    | ❔      | ❔    | /   |
| qcom-cpu   | ❔      | ✔️    | ✅      | /     | /   |
| qcom-gpu   | ❔      | ✔️    | ✔️      | /     | /   |
| arm-cpu    | ❔      | ❔    | ✅      | /     | /   |
| arm-gpu    | ❔      | ❔    | ✔️      | /     | /   |
| apple-cpu  | /       | /     | /       | ✔️    | ✅  |
| apple-gpu  | /       | /     | /       | ✔️    | ✔️  |
| ibm-cpu    | /       | ✔️     | /       | /    | /  |

---

## Project examples

- <https://github.com/nihui/ncnn-android-squeezenet>
- <https://github.com/nihui/ncnn-android-styletransfer>
- <https://github.com/nihui/ncnn-android-mobilenetssd>
- <https://github.com/moli232777144/mtcnn_ncnn>
- <https://github.com/nihui/ncnn-android-yolov5>
- <https://github.com/xiang-wuu/ncnn-android-yolov7>
- <https://github.com/nihui/ncnn-android-scrfd> 🤩
- <https://github.com/shaoshengsong/qt_android_ncnn_lib_encrypt_example>

<img src="https://github.com/nihui/ncnn-assets/raw/master/20181217/ncnn-2.jpg" height ="230"/><img src="https://github.com/nihui/ncnn-assets/raw/master/20181217/4.jpg" height ="230"/><img src="https://github.com/nihui/ncnn-assets/raw/master/20181217/ncnn-33.jpg" height ="230"/><img src="https://github.com/nihui/ncnn-assets/raw/master/20181217/ncnn-m.png" height ="230"/><img src="https://github.com/nihui/ncnn-android-yolov5/raw/master/screenshot.jpg" height ="230"/><img src="https://github.com/nihui/ncnn-android-scrfd/raw/master/screenshot.jpg" height ="230"/><br>

- <https://github.com/magicse/ncnn-colorization-siggraph17><br>
<img src="https://user-images.githubusercontent.com/13585785/189326958-f5a8d6f8-caef-49bf-88da-ae494371195d.jpg" width ="700"/>

- <https://github.com/mizu-bai/ncnn-fortran> Call ncnn from Fortran

- <https://github.com/k2-fsa/sherpa> Use ncnn for real-time speech
  recognition (i.e., speech-to-text); also support embedded devices and provide
  mobile Apps (e.g., Android App)

---

## License

[BSD 3 Clause](LICENSE.txt)
