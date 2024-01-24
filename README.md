![ncnn](https://raw.githubusercontent.com/Tencent/ncnn/master/images/256-ncnn.png)

# ncnn

[![License](https://img.shields.io/badge/license-BSD_3_Clause-blue.svg?style=for-the-badge)](LICENSE.txt)
[![Download Total Count](https://img.shields.io/github/downloads/Tencent/ncnn/total.svg?style=for-the-badge)](https://github.com/Tencent/ncnn/releases)
[![codecov](https://img.shields.io/codecov/c/github/Tencent/ncnn/master?style=for-the-badge)](https://codecov.io/gh/Tencent/ncnn)

ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. 
ncnn is deeply considerate about deployment and uses on mobile phones from the beginning of design.
ncnn does not have third-party dependencies.
It is cross-platform and runs faster than all known open-source frameworks on mobile phone cpu.
Developers can easily deploy deep learning algorithm models to the mobile platform by using efficient ncnn implementation, creating intelligent APPs, and bringing artificial intelligence to your fingertips. 
ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu, and so on.

ncnn 是一个为手机端极致优化的高性能神经网络前向计算框架。
ncnn 从设计之初深刻考虑手机端的部署和使用。
无第三方依赖，跨平台，手机端 cpu 的速度快于目前所有已知的开源框架。
基于 ncnn，开发者能够将深度学习算法轻松移植到手机端高效执行，
开发出人工智能 APP，将 AI 带到你的指尖。
ncnn 目前已在腾讯多款应用中使用，如：QQ，Qzone，微信，天天 P 图等。

---

<table>
<tr>
<td>
<b>技术交流 QQ 群</b><br />
637093648 (超多大佬)<br />
答案：卷卷卷卷卷（已满）
</td>
<td rowspan=2>
<b>Telegram Group</b>

<https://t.me/ncnnyes>
</td>
<td rowspan=2>
<b>Discord Channel</b>

<https://discord.gg/YRsxgmF>
</td>
</tr>
<tr>
<td>
<b>Pocky QQ 群（MLIR YES!）</b><br />
677104663 (超多大佬)<br />
答案：multi-level intermediate representation
</td>
</tr>
</table>

---

## Download & Build status

https://github.com/Tencent/ncnn/releases/latest


<table>
<tr>
<td rowspan=2>
  <img src="https://user-images.githubusercontent.com/25181517/192108372-f71d70ac-7ae6-4c0d-8395-51d8870c2ef0.png" width="120" height="auto">
</td>
<td colspan=3>

  **[how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build) on Linux / Windows / macOS / Raspberry Pi3, Pi4 / POWER / Android / NVIDIA Jetson / iOS / WebAssembly / AllWinner D1 / Loongson 2K1000**

</td>
</tr>
<tr>
<td>Source</td>
<td colspan=2>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-full-source.zip)

</td>
</tr>

<tr>
<td rowspan=3>
  <img src="https://user-images.githubusercontent.com/25181517/117269608-b7dcfb80-ae58-11eb-8e66-6cc8753553f0.png" width="120" height="auto">
</td>
<td colspan=3>

- [Build for Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)
- [Build for Termux on Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-termux-on-android)

</td>
</tr>
<tr>
<td>Android</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-android-vulkan.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-android-vulkan-shared.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/android-armv7-gpu.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Aandroid-armv7-gpu)

</td>
</tr>
<tr>
<td>Android cpuonly</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-android.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-android-shared.zip)

</td>
</tr>

<tr>
<td rowspan=5>
  <img src="https://user-images.githubusercontent.com/25181517/121406611-a8246b80-c95e-11eb-9b11-b771486377f6.png" width="120" height="auto">
</td>
<td colspan=3>
  
- [Build for iOS on macOS with xcode](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-macos-with-xcode)

</td>
</tr>
<tr>
<td>iOS</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ios-vulkan.zip)
  [<img src="https://img.shields.io/badge/+bitcode-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ios-vulkan-bitcode.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/ios-arm64-gpu.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Aios-arm64-gpu)

</td>
</tr>
<tr>
<td>iOS cpuonly</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ios.zip)
  [<img src="https://img.shields.io/badge/+bitcode-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ios-bitcode.zip)

</td>
</tr>
<tr>
<td>iOS-Simulator</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ios-simulator-vulkan.zip)
  [<img src="https://img.shields.io/badge/+bitcode-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ios-simulator-vulkan-bitcode.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/ios-simulator-gpu.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Aios-simulator-gpu)

</td>
</tr>
<tr>
<td>iOS-Simulator cpuonly</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ios-simulator.zip)
  [<img src="https://img.shields.io/badge/+bitcode-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ios-simulator-bitcode.zip)

</td>
</tr>

<tr>
<td rowspan=11>
  <img src="https://user-images.githubusercontent.com/25181517/186884152-ae609cca-8cf1-4175-8d60-1ce1fa078ca2.png" width="120" height="auto">
</td>
<td colspan=3>
  
- [Build for macOS](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macos)

</td>
</tr>
<tr>
<td>macOS</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-macos-vulkan.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/macos-arm64-gpu.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Amacos-arm64-gpu)

</td>
</tr>
<tr>
<td>macOS cpuonly</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-macos.zip)

</td>
</tr>
<tr>
<td>Mac-Catalyst</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-mac-catalyst-vulkan.zip)
  [<img src="https://img.shields.io/badge/+bitcode-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-mac-catalyst-vulkan-bitcode.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/mac-catalyst-arm64-gpu.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Amac-catalyst-arm64-gpu)

</td>
</tr>
<tr>
<td>Mac-Catalyst cpuonly</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-mac-catalyst.zip)
  [<img src="https://img.shields.io/badge/+bitcode-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-mac-catalyst-bitcode.zip)

</td>
</tr>
<tr>
<td>watchOS</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-watchos.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/watchos-cpu.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Awatchos-cpu)

</td>
</tr>
<tr>
<td>watchOS-Simulator</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-watchos-simulator.zip)

</td>
</tr>
<tr>
<td>tvOS</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-tvos.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/tvos-cpu.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Atvos-cpu)

</td>
</tr>
<tr>
<td>tvOS-Simulator</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-tvos-simulator.zip)

</td>
</tr>
<tr>
<td>Apple xcframework</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-apple-vulkan.zip)
  [<img src="https://img.shields.io/badge/+bitcode-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-apple-vulkan-bitcode.zip)

</td>
<td rowspan=2>

</td>
</tr>
<tr>
<td>Apple xcframework cpuonly</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-apple.zip)
  [<img src="https://img.shields.io/badge/+bitcode-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-apple-bitcode.zip)

</td>
</tr>

<tr>
<td rowspan=3>
  <img src="https://user-images.githubusercontent.com/25181517/186884153-99edc188-e4aa-4c84-91b0-e2df260ebc33.png" width="120" height="auto">
</td>
<td colspan=3>

- [Build for Linux / NVIDIA Jetson / Raspberry Pi3, Pi4 / POWER](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)

</td>
</tr>
<tr>
<td>Ubuntu 20.04</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ubuntu-2004.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ubuntu-2004-shared.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-x64-gpu-gcc.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-x64-gpu-gcc)

</td>
</tr>
<tr>
<td>Ubuntu 22.04</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ubuntu-2204.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-ubuntu-2204-shared.zip)

</td>
</tr>

<tr>
<td rowspan=5>
  <img alt="windows" src="https://user-images.githubusercontent.com/25181517/186884150-05e9ff6d-340e-4802-9533-2c3f02363ee3.png" width="120" height="auto">
</td>
<td colspan=3>

- [Build for Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)

</td>
</tr>
<tr>
<td>VS2015</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-windows-vs2015.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-windows-vs2015-shared.zip)

</td>
<td rowspan=4>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/windows-x64-gpu.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Awindows-x64-gpu)

</td>
</tr>
<tr>
<td>VS2017</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-windows-vs2017.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-windows-vs2017-shared.zip)

</td>
</tr>
<tr>
<td>VS2019</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-windows-vs2019.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-windows-vs2019-shared.zip)

</td>
</tr>
<tr>
<td>VS2022</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-windows-vs2022.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-windows-vs2022-shared.zip)

</td>
</tr>

<tr>
<td rowspan=2>
  <img src="https://user-images.githubusercontent.com/25181517/188324036-d704ac9a-6e61-4722-b978-254b25b61bed.png" width="120" height="auto">
</td>
<td colspan=3>

- [Build for WebAssembly](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-webassembly)

</td>
</tr>
<tr>
<td>WebAssembly</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20240102-webassembly.zip)

</td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/web-assembly.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Aweb-assembly)

</td>
</tr>

<tr>
<td rowspan=8>
  <img src="https://github.com/marwin1991/profile-technology-icons/assets/76662862/2481dc48-be6b-4ebb-9e8c-3b957efe69fa" width="120" height="auto">
</td>
<td colspan=3>

- [Build for ARM Cortex-A family with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-arm-cortex-a-family-with-cross-compiling)
- [Build for Hisilicon platform with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-hisilicon-platform-with-cross-compiling)
- [Build for AllWinner D1](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-allwinner-d1)
- [Build for Loongson 2K1000](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-loongson-2k1000)
- [Build for QNX](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-qnx)

</td>
</tr>
<tr>
<td>Linux (arm)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-arm-cpu-gcc.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-arm-cpu-gcc)

</td>
</tr>
<tr>
<td>Linux (aarch64)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-aarch64-cpu-gcc.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-aarch64-cpu-gcc)

</td>
</tr>
<tr>
<td>Linux (mips)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-mips-cpu-gcc.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-mips-cpu-gcc)

</td>
</tr>
<tr>
<td>Linux (mips64)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-mips64-cpu-gcc.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-mips64-cpu-gcc)

</td>
</tr>
<tr>
<td>Linux (ppc64)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-ppc64-cpu-gcc.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-ppc64-cpu-gcc)

</td>
</tr>
<tr>
<td>Linux (riscv64)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-riscv64-cpu-gcc.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-riscv64-cpu-gcc)

</td>
</tr>
<tr>
<td>Linux (loongarch64)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-loongarch64-cpu-gcc.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-loongarch64-cpu-gcc)

</td>
</tr>

</table>


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
