![ncnn](https://raw.githubusercontent.com/Tencent/ncnn/master/images/256-ncnn.png)

# ncnn

[![License](https://img.shields.io/badge/license-BSD_3_Clause-blue.svg?style=for-the-badge)](LICENSE.txt)
[![Download Total Count](https://img.shields.io/github/downloads/Tencent/ncnn/total.svg?style=for-the-badge)](https://github.com/Tencent/ncnn/releases)
[![codecov](https://img.shields.io/codecov/c/github/Tencent/ncnn/master?style=for-the-badge)](https://codecov.io/gh/Tencent/ncnn)

ncnn is a high-performance neural network inference framework optimized for mobile, embedded, and desktop deployment.
It has no third-party runtime dependencies, runs across CPU and Vulkan GPU backends, and provides tools such as pnnx for converting PyTorch and ONNX models to ncnn.
Developers can deploy deep learning models efficiently on phones, PCs, browsers, and edge devices.
ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu, and so on.

ncnn 是一个面向移动端、嵌入式和桌面端部署优化的高性能神经网络推理框架。
ncnn 无第三方运行时依赖，支持 CPU 和 Vulkan GPU 后端，并提供 pnnx 等工具将 PyTorch 和 ONNX 模型转换为 ncnn 模型。
基于 ncnn，开发者可以将深度学习模型高效部署到手机、PC、浏览器和边缘设备上。
ncnn 目前已在腾讯多款应用中使用，如：QQ，Qzone，微信，天天 P 图等。

---

## Quick Start

The recommended beginner path is PyTorch -> pnnx -> ncnn.

<table>
<tr>
<td width="50%" valign="top">

**Install pnnx in a PyTorch environment**

```shell
pip3 install pnnx
```

**Export a PyTorch model to ncnn**

```python
import torch
import torch.nn as nn
import pnnx

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.mean((2, 3))
        return self.fc(x)

model = Model().eval()

x = torch.rand(1, 3, 224, 224)
pnnx.export(model, "model.pt", (x,))
```

This generates `model.ncnn.param` and `model.ncnn.bin`.

</td>
<td width="50%" valign="top">

**Run with ncnn C++ API**

```cpp
#include "net.h"

ncnn::Net net;
net.load_param("model.ncnn.param");
net.load_model("model.ncnn.bin");

ncnn::Mat in(224, 224, 3);

auto ex = net.create_extractor();
ex.input("in0", in);

ncnn::Mat out;
ex.extract("out0", out);
```

**Or use Python**

```python
import numpy as np
import ncnn

net = ncnn.Net()
net.load_param("model.ncnn.param")
net.load_model("model.ncnn.bin")

x = np.zeros((3, 224, 224), np.float32)
mat = ncnn.Mat(x)

ex = net.create_extractor()
ex.input("in0", mat)

ret, out = ex.extract("out0")
print(np.array(out).shape)
```

</td>
</tr>
</table>

See [pnnx](tools/pnnx), [use ncnn with PyTorch or ONNX](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx), [Python API](python), and [examples](examples) for complete workflows.

---

## Community

<table>
<tr>
<td>
<b>技术交流 QQ 群</b><br />
637093648 (超多大佬)<br />
答案：卷卷卷卷卷（已满）
</td>
<td rowspan=3>
<b>Telegram Group</b>

<https://t.me/ncnnyes>
</td>
<td rowspan=3>
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
<tr>
<td>
<b>他们都不知道 pnnx 有多好用群</b><br />
818998520 (新群！)
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

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-full-source.zip)

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

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-android-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-android.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/android.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Aandroid)

</td>
</tr>
<tr>
<td>Android shared</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-android-vulkan-shared.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-android-shared.zip)

</td>
</tr>

<tr>
<td rowspan=3>
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/37/HMOS_Logo_Icon.svg" width="120" height="auto">
</td>
<td colspan=3>

- [Build for HarmonyOS with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-harmonyos-with-cross-compiling)

</td>
</tr>
<tr>
<td>HarmonyOS</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-harmonyos-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-harmonyos.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/harmonyos.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Aharmonyos)

</td>
</tr>
<tr>
<td>HarmonyOS shared</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-harmonyos-vulkan-shared.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-harmonyos-shared.zip)

</td>
</tr>

<tr>
<td rowspan=3>
  <img src="https://user-images.githubusercontent.com/25181517/121406611-a8246b80-c95e-11eb-9b11-b771486377f6.png" width="120" height="auto">
</td>
<td colspan=3>

- [Build for iOS on macOS with xcode](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-macos-with-xcode)

</td>
</tr>
<tr>
<td>iOS</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-ios-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-ios.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/ios.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Aios)

</td>
</tr>
<tr>
<td>iOS-Simulator</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-ios-simulator-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-ios-simulator.zip)

</td>
</tr>

<tr>
<td rowspan=10>
  <img src="https://user-images.githubusercontent.com/25181517/186884152-ae609cca-8cf1-4175-8d60-1ce1fa078ca2.png" width="120" height="auto">
</td>
<td colspan=3>

- [Build for macOS](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macos)

</td>
</tr>
<tr>
<td>macOS</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-macos-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-macos.zip)

</td>
<td rowspan=1>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/macos.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Amacos)

</td>
</tr>
<tr>
<td>Mac-Catalyst</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-mac-catalyst-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-mac-catalyst.zip)

</td>
<td rowspan=1>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/mac-catalyst.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Amac-catalyst)

</td>
</tr>
<tr>
<td>watchOS</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-watchos.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/watchos.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Awatchos)

</td>
</tr>
<tr>
<td>watchOS-Simulator</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-watchos-simulator.zip)

</td>
</tr>
<tr>
<td>tvOS</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-tvos-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-tvos.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/tvos.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Atvos)

</td>
</tr>
<tr>
<td>tvOS-Simulator</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-tvos-simulator-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-tvos-simulator.zip)

</td>
</tr>
<tr>
<td>visionOS</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-visionos-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-visionos.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/visionos.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Avisionos)

</td>
</tr>
<tr>
<td>visionOS-Simulator</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-visionos-simulator-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-visionos-simulator.zip)

</td>
</tr>
<tr>
<td>Apple xcframework</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-apple-vulkan.zip)
  [<img src="https://img.shields.io/badge/+cpuonly-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-apple.zip)

</td>
<td rowspan=1>

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
<td>Ubuntu 22.04</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-ubuntu-2204.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-ubuntu-2204-shared.zip)

</td>
<td rowspan=2>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-x64-gpu-gcc.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-x64-gpu-gcc)

</td>
</tr>
<tr>
<td>Ubuntu 24.04</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-ubuntu-2404.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-ubuntu-2404-shared.zip)

</td>
</tr>

<tr>
<td rowspan=5>
  <img alt="windows" src="https://user-images.githubusercontent.com/25181517/186884150-05e9ff6d-340e-4802-9533-2c3f02363ee3.png" width="120" height="auto">
</td>
<td colspan=3>

- [Build for Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)
- [Build for Windows x64 using MinGW-w64](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-mingw-w64)

</td>
</tr>
<tr>
<td>VS2015</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-windows-vs2015.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-windows-vs2015-shared.zip)

</td>
<td rowspan=4>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/windows.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Awindows)

</td>
</tr>
<tr>
<td>VS2017</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-windows-vs2017.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-windows-vs2017-shared.zip)

</td>
</tr>
<tr>
<td>VS2019</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-windows-vs2019.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-windows-vs2019-shared.zip)

</td>
</tr>
<tr>
<td>VS2022</td>
<td>

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-windows-vs2022.zip)
  [<img src="https://img.shields.io/badge/+shared-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-windows-vs2022-shared.zip)

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

  [<img src="https://img.shields.io/badge/download-blue?style=for-the-badge">](https://github.com/Tencent/ncnn/releases/latest/download/ncnn-20260526-webassembly.zip)

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

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-arm.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-arm)

</td>
</tr>
<tr>
<td>Linux (aarch64)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-aarch64.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-aarch64)

</td>
</tr>
<tr>
<td>Linux (mips)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-mips.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-mips)

</td>
</tr>
<tr>
<td>Linux (mips64)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-mips64.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-mips64)

</td>
</tr>
<tr>
<td>Linux (ppc64)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-ppc64.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-ppc64)

</td>
</tr>
<tr>
<td>Linux (riscv64)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-riscv64.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-riscv64)

</td>
</tr>
<tr>
<td>Linux (loongarch64)</td>
<td></td>
<td>

  [<img src="https://img.shields.io/github/actions/workflow/status/Tencent/ncnn/linux-loongarch64.yml?branch=master&style=for-the-badge&label=build">](https://github.com/Tencent/ncnn/actions?query=workflow%3Alinux-loongarch64)

</td>
</tr>

</table>

---

## Build

Use the prebuilt packages above when possible. To build from source, see the full [how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build) guide for Linux, Windows, macOS, Android, iOS, WebAssembly, HarmonyOS, Raspberry Pi, Jetson, and embedded targets.

Common Linux build:

```bash
git clone --recursive https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..
cmake --build . -j$(nproc)
```

---

## Model Conversion

| Source model | Recommended path | Docs |
| --- | --- | --- |
| PyTorch | `pnnx.export(model, "model.pt", (input_tensor,))` or `pnnx model.pt inputshape=[...]` | [pnnx](tools/pnnx), [PyTorch / ONNX guide](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx) |
| ONNX | `pnnx model.onnx` | [pnnx](tools/pnnx), [onnx tools](tools/onnx) |
| ncnn model optimization | `ncnnoptimize model.param model.bin new.param new.bin flag` | [quantization](tools/quantize), [model file spec](https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure) |
| Legacy Caffe / MXNet / Darknet | Use compatibility converters when maintaining older models | [caffe](tools/caffe), [mxnet](tools/mxnet), [darknet](tools/darknet), [AlexNet legacy tutorial](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet) |

Use [Netron](https://netron.app) to inspect `.param`, `.onnx`, and `.pnnx.param` graphs.

---

## Features

- No third-party runtime dependencies and no BLAS / NNPACK requirement.
- Pure C++ implementation with C API and Python binding.
- Optimized CPU inference for mobile and embedded processors, including ARM NEON and multi-core scheduling.
- Vulkan GPU acceleration for supported platforms.
- Low memory footprint with explicit blob/workspace allocator design.
- Supports multi-input, multi-output, and multi-branch graphs.
- PyTorch and ONNX conversion through pnnx, plus legacy converter support for older model formats.
- Supports fp16 storage/arithmetic paths, int8 quantized inference, model optimization, and custom layers.
- Direct memory reference loading for `.param` and `.bin` models.

---

## Model and Workload Coverage

ncnn is still strong for classic and mobile CNN workloads, but current usage is broader than CNN-only deployment.

- Classification and backbones: [VGG](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014), [AlexNet](https://github.com/BVLC/caffe/tree/9b891540183ddc834a02b2bd81b31afae71b2153/models/bvlc_alexnet), [GoogleNet](https://github.com/BVLC/caffe/tree/9b891540183ddc834a02b2bd81b31afae71b2153/models/bvlc_googlenet), Inception, [ResNet](https://github.com/tornadomeet/ResNet), [DenseNet](https://github.com/liuzhuang13/DenseNet), [SENet](https://github.com/hujie-frank/SENet), [SqueezeNet](https://github.com/forresti/SqueezeNet), MobileNet, ShuffleNet, MNasNet.
- Detection and face: SSD, Faster R-CNN, R-FCN, [MTCNN](https://github.com/ipazc/mtcnn), [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface), [scrfd](https://github.com/nihui/ncnn-android-scrfd), [YOLOv2](https://github.com/longcw/yolo2-pytorch), [YOLOv3](https://github.com/ultralytics/yolov3), [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4), [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv7](https://github.com/WongKinYiu/yolov7), [YOLOv8](https://github.com/nihui/ncnn-android-yolov8), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [NanoDet](https://github.com/RangiLyu/nanodet).
- Segmentation, pose, and OCR: FCN, [PSPNet](https://github.com/hszhao/PSPNet), [UNet](https://github.com/zhixuhao/unet), [YOLACT](https://github.com/dbolya/yolact), [SimplePose](https://github.com/dog-qiuqiu/Ultralight-SimplePose), PP-OCR examples.
- Audio, generation, and language workloads are represented by community projects and examples where the model operators are supported.

For operator-level detail, see [supported PyTorch operator status](tools/pnnx#supported-pytorch-operator-status), [supported ONNX operator status](tools/pnnx#supported-onnx-operator-status), and [operation param weight table](https://github.com/Tencent/ncnn/wiki/operation-param-weight-table).

---

## Project Examples

| Area | Project |
| --- | --- |
| Image generation | [zimage-ncnn-vulkan](https://github.com/nihui/zimage-ncnn-vulkan) - Z-Image generation with ncnn and Vulkan |
| LLM / embedding / vision-language | [ncnn_llm](https://github.com/futz12/ncnn_llm) - LLM, embedding, and vision-language examples with ncnn |
| Android classification | [ncnn-android-squeezenet](https://github.com/nihui/ncnn-android-squeezenet) |
| Android style transfer | [ncnn-android-styletransfer](https://github.com/nihui/ncnn-android-styletransfer) |
| Android detection | [ncnn-android-mobilenetssd](https://github.com/nihui/ncnn-android-mobilenetssd), [ncnn-android-yolov5](https://github.com/nihui/ncnn-android-yolov5), [ncnn-android-yolov7](https://github.com/xiang-wuu/ncnn-android-yolov7), [ncnn-android-scrfd](https://github.com/nihui/ncnn-android-scrfd) |
| Face detection | [mtcnn_ncnn](https://github.com/moli232777144/mtcnn_ncnn) |
| Qt / Android integration | [qt_android_ncnn_lib_encrypt_example](https://github.com/shaoshengsong/qt_android_ncnn_lib_encrypt_example) |
| Colorization | [ncnn-colorization-siggraph17](https://github.com/magicse/ncnn-colorization-siggraph17) |
| Fortran binding | [ncnn-fortran](https://github.com/mizu-bai/ncnn-fortran) |
| Speech recognition | [sherpa](https://github.com/k2-fsa/sherpa) - real-time speech recognition on embedded and mobile devices |

---

## Documentation And FAQ

| Topic | Links |
| --- | --- |
| Build | [how to build](https://github.com/Tencent/ncnn/wiki/how-to-build) |
| PyTorch / ONNX conversion | [use ncnn with PyTorch or ONNX](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx), [pnnx](tools/pnnx), [PyTorch converter notes](tools/pytorch) |
| API and examples | [C++ examples](examples), [Python API](python), [low-level operation API](https://github.com/Tencent/ncnn/wiki/low-level-operation-api) |
| Model format | [param and model file spec](https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure), [operation param weight table](https://github.com/Tencent/ncnn/wiki/operation-param-weight-table) |
| Extension | [custom layer guide](https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step), [plugin tools](tools/plugin) |
| FAQ | [deepwiki](https://deepwiki.com/Tencent/ncnn), [throw error](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-throw-error), [wrong result](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result), [Vulkan](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-vulkan) |
| Legacy beginner material | [use ncnn with AlexNet](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet), [AlexNet Chinese tutorial](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet.zh) |

---

## License

[BSD 3 Clause](LICENSE.txt)
