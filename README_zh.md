# ncnn_mp

**ncnn_mp** 是一个为 MicroPython 打造的外部 C 模块，它将腾讯的 [ncnn](https://github.com/Tencent/ncnn) 高性能神经网络推理框架引入到资源受限的微控制器中。本仓库提供了 **ncnn C API** 的绑定，让你可以直接在 ESP32、STM32 等嵌入式设备上运行 AI 推理。

尽管 `ncnn` 官方仓库中已支持 Python 绑定，但 `ncnn_mp` 使开发者能够在嵌入式平台上使用 Python 风格且面向对象的 ncnn API。

## 特点

- **面向对象的 API 设计**: 将 ncnn 的 C 风格句柄转换为 Python 对象 (`Allocator`, `Option`, `Mat`, `Blob`, `ParamDict`, `DataReader`, `ModelBin`, `Layer`, `Net`, and `Extractor`)。
- **为 MCU 设计**: 使用一层对 **ncnn C API** 的封装，使其非常适用于内存和处理能力有限的设备。
- **开发体验**: 仓库中包含 `.pyi` 文件，可在 IDE 中提供**自动补全**、**类型提示**和**文档**功能。
- **跨平台**: 可为多种 MicroPython 平台实现轻松构建，并提供了 Unix 和 ESP32-S3 的构建示例。

---

## 构建

### 依赖

  - [ncnn](https://github.com/Tencent/ncnn)
  - [MicroPython](https://github.com/micropython/micropython)
  - CMake / GNU Make

### 准备工作

```bash
git submodule update --init --recursive
```

### 为 Linux 构建 (Unix Port, 使用 Make)

1.  **构建 ncnn**

```bash
cd ncnn
mkdir build && cd build
# 一个功能相对完整的示例配置
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=ON -DNCNN_VULKAN=ON -DNCNN_BUILD_BENCHMARK=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=./install ..
make -j4
make install
```

2.  **构建 MicroPython 固件**

```bash
cd micropython/ports/unix/
make -C ../../mpy-cross -j4
make submodules -j4
make USER_C_MODULES=../../../modules USE_VULKAN=1 -j4
./build-standard/micropython ../../../examples/main.py
```

> **调试**: 在构建 ncnn 时，将 `CMAKE_BUILD_TYPE` 更改为 `DEBUG`。然后使用 `make USER_C_MODULES=../../../modules USE_VULKAN=1 NCNN_INSTALL_PREFIX=../../../ncnn/build-debug/install DEBUG=1 COPT=-O0 -j4` 命令构建 MicroPython。

### 为 ESP32-S3 构建 (交叉编译, 使用 CMake)

这个示例假设你在 Linux 主机上为 ESP32-S3 目标进行交叉编译。

0.  **准备 ESP-IDF 工具链**

你可以将此仓库克隆到任意位置。但在使用 `esp-idf` sdk 时，你必须进入仓库目录运行 `source ./export.sh` 为当前终端会话配置环境。

```bash
git clone https://github.com/espressif/esp-idf
cd esp-idf
```

MicroPython 需要特定版本的 `esp-idf`。截至 **2025年8月**，支持的版本包括 `v5.2`、`v5.2.2`、`v5.3`、`v5.4`、`v5.4.1` 和 `v5.4.2`。请查阅 [官方 MicroPython ESP32 移植文档](https://github.com/micropython/micropython/blob/master/ports/esp32/README.md) 获取最新的版本列表。

```bash
git checkout v5.4.2
git submodule update --init --recursive
./install.sh esp32s3
source ./export.sh
```

1.  **为 ESP32-S3 构建 ncnn**

```bash
cd ncnn
mkdir build-esp32s3 && cd build-esp32s3
# 你可以在此处添加配置以最小化 ncnn 库的大小
cmake -DCMAKE_TOOLCHAIN_FILE=../../toolchains/esp32s3.toolchain.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=./install ..
make -j4
make install
```

2.  **构建 MicroPython 固件**

```bash
cd micropython/ports/esp32
make -C ../../mpy-cross -j4
make submodules BOARD=ESP32_GENERIC_S3 -j4
idf.py -D MICROPY_BOARD=ESP32_GENERIC_S3 -D USER_C_MODULES=../../../../modules/ncnn_mp/micropython.cmake -D NCNN_INSTALL_PREFIX=../../../../ncnn/build-esp32s3/install build
```

3.  **烧录到设备**

```bash
# [可选]: 擦除整个闪存芯片
idf.py -p /dev/ttyACM0 erase-flash

# 烧录新固件
idf.py -p /dev/ttyACM0 flash

# 监控设备输出
# 同等于: tio /dev/ttyACM0
idf.py -p /dev/ttyACM0 monitor
```

---

## 参考资料
- MicroPython 外部 C 模块文档: [MicroPython external C modules](https://docs.micropython.org/en/latest/develop/cmodules.html)
- 如何构建最小化的 ncnn 库: [build minimal library](https://github.com/Tencent/ncnn/wiki/build-minimal-library)
- ESP-IDF ESP32-S3 官方指南: [Get Started](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/index.html)
- 在 WSL 中使用 USB 设备: [Connect USB devices](https://learn.microsoft.com/en-us/windows/wsl/connect-usb#attach-a-usb-device)