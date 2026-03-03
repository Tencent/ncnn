# Build ncnn on Windows XP

> **Contributors:** [@Sugar-Baby](https://github.com/Sugar-Baby) and [@AtomAlpaca](https://github.com/AtomAlpaca)

## 0. 环境准备

#### 0.1 虚拟机设置

我使用的是[我的MSDN](https://www.imsdn.cn/)提供的[Windows XP SP3 x64版本](https://www.imsdn.cn/operating-systems/windows-xp/)。虚拟机使用Oracle VM VirtualBox，内存4GB，存储空间64GB（C盘16GB，D盘48GB）。

**在虚拟机关机的情况下**，点击虚拟机管理器界面的"设置"-"网络"-"高级"，将控制芯片改为PCnet-FAST III，混杂模式设置为拒绝，勾选接入网线，点击"OK"保存。重启虚拟机就可以连接上网络了。

点击虚拟机界面的"设备"-"安装增强功能..."，在虚拟机中进入"我的电脑"，刷新后出现"VirtualBox Guest Additions (D: )"，右键选择"自动播放"，完成安装后重启。

点击虚拟机界面的"设备"-"共享粘贴板"，设置为"双向"。点击"设备"-"共享文件夹"-"共享文件夹.."，点击右侧加号，在"共享文件夹路径"中选择"其他..."，然后选择需要共享的主机文件夹。勾选"自动挂载"和"固定分配"，点击"OK"保存。在虚拟机中进入"我的电脑"，刷新后出现'VBoxSvr' 上的 <主机文件夹名称>，双击进入就可以双向传输文件了。

#### 0.2 开发环境配置

浏览器推荐[Mypal 68](https://www.mypal-browser.org/download.html)，注意要选择32位版本。Windows XP自带ZIP文件解压。安装后就可以访问互联网了。

从Github下载[w64devkit](https://github.com/skeeto/w64devkit)，选择x86版本。这里下载的是一个自解压的7z文件，在虚拟机中解压即可。

在"开始"-"控制面板"-"切换到经典视图"-"系统"-"高级"-"环境变量"-"系统变量"中，选择Path，点击"编辑"，在字符串末尾加入一个分号(;)，然后粘贴w64devkit下bin文件夹的目录。点击"确定"保存之后可以打开命令提示符输入例如c++的命令验证是否成功加入环境变量。

由于年代过于久远，Git的官方release已经没有兼容Windows XP的版本了。最后一个兼容的版本(1.9.5)可以在[这里](https://www.xiazaiba.com/html/29352.html)下载。

为了使用Git，需要安装[Win32 OpenSSL](https://slproweb.com/products/Win32OpenSSL.html)。选择Win32 OpenSSL Light版本。这个过程中会附带安装VC++ 2022运行时库。

如果因为协议、代理等问题不能在虚拟机中使用Git，也可以下载ZIP版本后在虚拟机中解压。

需要手动下载[CMake最后支持Windows XP的版本](https://github.com/Kitware/CMake/releases/download/v3.10.3/cmake-3.10.3-win32-x86.zip)。建议解压在C:\Program Files下，并且需要设置系统变量，到CMake目录下的bin文件夹。具体可以参考上面w64devkit的方法。

## 1. 编译

### 1.1 使用 MinGW-w64

运行

```bash
cd <ncnn-root-dir>
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/windows-xp-mingw.toolchain.cmake -DNCNN_VULKAN=OFF -DNCNN_SIMPLEOCV=ON -DNCNN_AVX=OFF -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles" ..
make -j2
make install
```

由于平台性能的限制，Vulkan SDK 最低要求 Windows 7 SP1，XP 无法安装官方驱动和工具链，因此需要关闭Vulkan选项。同时需要使用简化版 OpenCV 替代库NCNN_SIMPLEOCV。

### 1.2 使用 Clang

需要先配置 MinGW-w64 环境，然后安装 Clang 6.0 或更高版本。

```bash
cd <ncnn-root-dir>
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/windows-xp-clang.toolchain.cmake -DNCNN_SIMPLEOCV=ON -DNCNN_SIMPLEOMP=ON -DNCNN_AVX=OFF -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles" ..
make -j2
make install
```

### 1.3 使用 Visual Studio (MSVC)

需要安装支持 Windows XP 的 v141_xp 工具集：

1. 打开 Visual Studio 安装程序（工具 → 获取工具和功能）
2. 选择"使用 C++ 的桌面开发"
3. 在摘要部分选择"对 C++ 的 Windows XP 支持"
4. 点击修改

```bash
cd <ncnn-root-dir>
mkdir build
cd build
cmake -A WIN32 -G "Visual Studio 17 2022" -T v141_xp -DNCNN_SIMPLEOCV=ON -DNCNN_OPENMP=OFF -DNCNN_AVX=OFF -DNCNN_BUILD_WITH_STATIC_CRT=ON -DCMAKE_TOOLCHAIN_FILE=../toolchains/windows-xp-msvc.toolchain.cmake ..
cmake --build . --config Release -j 2
cmake --build . --config Release --target install
```

## 2. 测试

### 2.1 benchncnn

将benchmark目录下的所有文件复制到build/benchmark目录下。在命令提示符中cd到build/benchmark， 然后运行

```bash
benchncnn [测试的循环次数] [线程数] [节能模式]
```

其中，节能模式取值为0时关闭，为1时打开。

### 2.2 examples

从[这里](https://github.com/nihui/ncnn-assets/tree/master/models)可以下载到所有需要的param和bin文件。需要注意的是，ZF_faster_rcnn_final.bin开头的三个文件（.zip，.z01，.z02）最好先放在主机上解压出bin文件再传进虚拟机。

把这些文件放在build/examples目录下。

我写了一个bat脚本来批量测试这些模型：

```batch
@echo off
setlocal enabledelayedexpansion

set EXAMPLES_DIR=<ncnn-root-dir>\BUILD\EXAMPLES
set IMAGE_PATH=<ncnn-root-dir>\IMAGES\256-ncnn.png
set LOG_FILE=test_results.log

echo NCNN Examples Test Results > %LOG_FILE%
echo ========================= >> %LOG_FILE%
echo Test started: %date% %time% >> %LOG_FILE%
echo. >> %LOG_FILE%

for %%f in ("%EXAMPLES_DIR%\*.exe") do (
    set EXE_NAME=%%~nf
    set EXE_PATH=%%f
    echo Testing: !EXE_NAME! >> %LOG_FILE%
    echo -------------------------------- >> %LOG_FILE%

    !EXE_PATH! "%IMAGE_PATH%" >> %LOG_FILE% 2>&1

    if errorlevel 1 (
        echo [ERROR] !EXE_NAME! failed to run. >> %LOG_FILE%
    ) else (
        echo [SUCCESS] !EXE_NAME! completed. >> %LOG_FILE%
    )
    echo. >> %LOG_FILE%
)

echo Test finished: %date% %time% >> %LOG_FILE%
echo Results saved to %LOG_FILE%
endlocal
```

把这个bat脚本放在build/examples目录下，替换掉所有的`<ncnn-root-dir>`，双击运行。通过生成的test_results.log即可查看所有模型的结果。

通过修改`set IMAGE_PATH=<ncnn-root-dir>\IMAGES\256-ncnn.png`中的路径来更换需要测试的文件。