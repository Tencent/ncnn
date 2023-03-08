# Protobuf 类问题解决方法

## 问题分析

protobuf 有关的报错，一般都是两个原因：

1. 需要的 pb 没安装/`FindProtobuf.cmake`不存在，最终 `find_package` 失败
2. 系统不止一套 pb，导致 bin/lib/include 三者不匹配

如果你遇到了这些报错，都可以通过本文档解决：

1. Linux 编译 `caffe2ncnn` 时报 `Protobuf not found`
2. 编译 `caffe2ncnn` 时报 protoc 和 protobuf.so 版本不匹配

## （推荐）通用处理办法

这个办法包治百病，**不管什么情况一定生效**

1. 编译下载 protobuf，以 3.20.0 版本为例

```bash
$ wget https://github.com/protocolbuffers/protobuf/releases/download/v3.20.0/protobuf-cpp-3.20.0.tar.gz
$ tar xvf protobuf-cpp-3.20.0.tar.gz
$ cd protobuf-3.20.0/
$ ./configure --prefix=/path/to/install
$ make && make install
```
注意需要 `--prefix`，不要装到系统里。能遇到这些错，说明本来系统环境就有问题，再给系统环境装 lib 就更乱了。

2. 修改 cmake

找到报错的 CMakeLists.txt，在 `find_package` 前插入 protobuf 路径。

```bash
# 加入下面 1 行
list(APPEND CMAKE_PREFIX_PATH "/path/to/install")

find_package(Protobuf REQUIRED)
...
```

3. 调整 cmake 选项

`cmake ..` 时，额外加入选项 `-DProtobuf_PROTOC_EXECUTABLE=/path/to/install/bin/protoc`

```bash
$ cd /path/to/ncnn/build
$ rm -rf CMakeCache
# 加入新选项
$ cmake .. -DProtobuf_PROTOC_EXECUTABLE=/path/to/install/bin/protoc 
$ ...
```

## （不推荐）自己改环境变量

### 一、遇到 `Protobuf not found`

是因为 protobuf 未安装或环境变量未设置

1. 安装 protobuf

Ubuntu 系统尝试以下命令
```bash
$ sudo apt-get install libprotobuf-dev protobuf-compiler
```

CentOS 尝试
```bash
$ sudo yum install protobuf-devel.x86_64 protobuf-compiler.x86_64
```

2. 然后设置 C++ 环境

在 LD_LIBRARY_PATH 增加参数

```bash
$ export LD_LIBRARY_PATH=${YOUR_PROTOBUF_LIB_PATH}:$LD_LIBRARY_PATH
```

### 二、遇到 protoc 和 protobuf.so 版本不匹配

1. 先看 protoc 需要的 so 版本号
```bash
$ ldd `whereis protoc| awk '{print $2}'` | grep libprotobuf.so
```

例如是 libprotobuf.so.10

2. 然后搜这个文件所在的路径
```bash
$ cd / && find . -type f | grep libprotobuf.so.10
```

假设在`/home/user/mydir`

3. 设置 protobuf.so 的搜索目录
```bash
$ export LD_LIBRARY_PATH=/home/user/mydir:$LD_LIBRARY_PATH
```

### 三、行走江湖必备
关于环境变量设置、工具和技巧，强烈建议学习下 https://missing.csail.mit.edu/ 
