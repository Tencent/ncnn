### Linux 编译 `caffe2ncnn` 时报 `Protobuf not found`

一般是因为 protobuf 未安装或环境变量未设置

1. 安装 protobuf

Ubuntu 系统尝试以下命令
> sudo apt-get install libprotobuf-dev protobuf-compiler

CentOS 尝试
> sudo yum install protobuf-devel.x86_64 protobuf-compiler.x86_64

2. 然后设置 C++ 环境
打开`~/.bashrc`，在末尾增加
> export LD_LIBRARY_PATH=${YOUR_PROTOBUF_LIB_PATH}:$LD_LIBRARY_PATH

3. 让配置生效
> source ~/.bashrc


### 编译 `caffe2ncnn` 时报 protoc 和 protobuf.so 版本不匹配

一般是因为系统安装了不止一个 protobuf。

#### 直接改链接路径
1. 先看 protoc 需要的 so 版本号
> ldd \`whereis protoc| awk '{print $2}'\` | grep libprotobuf.so

例如是 libprotobuf.so.10

2. 然后搜这个文件所在的路径
> cd / && find . -type f | grep libprotobuf.so.10

假设在`/home/user/mydir`

3. 设置 protobuf.so 的搜索目录
打开`~/.bashrc`，在末尾增加
> export LD_LIBRARY_PATH=/home/user/mydir:$LD_LIBRARY_PATH

4. 让配置生效
> source ~/.bashrc

#### 如果以上办法不行的话，尝试源码安装 protobuf

1. 首先在 [protobuf/releases](https://github.com/protocolbuffers/protobuf/releases/tag/v3.10.0) 下载所需的 pb 版本，例如需要 v3.10.0 。注意要下载 -cpp 后缀的压缩包。

2. 解压到某一目录，然后编译
>  tar xvf protobuf-cpp-3.10.0.tar.gz && cd protobuf-3.10.0/
./configure --prefix=/your_install_dir && make -j 3 && make install

3. **不不不要**忽略`--prefix`直接安装到系统目录，源码编译好的 so 和头文件在`your_install_dir`里

4. 设置 protobuf.so 的搜索目录
打开`~/.bashrc`，在末尾增加

```bash
export LD_LIBRARY_PATH=/your_install_dir/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/your_install_dir/include:$CPLUS_INCLUDE_PATH
```

5. 让配置生效
> source ~/.bashrc

#### 如果以上办法还不行
尝试删除已有protobuf（注意不要删到系统自带的，新手请谨慎），然后用以下命令重装所需的 so
> sudo apt-get install --reinstall libprotobuf8

版本号需改为自己的版本号

### Windows 出现此类问题，基本思路也是 IDE 改环境变量

### 行走江湖必备
关于环境变量设置、工具和技巧，强烈建议学习下 https://missing.csail.mit.edu/ 
