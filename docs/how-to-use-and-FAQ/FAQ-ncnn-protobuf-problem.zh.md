### 编译 `caffe2ncnn` 时报 `Protobuf not found`

一般是因为 protobuf 未安装或环境变量未设置

##### Ubuntu 系统尝试以下命令
> sudo apt-get install libprotobuf-dev

##### CentOS 尝试
> sudo yum install libprotobuf-dev

##### 然后设置 C++ 环境
打开`~/.bashrc`，在末尾增加
> export LD_LIBRARY_PATH=${YOUR_PROTOBUF_LIB_PATH}:$LD_LIBRARY_PATH

##### 让配置生效
> source ~/.bashrc


### 编译 `caffe2ncnn` 时报 protoc 和 protobuf.so 版本不匹配

一般是因为系统安装了不止一个 protobuf。

##### 设置 C++ so 的目录和 protobuf 版本一致
打开`~/.bashrc`，在末尾增加
> export LD_LIBRARY_PATH=${YOUR_PROTOBUF_LIB_PATH}:$LD_LIBRARY_PATH

##### 让配置生效
> source ~/.bashrc

如果以上办法不行的话，尝试删除已有protobuf（注意不要删到系统自带的，新手请谨慎），然后用以下命令重装
> sudo apt-get install --reinstall libprotobuf8
版本号需改为自己的版本号


### 行走江湖必备
关于环境变量设置、工具和技巧，强烈建议学习下 https://missing.csail.mit.edu/ 
