# ncnn MicroPython 模块

ncnn 神经网络推理框架的 MicroPython 绑定。

## 文件说明

- `ncnn_module.c` - 主要C模块实现
- `micropython.mk` - MicroPython编译配置
- `micropython.cmake` - CMake编译配置
- `test/test_api.py` - API完整测试

## 编译步骤

### 1. 编译ncnn库
```bash
cd /path/to/ncnn
mkdir build && cd build  
cmake -DNCNN_C_API=ON -DNCNN_STDIO=ON -DNCNN_STRING=ON ..
make -j4
```

### 2. 获取MicroPython源码
```bash
git clone https://github.com/micropython/micropython.git
cd micropython  
git submodule update --init
make -C mpy-cross
```

### 3. 编译包含ncnn模块的MicroPython
```bash
cd micropython/ports/unix
make USER_C_MODULES=/path/to/ncnn/modules -j4
```

## 已实现的API

### 模块函数
- `ncnn.version()` - 获取版本信息
- `ncnn.init()` - 初始化模块

### Mat类
- `Mat()`, `Mat(w)`, `Mat(w,h)`, `Mat(w,h,c)`, `Mat(w,h,d,c)` - 构造函数
- `dims()` - 获取维度数
- `w()`, `h()`, `c()` - 获取宽度、高度、通道数
- `fill(value)` - 填充数值

### Net类
- `Net()` - 构造函数
- `load_param(path)` - 加载参数文件
- `load_model(path)` - 加载模型文件
- `create_extractor()` - 创建提取器

### Extractor类
- `input(index, mat)` - 设置输入
- `extract(index)` - 提取输出

## 测试运行

```bash
/path/to/built/micropython test/test_api.py
```

## 使用示例

```python
import ncnn

# 初始化
ncnn.init()

# 创建张量
mat = ncnn.Mat(224, 224, 3)
mat.fill(0.5)

# 创建网络
net = ncnn.Net()

# 创建提取器
ex = net.create_extractor()
```