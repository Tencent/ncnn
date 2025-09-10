import numpy as np
import struct

def generate_npy_file(filename, data, use_bracket=True, fortran_order=False):
    """
    生成自定义npy文件，可控制shape的括号格式和内存顺序
    
    参数:
        filename: 输出文件名
        data: numpy数组数据
        use_bracket: True用[]，False用()
        fortran_order: 是否使用Fortran顺序（列优先）
    """
    # 获取数据类型描述符（如float32对应'<f4'）
    descr = data.dtype.descr[0][1]
    
    # 构造头信息字典
    header_dict = {
        "descr": descr,
        "fortran_order": fortran_order,
        "shape": data.shape if not use_bracket else list(data.shape)
    }
    
    # 转换为字符串，处理引号和布尔值
    header_str = str(header_dict).replace(" ", "").replace("'", '"').replace("False", "false").replace("True", "true")
    
    # 填充为4的倍数长度
    pad_len = (4 - (len(header_str) % 4)) % 4
    header_str += " " * pad_len
    
    # 构造npy文件二进制内容
    magic = b"\x93NUMPY"               # 魔法数
    version = b"\x01\x00"              # 版本1.0
    header_len = struct.pack("<I", len(header_str))  # 头长度（小端）
    npy_bytes = magic + version + header_len + header_str.encode("ascii") + data.tobytes()
    
    # 写入文件
    with open(filename, "wb") as f:
        f.write(npy_bytes)
    print(f"生成测试文件: {filename}，shape={data.shape}，dtype={data.dtype}，括号={'[]' if use_bracket else '()'}")

# 1. 基础测试（1D数组，方括号）
data1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
generate_npy_file("test_1d_bracket.npy", data1, use_bracket=True)

# 2. 基础测试（1D数组，圆括号）
generate_npy_file("test_1d_parenthesis.npy", data1, use_bracket=False)

# 3. 2D数组测试（方括号）
data2 = np.random.rand(3, 4).astype(np.float32)
generate_npy_file("test_2d_bracket.npy", data2, use_bracket=True)

# 4. 2D数组测试（圆括号）
generate_npy_file("test_2d_parenthesis.npy", data2, use_bracket=False)

# 5. 3D数组测试（方括号）
data3 = np.random.rand(2, 3, 4).astype(np.float32)
generate_npy_file("test_3d_bracket.npy", data3, use_bracket=True)

# 6. 4D数组（ResNet输入格式，圆括号）
data4 = np.random.rand(1, 3, 224, 224).astype(np.float32)
generate_npy_file("test_4d_parenthesis.npy", data4, use_bracket=False)

# 7. Fortran顺序（列优先）测试
data5 = np.random.rand(2, 2).astype(np.float32)
generate_npy_file("test_fortran_order.npy", data5, use_bracket=True, fortran_order=True)

# 8. 带空格的shape字符串（测试空格兼容性）
# 手动构造特殊头信息（带空格的shape）
def generate_spaced_shape_npy():
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    descr = data.dtype.descr[0][1]
    # 构造带空格的shape字符串
    header_str = f'{{"descr":"{descr}","fortran_order":false,"shape": [ 2 , 2 ]}}'
    pad_len = (4 - (len(header_str) % 4)) % 4
    header_str += " " * pad_len
    
    magic = b"\x93NUMPY"
    version = b"\x01\x00"
    header_len = struct.pack("<I", len(header_str))
    npy_bytes = magic + version + header_len + header_str.encode("ascii") + data.tobytes()
    
    with open("test_spaced_shape.npy", "wb") as f:
        f.write(npy_bytes)
    print("生成测试文件: test_spaced_shape.npy（带空格的shape）")

generate_spaced_shape_npy()

print("\n所有测试文件生成完成！共8个测试文件")
    
