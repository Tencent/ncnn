import numpy as np
import struct

# 1. 生成输入数据（resnet18输入：1,3,224,224 float32）
data = np.random.randn(1, 3, 224, 224).astype(np.float32)
data_bytes = data.tobytes()  # 转为二进制数据

# 2. 构造npy头信息（JSON格式，严格符合npy规范）
header_dict = {
    "descr": "<f4",          # 数据类型：little-endian float32
    "fortran_order": False,  # C风格内存顺序（行优先）
    "shape": data.shape      # 数据形状：(1,3,224,224)
}
header_str = str(header_dict).replace(" ", "").replace("'", '"')  # 去除空格，单引号转双引号
# 头信息需用空格填充到4的倍数（npy规范要求）
pad_len = (4 - (len(header_str) % 4)) % 4
header_str += " " * pad_len

# 3. 构造npy文件完整二进制内容
# 3.1 魔法数（6字节）：0x93 0x4E 0x55 0x4D 0x50 0x59
magic = b"\x93NUMPY"
# 3.2 版本号（2字节）：1.0
version = b"\x01\x00"
# 3.3 头长度（4字节，little-endian）：header_str的长度
header_len = struct.pack("<I", len(header_str))  # <I 表示little-endian uint32
# 3.4 拼接所有部分
npy_bytes = magic + version + header_len + header_str.encode("ascii") + data_bytes

# 4. 保存为文件
with open("raw_standard_input.npy", "wb") as f:
    f.write(npy_bytes)

print("✅ 绝对标准的npy文件生成完成：raw_standard_input.npy")
print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
print(f"头信息: {header_str}")
print(f"头长度: {len(header_str)} 字节（符合4的倍数要求）")
