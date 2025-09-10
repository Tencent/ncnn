# test_binary_write_read.py
import struct

# 1. 写入固定二进制数据（10个float32值：1.0,2.0,...,10.0）
data = [1.0 + i for i in range(10)]
binary_data = struct.pack("<10f", *data)  # <10f：little-endian，10个float32

# 写入文件
with open("test_binary.bin", "wb") as f:
    f.write(binary_data)
print(f"✅ 已写入二进制文件：test_binary.bin（大小：{len(binary_data)} 字节）")

# 2. 读取文件并验证
with open("test_binary.bin", "rb") as f:
    read_binary = f.read()
    read_data = struct.unpack("<10f", read_binary)

# 对比写入与读取的数据
is_consistent = all(abs(a - b) < 1e-6 for a, b in zip(data, read_data))
if is_consistent:
    print("✅ 写入与读取的数据完全一致！")
    print(f"   读取结果：{[round(x,1) for x in read_data]}")
else:
    print("❌ 写入与读取的数据不一致！环境存在二进制写入问题")
    print(f"   预期：{data}")
    print(f"   实际：{read_data}")
