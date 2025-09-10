# rebuild_npy.py
import struct

def build_valid_npy(file_path, shape):
    # 1. 计算总元素数
    total = 1
    for d in shape:
        total *= d
    
    # 2. 构建标准头信息（JSON 格式）
    header = f'{{"descr":"<f4","fortran_order":false,"shape":{shape}}}\n'
    # 3. 头信息 16 字节对齐
    pad = (16 - len(header) % 16) % 16
    header += ' ' * pad
    header_len = len(header)
    
    # 4. 手动写入二进制（逐字段控制）
    with open(file_path, 'wb') as f:
        f.write(b'\x93NUMPY')          # 魔法数
        f.write(b'\x01\x00')          # 版本号
        f.write(struct.pack('<I', header_len))  # 头长度（小端）
        f.write(header.encode())      # 头内容
        f.write(struct.pack('<' + 'f'*total, *(1.0 for _ in range(total))))  # 数据（全1.0，便于验证）
    print(f"✅ 生成有效文件: {file_path}，头长度: {header_len} 字节")

# 生成模型需要的两个输入文件
build_valid_npy("valid_input1.npy", (1, 3, 64, 64))
build_valid_npy("valid_input2.npy", (1, 1, 100))
