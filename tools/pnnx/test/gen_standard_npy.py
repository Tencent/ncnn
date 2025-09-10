import numpy as np
import os  # 导入Python标准库os，用于获取文件大小

# 生成与resnet18输入完全匹配的张量：shape=(1,3,224,224)，dtype=float32
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
# 保存为标准npy 1.0格式
np.save("standard_input.npy", input_data)

# 验证文件信息
saved_data = np.load("standard_input.npy")
print(f"生成成功！文件：standard_input.npy")
print(f"Shape: {saved_data.shape}, Dtype: {saved_data.dtype}")
print(f"文件大小: {os.path.getsize('standard_input.npy')} bytes")  # 用os.path.getsize，而非np.os
