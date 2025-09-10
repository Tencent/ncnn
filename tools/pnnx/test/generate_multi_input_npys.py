# generate_multi_input_npys.py
import numpy as np

# 输入1：模拟图像输入（4维，shape [1,3,224,224]，float32）
input1 = np.random.rand(1, 3, 224, 224).astype(np.float32)
np.save("multi_input1_image.npy", input1)
print(f"生成多输入1：multi_input1_image.npy，shape={input1.shape}，dtype={input1.dtype}")

# 输入2：模拟掩码输入（3维，shape [1,224,224]，float32）
input2 = np.random.rand(1, 224, 224).astype(np.float32)
np.save("multi_input2_mask.npy", input2)
print(f"生成多输入2：multi_input2_mask.npy，shape={input2.shape}，dtype={input2.dtype}")
