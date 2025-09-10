import numpy as np

# 生成第一个输入文件: (1, 3, 224, 224) float32
input1 = np.random.rand(1, 3, 224, 224).astype(np.float32)
np.save("test_input1.npy", input1, allow_pickle=False)
print(f"生成 test_input1.npy: 形状 {input1.shape}, 类型 {input1.dtype}")

# 生成第二个输入文件: (1, 128, 32) float32
input2 = np.random.rand(1, 128, 32).astype(np.float32)
np.save("test_input2.npy", input2, allow_pickle=False)
print(f"生成 test_input2.npy: 形状 {input2.shape}, 类型 {input2.dtype}")

# 验证文件可读取
try:
    loaded1 = np.load("test_input1.npy")
    loaded2 = np.load("test_input2.npy")
    print("文件验证成功:")
    print(f"  test_input1.npy 加载后形状: {loaded1.shape}")
    print(f"  test_input2.npy 加载后形状: {loaded2.shape}")
except Exception as e:
    print(f"文件生成错误: {e}")

