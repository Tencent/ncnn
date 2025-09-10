import torch
import torch.nn as nn
import numpy as np

# 定义双输入模型
class TwoInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 处理第一个输入 (1, 3, 64, 64)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 64 * 64, 128)
        
        # 处理第二个输入 (1, 1, 100)
        self.fc2 = nn.Linear(100, 128)
        
        # 融合特征并输出
        self.fc3 = nn.Linear(128 + 128, 10)  # 两个分支特征拼接

    def forward(self, x1, x2):
        # 处理第一个输入 (卷积+全连接)
        x1 = self.conv1(x1)                  # (1,3,64,64) → (1,16,64,64)
        x1 = x1.view(x1.size(0), -1)         # 展平 → (1, 16*64*64)
        x1 = self.fc1(x1)                    # → (1, 128)
        
        # 处理第二个输入 (全连接)
        x2 = self.fc2(x2)                    # (1,1,100) → (1,1,128)
        x2 = x2.view(x2.size(0), -1)         # 展平 → (1, 128)
        
        # 融合特征并输出
        x = torch.cat([x1, x2], dim=1)       # 拼接 → (1, 256)
        x = self.fc3(x)                      # 输出 → (1, 10)
        return x

# 生成模型并保存
if __name__ == "__main__":
    # 创建模型实例
    model = TwoInputModel()
    
    # 生成符合输入形状的示例数据
    input1 = torch.randn(1, 3, 64, 64)    # 第一个输入形状
    input2 = torch.randn(1, 1, 100)      # 第二个输入形状
    
    # 转换为TorchScript格式并保存
    traced_model = torch.jit.trace(model, (input1, input2))
    traced_model.save("two_input_model.pt")
    print("模型保存成功: two_input_model.pt")
    print(f"输入1形状要求: {input1.shape}")
    print(f"输入2形状要求: {input2.shape}")
    
    # 生成并保存对应的npy输入文件
    np.save("input1.npy", input1.numpy(), allow_pickle=False)
    np.save("input2.npy", input2.numpy(), allow_pickle=False)
    print("输入文件生成成功: input1.npy, input2.npy")

