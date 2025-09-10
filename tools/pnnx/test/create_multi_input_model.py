# create_multi_input_model.py（生成双输入PyTorch模型）
import torch
import torch.nn as nn

# 双输入模型：输入1（图像[1,3,224,224]）+ 输入2（掩码[1,224,224]）
class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 处理图像输入
        self.conv2 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 处理掩码输入
        self.fc = nn.Linear(16*224*224*2, 10)  # 拼接两个输入的特征后分类

    def forward(self, x1, x2):  # x1: 图像输入，x2: 掩码输入
        x1 = self.conv1(x1)  # [1,3,224,224] → [1,16,224,224]
        x2 = self.conv2(x2.unsqueeze(1))  # [1,224,224] → [1,1,224,224] → [1,16,224,224]
        x = torch.cat([x1, x2], dim=1)  # 拼接 → [1,32,224,224]
        x = x.view(x.size(0), -1)  # 展平 → [1, 32*224*224]
        x = self.fc(x)  # 分类 → [1,10]
        return x

# 生成并保存模型
model = MultiInputModel()
dummy_x1 = torch.rand(1, 3, 224, 224)  # 匹配 multi_input1_image.npy
dummy_x2 = torch.rand(1, 224, 224)     # 匹配 multi_input2_mask.npy
# 用trace保存（pnnx支持trace生成的模型）
traced_model = torch.jit.trace(model, (dummy_x1, dummy_x2))
traced_model.save("multi_input_model.pt")
print("生成双输入模型：multi_input_model.pt")
