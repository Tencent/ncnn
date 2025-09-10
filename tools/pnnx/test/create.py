import torch
import torchvision.models as models

# 加载预训练ResNet18并转为TorchScript
model = models.resnet18(pretrained=True)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)  # 匹配ResNet18输入形状
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("resnet18.pt")  # 保存为TorchScript模型
