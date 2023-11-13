import os
import torch
nn = torch.nn

class triDmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=6, kernel_size=2, stride=2, padding=0)
        self.conv3d_depth_wise = nn.Conv3d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, groups=6)
        self.deconv3d = nn.ConvTranspose3d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.deconv3d_depth_wise = nn.ConvTranspose3d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, groups=6)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.conv3d_depth_wise(x)
        x = self.deconv3d(x)
        x = self.deconv3d_depth_wise(x)
        return x

if __name__ == "__main__":
    input = torch.randn(1, 3, 10, 20, 30)
    model = triDmodel()
    output = model(input)

    mod = torch.jit.trace(model, input)
    os.system("mkdir -p output")
    os.system("mkdir -p ncnn_model")
    mod.save("output/3d_net.pt")
    os.system("sh 2ncnn.sh")

