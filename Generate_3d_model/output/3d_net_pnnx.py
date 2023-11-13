import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv3d = nn.Conv3d(bias=True, dilation=(1,1,1), groups=1, in_channels=3, kernel_size=(2,2,2), out_channels=6, padding=(0,0,0), padding_mode='zeros', stride=(2,2,2))
        self.conv3d_depth_wise = nn.Conv3d(bias=True, dilation=(1,1,1), groups=6, in_channels=6, kernel_size=(3,3,3), out_channels=6, padding=(1,1,1), padding_mode='zeros', stride=(1,1,1))
        self.deconv3d = nn.ConvTranspose3d(bias=True, dilation=(1,1,1), groups=1, in_channels=6, kernel_size=(3,3,3), out_channels=6, output_padding=(0,0,0), padding=(1,1,1), stride=(1,1,1))
        self.deconv3d_depth_wise = nn.ConvTranspose3d(bias=True, dilation=(1,1,1), groups=6, in_channels=6, kernel_size=(3,3,3), out_channels=6, output_padding=(0,0,0), padding=(1,1,1), stride=(1,1,1))

        archive = zipfile.ZipFile('output/3d_net.pnnx.bin', 'r')
        self.conv3d.bias = self.load_pnnx_bin_as_parameter(archive, 'conv3d.bias', (6), 'float32')
        self.conv3d.weight = self.load_pnnx_bin_as_parameter(archive, 'conv3d.weight', (6,3,2,2,2), 'float32')
        self.conv3d_depth_wise.bias = self.load_pnnx_bin_as_parameter(archive, 'conv3d_depth_wise.bias', (6), 'float32')
        self.conv3d_depth_wise.weight = self.load_pnnx_bin_as_parameter(archive, 'conv3d_depth_wise.weight', (6,1,3,3,3), 'float32')
        self.deconv3d.bias = self.load_pnnx_bin_as_parameter(archive, 'deconv3d.bias', (6), 'float32')
        self.deconv3d.weight = self.load_pnnx_bin_as_parameter(archive, 'deconv3d.weight', (6,6,3,3,3), 'float32')
        self.deconv3d_depth_wise.bias = self.load_pnnx_bin_as_parameter(archive, 'deconv3d_depth_wise.bias', (6), 'float32')
        self.deconv3d_depth_wise.weight = self.load_pnnx_bin_as_parameter(archive, 'deconv3d_depth_wise.weight', (6,1,3,3,3), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.conv3d(v_0)
        v_2 = self.conv3d_depth_wise(v_1)
        v_3 = self.deconv3d(v_2)
        v_4 = self.deconv3d_depth_wise(v_3)
        return v_4

def export_torchscript():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 10, 20, 30, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("output/3d_net_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 10, 20, 30, dtype=torch.float)

    torch.onnx._export(net, v_0, "output/3d_net_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def test_inference():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 10, 20, 30, dtype=torch.float)

    return net(v_0)
