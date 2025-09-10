import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=16, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.fc1 = nn.Linear(bias=True, in_features=65536, out_features=128)
        self.fc2 = nn.Linear(bias=True, in_features=100, out_features=128)
        self.fc3 = nn.Linear(bias=True, in_features=256, out_features=10)

        archive = zipfile.ZipFile('two_input_model.pnnx.bin', 'r')
        self.conv1.bias = self.load_pnnx_bin_as_parameter(archive, 'conv1.bias', (16), 'float32')
        self.conv1.weight = self.load_pnnx_bin_as_parameter(archive, 'conv1.weight', (16,3,3,3), 'float32')
        self.fc1.bias = self.load_pnnx_bin_as_parameter(archive, 'fc1.bias', (128), 'float32')
        self.fc1.weight = self.load_pnnx_bin_as_parameter(archive, 'fc1.weight', (128,65536), 'float32')
        self.fc2.bias = self.load_pnnx_bin_as_parameter(archive, 'fc2.bias', (128), 'float32')
        self.fc2.weight = self.load_pnnx_bin_as_parameter(archive, 'fc2.weight', (128,100), 'float32')
        self.fc3.bias = self.load_pnnx_bin_as_parameter(archive, 'fc3.bias', (10), 'float32')
        self.fc3.weight = self.load_pnnx_bin_as_parameter(archive, 'fc3.weight', (10,256), 'float32')
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

    def forward(self, v_0, v_1):
        v_2 = self.conv1(v_0)
        v_3 = v_2.view(1, 65536)
        v_4 = self.fc1(v_3)
        v_5 = self.fc2(v_1)
        v_6 = v_5.view(1, 128)
        v_7 = torch.cat((v_4, v_6), dim=1)
        v_8 = self.fc3(v_7)
        return v_8

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 64, 64, dtype=torch.float)
    v_1 = torch.rand(1, 1, 100, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1))
    mod.save("two_input_model_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 64, 64, dtype=torch.float)
    v_1 = torch.rand(1, 1, 100, dtype=torch.float)

    torch.onnx.export(net, (v_0, v_1), "two_input_model_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1'], output_names=['out0'])

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 64, 64, dtype=torch.float)
    v_1 = torch.rand(1, 1, 100, dtype=torch.float)

    return net(v_0, v_1)

if __name__ == "__main__":
    print(test_inference())
