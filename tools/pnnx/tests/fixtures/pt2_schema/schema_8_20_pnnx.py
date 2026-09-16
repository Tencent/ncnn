# pnnx model stat
# model inputshape = [1,3,8,8]f32
# FLOPS = 14.592K
# memory OPS = 1.584K

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

# torch 2.x renamed torch.var/std unbiased to correction; the
# generated calls pick the right keyword at runtime
_torch_has_correction = int(torch.__version__.split('.')[0]) >= 2

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv2d_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=4, padding=(1,1), padding_mode='zeros', stride=(1,1))

        archive = zipfile.ZipFile('/tmp/tmp21x3yz4p/schema_8_20.pnnx.bin', 'r')
        self.conv2d_0.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_0.bias', (4), 'float32')
        self.conv2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_0.weight', (4,3,3,3), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        if dtype == 'bfloat16':
            # numpy has no native bfloat16; read the 2-byte raw words as int16 and reinterpret (bit-preserving)
            m = np.memmap(tmppath, dtype='int16', mode='r', shape=shape).copy()
            os.remove(tmppath)
            return torch.from_numpy(m).view(torch.bfloat16)
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.conv2d_0(v_0)
        v_2 = F.relu(v_1)
        v_3 = (v_2 + 1)
        return v_3

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 8, 8, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("/home/edwards/tx_opensource/ncnn/tools/pnnx/tests/fixtures/pt2_schema/schema_8_20_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 8, 8, dtype=torch.float)

    torch.onnx.export(net, v_0, "/home/edwards/tx_opensource/ncnn/tools/pnnx/tests/fixtures/pt2_schema/schema_8_20_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 8, 8, dtype=torch.float)

    import pnnx
    pnnx.export(net, "/home/edwards/tx_opensource/ncnn/tools/pnnx/tests/fixtures/pt2_schema/schema_8_20_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 8, 8, dtype=torch.float)

    return net(v_0)

def export_exported_program(example_inputs=None, out_path=None):
    net = Model()
    net.eval()

    if example_inputs is None:
        torch.manual_seed(0)
        v_0 = torch.rand(1, 3, 8, 8, dtype=torch.float)
        example_inputs = (v_0)

    ep = torch.export.export(net, example_inputs)
    if out_path is None:
        out_path = os.path.splitext(os.path.abspath(__file__))[0] + '_reexported.pt2'
    torch.export.save(ep, out_path)
    return ep

if __name__ == "__main__":
    print(test_inference())
