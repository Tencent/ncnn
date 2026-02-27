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

class TopK(nn.Module):
    def __init__(self, axis=1, largest=1, sorted=1):
        super(TopK, self).__init__()
        self.axis = axis
        self.largest = largest
        self.sorted = sorted
    def forward(self, x, k):
        # Torch topk returns (values, indices)
        return torch.topk(x, k.item() if hasattr(k, 'item') else k, dim=self.axis, largest=bool(self.largest), sorted=bool(self.sorted))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.TopK_0 = TopK(axis=1, largest=1, sorted=1)
        self.TopK_1 = TopK(axis=3, largest=0, sorted=1)
        self.TopK_2 = TopK(axis=0, largest=1, sorted=1)

        archive = zipfile.ZipFile('test_torch_topk.pnnx.bin', 'r')
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

    def forward(self, v_0, v_1, v_2):
        v_3 = 2
        v_4, v_5 = self.TopK_0(v_0, v_3)
        v_6 = 4
        v_7, v_8 = self.TopK_1(v_1, v_6)
        v_9 = 3
        v_10, v_11 = self.TopK_2(v_2, v_9)
        return v_4, v_5, v_7, v_8, v_10, v_11

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 16, dtype=torch.float)
    v_1 = torch.rand(1, 5, 9, 11, dtype=torch.float)
    v_2 = torch.rand(14, 8, 5, 9, 10, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1, v_2))
    mod.save("test_torch_topk_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 16, dtype=torch.float)
    v_1 = torch.rand(1, 5, 9, 11, dtype=torch.float)
    v_2 = torch.rand(14, 8, 5, 9, 10, dtype=torch.float)

    torch.onnx.export(net, (v_0, v_1, v_2), "test_torch_topk_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1', 'in2'], output_names=['out0', 'out1', 'out2', 'out3', 'out4', 'out5'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 16, dtype=torch.float)
    v_1 = torch.rand(1, 5, 9, 11, dtype=torch.float)
    v_2 = torch.rand(14, 8, 5, 9, 10, dtype=torch.float)

    import pnnx
    pnnx.export(net, "test_torch_topk_pnnx.py.pt", (v_0, v_1, v_2))

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 16, dtype=torch.float)
    v_1 = torch.rand(1, 5, 9, 11, dtype=torch.float)
    v_2 = torch.rand(14, 8, 5, 9, 10, dtype=torch.float)

    return net(v_0, v_1, v_2)

if __name__ == "__main__":
    print(test_inference())
