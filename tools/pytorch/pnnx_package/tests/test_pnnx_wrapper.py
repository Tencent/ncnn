if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import pnnx
import torch
import torchvision
import torch.nn as nn


def test_pnnx_export():
    
    resnet_test_model = torchvision.models.resnet18()
    x = torch.rand(1, 3, 224, 224)
    pnnx.export(resnet_test_model, x)


def test_pnnx_run():

    torchscript_path = "resnet_test_model.pt"

    x = torch.rand(1, 3, 224, 224)
    y = torch.rand(1, 3)
    z = torch.rand(1, 3, 224, 224)
    x2 = torch.rand(1, 3)
    y2 = torch.rand(1, 3, 224, 224)
    z2 = torch.rand(1, 3)

    # pnnx.run(torchscript_path, x)
    # pnnx.run(torchscript_path, x, pnnxparam="model.pnnx.param",  device= "cpu",  moduleop =("models.common.Focus","models.yolo.Detect"))
    pnnx.run(torchscript_path, inputshape=(x,y,z), inputshape2=(x2,y2,z2), optlevel=0, pnnxparam="model.pnnx.param",  device= "cpu",  moduleop =("models.common.Focus","models.yolo.Detect"))



if __name__ == "__main__":
    # test_pnnx_export()
    test_pnnx_run()
