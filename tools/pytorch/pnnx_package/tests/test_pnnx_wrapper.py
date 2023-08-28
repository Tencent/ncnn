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
    pnnx.run(torchscript_path, x)




if __name__ == "__main__":
    test_pnnx_export()
    # test_pnnx_run()
