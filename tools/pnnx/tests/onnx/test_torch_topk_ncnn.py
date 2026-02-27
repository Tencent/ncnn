import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 16, dtype=torch.float)
    in1 = torch.rand(1, 5, 9, 11, dtype=torch.float)
    in2 = torch.rand(14, 8, 5, 9, 10, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        net.load_param("test_torch_topk.ncnn.param")
        net.load_model("test_torch_topk.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.numpy()).clone())
            ex.input("in1", ncnn.Mat(in1.numpy()).clone())
            ex.input("in2", ncnn.Mat(in2.numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)))
            _, out1 = ex.extract("out1")
            out.append(torch.from_numpy(np.array(out1)))
            _, out2 = ex.extract("out2")
            out.append(torch.from_numpy(np.array(out2)))
            _, out3 = ex.extract("out3")
            out.append(torch.from_numpy(np.array(out3)))
            _, out4 = ex.extract("out4")
            out.append(torch.from_numpy(np.array(out4)))
            _, out5 = ex.extract("out5")
            out.append(torch.from_numpy(np.array(out5)))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    print(test_inference())
