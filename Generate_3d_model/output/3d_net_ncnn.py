import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 10, 20, 30, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
         net.load_param("ncnn_model/3d_net.param")
         net.load_model("ncnn_model/3d_net.bin")

         with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)
