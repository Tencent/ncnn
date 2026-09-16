import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 8, 8, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        net.load_param("/home/edwards/tx_opensource/ncnn/tools/pnnx/tests/fixtures/pt2_schema/schema_8_20.ncnn.param")
        net.load_model("/home/edwards/tx_opensource/ncnn/tools/pnnx/tests/fixtures/pt2_schema/schema_8_20.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.numpy(), batch_index=0).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(out0.numpy(batch_index=0)))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    print(test_inference())
