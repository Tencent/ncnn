#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

import ncnn
import numpy as np
import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnnx_test_utils import SUPPORTED
from pnnx_test_utils import PT2_SKIP_RETURN_CODE
from pnnx_test_utils import pt2_producer_status
from pnnx_test_utils import _import_generated_module


class Model(nn.Module):
    def __init__(self, dtype=torch.bool):
        super().__init__()

        channels = 64
        self.conv0 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, 8, 3, padding=1)
        keep = (torch.arange(channels) % 3 != 1).reshape(1, channels, 1, 1)
        if dtype != torch.bool:
            keep = torch.linspace(-1.25, 1.75, channels, dtype=dtype).reshape(1, channels, 1, 1)
        if dtype == torch.float64:
            self.conv0.double()
            self.conv1.double()
        self.register_buffer("keep", keep)

    def forward(self, x):
        return self.conv1(self.conv0(x) * self.keep)


def test_attribute(dtype, fp16):
    print("attribute dtype=%s fp16=%d" % (dtype, fp16), flush=True)
    if pt2_producer_status() != SUPPORTED:
        print("skip unsupported pt2 producer", torch.__version__)
        raise SystemExit(PT2_SKIP_RETURN_CODE)

    torch.manual_seed(0)
    net = Model(dtype).eval()
    x = torch.randn(1, 3, 8, 8).to(net.conv0.weight.dtype)
    expected = net(x)[0]

    pnnx = Path(os.environ.get("PNNX_TEST_PNNX", "../../src/pnnx")).resolve()

    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        archive_path = work_dir / "attribute.pt2"
        input_path = work_dir / "in0.npy"

        exported_program = torch.export.export(net, (x,))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.export.save(exported_program, archive_path)
        np.save(input_path, x.numpy())

        result = subprocess.run(
            [str(pnnx), archive_path.name, "input=in0.npy", "fp16=%d" % fp16],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            return False

        previous_dir = Path.cwd()
        try:
            os.chdir(work_dir)
            module = _import_generated_module(work_dir / "attribute_pnnx.py", "attribute")
            with torch.no_grad():
                converted_net = module.Model().eval()
                keep = next(p for p in converted_net.parameters() if p.shape == net.keep.shape)
                torch.testing.assert_close(keep, net.keep)
                converted = converted_net(x)[0]
            torch.testing.assert_close(converted, expected)
        finally:
            os.chdir(previous_dir)

        with ncnn.Net() as ncnn_net:
            if ncnn_net.load_param(str(work_dir / "attribute.ncnn.param")) != 0:
                return False
            if ncnn_net.load_model(str(work_dir / "attribute.ncnn.bin")) != 0:
                return False

            with ncnn_net.create_extractor() as ex:
                input_data = np.ascontiguousarray(x.float().numpy()[0])
                ex.input("in0", ncnn.Mat(input_data).clone())
                ret, out = ex.extract("out0")
                if ret != 0:
                    return False
                actual = torch.from_numpy(np.array(out))

    tolerance = 1e-3 if fp16 else 1e-4
    if not torch.allclose(expected.float(), actual, rtol=tolerance, atol=tolerance):
        print("max abs diff", (expected - actual).abs().max())
        return False

    return True


def test():
    return all(test_attribute(dtype, fp16) for dtype, fp16 in
               ((torch.bool, 0), (torch.float64, 0), (torch.float64, 1),
                (torch.bfloat16, 0), (torch.bfloat16, 1)))


if __name__ == "__main__":
    if test():
        sys.exit(0)
    sys.exit(1)
