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


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        channels = 64
        self.conv0 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, 8, 3, padding=1)
        keep = (torch.arange(channels) % 3 != 1).reshape(1, channels, 1, 1)
        self.register_buffer("keep", keep)

    def forward(self, x):
        return self.conv1(self.conv0(x) * self.keep)


def test():
    if pt2_producer_status() != SUPPORTED:
        print("skip unsupported pt2 producer", torch.__version__)
        raise SystemExit(PT2_SKIP_RETURN_CODE)

    torch.manual_seed(0)
    net = Model().eval()
    x = torch.randn(1, 3, 8, 8)
    expected = net(x)[0]

    pnnx = Path(os.environ.get("PNNX_TEST_PNNX", "../../src/pnnx")).resolve()

    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        archive_path = work_dir / "bool_attribute.pt2"
        input_path = work_dir / "in0.npy"

        exported_program = torch.export.export(net, (x,))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.export.save(exported_program, archive_path)
        np.save(input_path, x.numpy())

        result = subprocess.run(
            [str(pnnx), archive_path.name, "input=in0.npy", "fp16=0"],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            return False

        with ncnn.Net() as ncnn_net:
            if ncnn_net.load_param(str(work_dir / "bool_attribute.ncnn.param")) != 0:
                return False
            if ncnn_net.load_model(str(work_dir / "bool_attribute.ncnn.bin")) != 0:
                return False

            with ncnn_net.create_extractor() as ex:
                ex.input("in0", ncnn.Mat(np.ascontiguousarray(x.numpy()[0])))
                ret, out = ex.extract("out0")
                if ret != 0:
                    return False
                actual = torch.from_numpy(np.array(out))

    if not torch.allclose(expected, actual, rtol=1e-4, atol=1e-4):
        print("max abs diff", (expected - actual).abs().max())
        return False

    return True


if __name__ == "__main__":
    if test():
        sys.exit(0)
    sys.exit(1)
