# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import zipfile

import torch
import torch.nn as nn

from pnnx_test_utils import import_model, model_formats, run_pnnx


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("offset", torch.rand(2))
        self.constant = torch.rand(2)

    def forward(self, x):
        return torch.relu(self.linear(x) + self.offset) * self.constant


def test():
    if "pt2" not in model_formats():
        return True

    try:
        from torch._export.serde.schema import SCHEMA_VERSION
        from torch._export.serde.serialize import serialize
    except ImportError:
        print("UNSUPPORTED_BY_TORCH_EXPORT: legacy serializer is unavailable")
        return True

    torch.manual_seed(0)
    net = Model().eval()
    x = torch.rand(1, 3)
    exported_program = torch.export.export(net, (x,))
    artifact = serialize(exported_program)

    model_path = "test_pnnx_exported_program_legacy.pt2"
    with zipfile.ZipFile(model_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("serialized_exported_program.json", artifact.exported_program)
        archive.writestr("serialized_state_dict.pt", artifact.state_dict)
        archive.writestr("serialized_constants.pt", artifact.constants)
        archive.writestr("serialized_example_inputs.pt", artifact.example_inputs)
        archive.writestr("version", ".".join(map(str, SCHEMA_VERSION)))

    output_prefix = "test_pnnx_exported_program_legacy"
    if run_pnnx(model_path, output_prefix).returncode != 0:
        return False

    expected = torch.export.load(model_path).module()(x)
    converted = import_model(output_prefix + "_pnnx.py", output_prefix + "_pnnx")(x)
    return torch.equal(expected, converted)


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
