#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import torch


PNNX = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]]


def torch_version_tuple():
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        raise AssertionError("cannot parse torch version %r" % torch.__version__)
    return tuple(int(value) for value in match.groups())


def save_exported_program(model, example_inputs, archive_path):
    program = torch.export.export(model.eval(), example_inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.export.save(program, archive_path)


def run_pnnx(work_dir, archive_path):
    return subprocess.run(
        [str(PNNX), archive_path.name],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def read_model_document(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        paths = [
            name for name in archive.namelist() if name.endswith("/models/model.json")
        ]
        if len(paths) != 1:
            raise AssertionError("expected one model.json, found %r" % paths)
        return json.loads(archive.read(paths[0]))


def load_generated_output(work_dir, basename):
    module_path = work_dir / (basename + "_pnnx.py")
    spec = importlib.util.spec_from_file_location(
        "test_pt2_version_compatibility_" + basename, module_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load generated module %s" % module_path)

    module = importlib.util.module_from_spec(spec)
    previous_work_dir = Path.cwd()
    try:
        os.chdir(work_dir)
        spec.loader.exec_module(module)
        return module.test_inference()
    finally:
        os.chdir(previous_work_dir)


class CompatibilityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(2, 1, 3, 3), requires_grad=False
        )

    def forward(self, x):
        added = torch.ops.aten.add.Tensor(x, x)
        flattened = torch.ops.aten.flatten.using_ints(added, 1)
        convolved = torch.ops.aten.conv2d.default(
            x, self.weight, None, [1, 1], [0, 0]
        )
        return flattened, convolved


class Pt2VersionCompatibilityTest(unittest.TestCase):
    def test_real_producer_archive(self):
        model = CompatibilityModel().eval()
        example_inputs = (torch.ones(1, 1, 5, 5),)

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "compatibility.pt2"
            save_exported_program(model, example_inputs, archive_path)
            result = run_pnnx(work_dir, archive_path)
            diagnostic = result.stderr.decode(errors="replace")

            if torch_version_tuple() < (2, 9):
                self.assertGreater(result.returncode, 0, diagnostic)
                self.assertIn(
                    "PyTorch 2.8 legacy pickled-payload PT2 is unsupported",
                    diagnostic,
                )
                return

            document = read_model_document(archive_path)
            self.assertEqual(document["schema_version"]["major"], 8)
            self.assertIsInstance(document["schema_version"]["minor"], int)
            self.assertIsInstance(document["opset_version"]["aten"], int)

            serialized_nodes = {
                node["target"]: [argument["name"] for argument in node["inputs"]]
                for node in document["graph_module"]["graph"]["nodes"]
            }
            self.assertEqual(
                serialized_nodes["torch.ops.aten.add.Tensor"], ["self", "other"]
            )
            self.assertEqual(
                serialized_nodes["torch.ops.aten.flatten.using_ints"],
                ["self", "start_dim"],
            )
            self.assertEqual(
                serialized_nodes["torch.ops.aten.conv2d.default"],
                ["input", "weight"],
            )
            self.assertEqual(result.returncode, 0, diagnostic)

            torch.manual_seed(0)
            expected = model(torch.rand(1, 1, 5, 5))
            actual = load_generated_output(work_dir, archive_path.stem)
            self.assertEqual(len(actual), len(expected))
            for expected_tensor, actual_tensor in zip(expected, actual):
                self.assertTrue(
                    torch.allclose(expected_tensor, actual_tensor, rtol=1e-4, atol=1e-4),
                    "generated output mismatch\nexpected=%s\nactual=%s"
                    % (expected_tensor, actual_tensor),
                )


if __name__ == "__main__":
    unittest.main()
