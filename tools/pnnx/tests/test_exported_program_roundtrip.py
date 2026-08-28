#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch


PNNX = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]]
TORCH_VERSION = tuple(
    int(component) for component in torch.__version__.split("+", 1)[0].split(".")[:2]
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))


class DtypeIdentityModel(torch.nn.Module):
    def forward(self, token_ids, values, mask):
        return token_ids, values, mask


class StateDtypeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bf16", torch.tensor([1.5, -2.25], dtype=torch.bfloat16))
        self.register_buffer(
            "chalf", torch.tensor([1 + 2j, -3 + 0.5j], dtype=torch.complex32)
        )
        self.register_buffer("empty_bf16", torch.empty((0, 2), dtype=torch.bfloat16))
        self.register_buffer("empty_chalf", torch.empty((0, 2), dtype=torch.complex32))

    def forward(self, value):
        return self.bf16, self.chalf, self.empty_bf16, self.empty_chalf


class EmptyIntListModel(torch.nn.Module):
    def forward(self, value):
        return torch.tile(value, ())


class EmptyFloatListModel(torch.nn.Module):
    def forward(self, value):
        return torch.ops.aten._test_optional_floatlist(value, [])


def save_exported_program(model, example_inputs, archive_path):
    program = torch.export.export(model.eval(), example_inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.export.save(program, archive_path)


@unittest.skipIf(
    TORCH_VERSION < (2, 9),
    "modern exported program packages require PyTorch 2.9 or newer",
)
class ExportedProgramRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.work_dir = Path(self.temp_dir.name)

    def save(self, name, model, example_inputs):
        archive_path = self.work_dir / (name + ".pt2")
        save_exported_program(model, example_inputs, archive_path)
        return archive_path

    def call(self, function, *args):
        previous_work_dir = Path.cwd()
        try:
            os.chdir(self.work_dir)
            return function(*args)
        finally:
            os.chdir(previous_work_dir)

    def run_pnnx(self, archive_path):
        result = subprocess.run(
            [str(PNNX), archive_path.name],
            cwd=self.work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def convert(self, archive_path):
        self.run_pnnx(archive_path)

        module_path = self.work_dir / (archive_path.stem + "_pnnx.py")
        spec = importlib.util.spec_from_file_location(
            "test_exported_program_roundtrip_" + archive_path.stem, module_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        self.call(spec.loader.exec_module, module)
        return module

    def test_generated_source_preserves_typed_inputs(self):
        example_inputs = (
            torch.ones(2, 3, dtype=torch.int64),
            torch.ones(2, 3, dtype=torch.float64),
            torch.ones(2, 3, dtype=torch.bool),
        )
        module = self.convert(
            self.save("dtypes", DtypeIdentityModel(), example_inputs)
        )
        source = (self.work_dir / "dtypes_pnnx.py").read_text()

        self.assertEqual(source.count("def _create_example_inputs():"), 1)
        for expression in (
            "torch.randint(0, 10, (2, 3), dtype=torch.long)",
            "torch.rand(2, 3, dtype=torch.double)",
            "torch.randint(0, 2, (2, 3), dtype=torch.bool)",
        ):
            self.assertIn(expression, source)

        generated_inputs = self.call(module._create_example_inputs)
        output = self.call(module.test_inference)
        self.assertEqual(
            [tensor.dtype for tensor in generated_inputs],
            [torch.int64, torch.float64, torch.bool],
        )
        self.assertEqual(
            [tensor.dtype for tensor in output],
            [torch.int64, torch.float64, torch.bool],
        )
        self.call(module.export_torchscript)
        self.assertTrue((self.work_dir / "dtypes_pnnx.py.pt").is_file())

    def test_generated_model_preserves_special_state_dtypes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = StateDtypeModel().eval()
            module = self.convert(self.save("state_dtypes", model, (torch.ones(1),)))
            actual = self.call(module.Model).eval()(*self.call(module._create_example_inputs))

        expected = model(torch.ones(1))
        for expected_tensor, actual_tensor in zip(expected, actual):
            self.assertEqual(actual_tensor.dtype, expected_tensor.dtype)
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            self.assertTrue(
                torch.equal(
                    actual_tensor.view(torch.uint8), expected_tensor.view(torch.uint8)
                )
            )

    def test_empty_lists_keep_typed_parameter_encoding(self):
        int_archive = self.save(
            "empty_int_list", EmptyIntListModel(), (torch.tensor(2.0),)
        )
        first_module = self.convert(int_archive)
        self.assertIn("=[]", (self.work_dir / "empty_int_list.pnnx.param").read_text())
        self.call(first_module.export_exported_program)
        second_module = self.convert(self.work_dir / "empty_int_list_pnnx.pt2")
        second_param = (self.work_dir / "empty_int_list_pnnx.pnnx.param").read_text()
        self.assertIn("=[]", second_param)
        self.assertNotIn("=()", second_param)
        self.assertEqual(self.call(second_module.test_inference).shape, torch.Size([]))

        float_archive = self.save(
            "empty_float_list", EmptyFloatListModel(), (torch.ones(0),)
        )
        self.run_pnnx(float_archive)
        float_param = (self.work_dir / "empty_float_list.pnnx.param").read_text()
        self.assertIn("=[]f", float_param)
        self.assertNotIn("=()", float_param)

    def test_exported_program_round_trip_does_not_overwrite_input(self):
        torch.manual_seed(42)
        model = TinyModel().eval()
        archive_path = self.save("tiny", model, (torch.ones(2, 4),))
        original_archive = archive_path.read_bytes()
        first_module = self.convert(archive_path)

        source = (self.work_dir / "tiny_pnnx.py").read_text()
        self.assertIn("def export_exported_program(example_inputs=None):", source)
        with self.assertRaisesRegex(TypeError, "example_inputs must be a tuple"):
            self.call(first_module.export_exported_program, torch.ones(3, 4))

        program = self.call(first_module.export_exported_program)
        self.assertIsInstance(program, torch.export.ExportedProgram)
        self.assertEqual(archive_path.read_bytes(), original_archive)

        roundtrip_path = self.work_dir / "tiny_pnnx.pt2"
        self.assertTrue(roundtrip_path.is_file())
        second_module = self.convert(roundtrip_path)

        torch.manual_seed(0)
        expected = model(torch.rand(2, 4))
        for module in (first_module, second_module):
            actual = self.call(module.test_inference)
            self.assertTrue(torch.allclose(expected, actual, rtol=1e-5, atol=1e-5))

        custom_inputs = (torch.full((3, 4), 0.25),)
        custom_program = self.call(first_module.export_exported_program, custom_inputs)
        expected = self.call(first_module.Model).eval()(*custom_inputs)
        actual = custom_program.module()(*custom_inputs)
        self.assertEqual(tuple(actual.shape), (3, 3))
        self.assertTrue(torch.allclose(expected, actual, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
