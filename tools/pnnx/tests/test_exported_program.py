#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PNNX = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]] + sys.argv[2:]
TORCH_VERSION = tuple(
    int(component) for component in torch.__version__.split("+", 1)[0].split(".")[:2]
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))


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


class TwoInputModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class BoolInputModel(torch.nn.Module):
    def forward(self, x):
        return torch.logical_not(x)


class ScalarInputModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


class OperatorReturnsModel(torch.nn.Module):
    def forward(self, x):
        torch.ops.aten._assert_tensor_metadata.default(x, dtype=torch.float32)
        left, right = torch.split(x, 2, dim=1)
        values, indices = torch.max(x, dim=1)
        return left, right, values, indices


class StaticScalarShapeModel(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], torch.div(x.shape[1], 2, rounding_mode="trunc"), 2)


class MaxIndicesModel(torch.nn.Module):
    def forward(self, x):
        return torch.max(x, dim=1)[1]


class SingleTupleModel(torch.nn.Module):
    def forward(self, x):
        return (torch.relu(x),)


class NestedInputModel(torch.nn.Module):
    def forward(self, x, nested):
        return x - 2 * nested[0] + 3 * nested[1][0] - 4 * nested[1][1]


class BufferLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        self.register_buffer("weight", weight)

    def forward(self, x):
        return torch.relu(torch.nn.functional.linear(x, self.weight))


class NonPersistentBufferLinearModel(BufferLinearModel):
    def __init__(self):
        super().__init__()
        self._non_persistent_buffers_set.add("weight")


class TensorConstantLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10

    def forward(self, x):
        return torch.relu(torch.nn.functional.linear(x, self.weight))


class Int64BufferAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "value", torch.arange(6, dtype=torch.int64).reshape(2, 3)
        )

    def forward(self, x):
        return x + self.value


class StridedWeightLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        storage = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.transposed_weight = torch.nn.Parameter(storage.t(), False)
        offset_storage = torch.arange(20, dtype=torch.float32)
        self.offset_weight = torch.nn.Parameter(
            offset_storage[5:13].reshape(2, 4), False
        )

    def forward(self, x):
        x = torch.relu(torch.nn.functional.linear(x, self.transposed_weight))
        return torch.relu(torch.nn.functional.linear(x, self.offset_weight))


class SharedStorageLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        storage = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
        self.first_weight = torch.nn.Parameter(storage[:3, :], False)
        self.second_weight = torch.nn.Parameter(
            storage.reshape(-1)[:6].reshape(2, 3), False
        )

    def forward(self, x):
        x = torch.relu(torch.nn.functional.linear(x, self.first_weight))
        return torch.relu(torch.nn.functional.linear(x, self.second_weight))


class EmptyViewStateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("full", torch.arange(8, dtype=torch.float32))
        self.register_buffer("edge", self.full.reshape(2, 4)[2:, 1:])

    def forward(self, x):
        return x + self.full, self.edge


class StaticWeightNormModel(torch.nn.Module):
    def __init__(self, v, g, dim):
        super().__init__()
        self.register_buffer("v", v)
        self.register_buffer("g", g)
        self.dim = dim

    def forward(self, x):
        return torch._weight_norm(self.v, self.g, self.dim) + x


class StateNameCollisionSubmodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([1.0, 2.0, 3.0, 4.0]), requires_grad=False
        )


class StateNameCollisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.foo = StateNameCollisionSubmodule()
        self.foo_weight = torch.nn.Parameter(
            torch.tensor([10.0, 20.0, 30.0, 40.0]), requires_grad=False
        )

    def forward(self, x):
        return x * self.foo.weight + self.foo_weight


class StaticSymbolArgumentModel(torch.nn.Module):
    def forward(self, x):
        x = torch.ops.aten.leaky_relu.default(x, 0.25)
        x = torch.ops.aten.avg_pool2d.default(
            x, [2, 2], [2, 2], [0, 0], True, True, None
        )
        x = torch.ops.aten.reshape.default(x, [1, 4])
        return torch.ops.aten.flatten.using_ints(x, 1, -1)


class SpacedEinsumModel(torch.nn.Module):
    def forward(self, query, key):
        return torch.einsum("b i d, b j d -> b i j", query, key)


class ScalarEinsumModel(torch.nn.Module):
    def forward(self, left, right):
        return torch.einsum(",", left, right)


class StringArgumentModel(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate="tanh")


class Int64MaxFillModel(torch.nn.Module):
    def forward(self, x):
        return torch.full(
            (2, 3), torch.iinfo(torch.int64).max, dtype=torch.int64
        )


class Float64OverflowModel(torch.nn.Module):
    def forward(self, x):
        return x * 1e100


class FloatArgumentModel(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.leaky_relu(x - 1, negative_slope=0.25)


class NegativeInfinityFillModel(torch.nn.Module):
    def forward(self, x):
        return torch.full_like(x, float("-inf"))


class OpenEndedSliceModel(torch.nn.Module):
    def forward(self, x):
        return x[:, 1:]


def run_pnnx(work_dir, model_path, *arguments):
    return subprocess.run(
        [str(PNNX), model_path.name, *arguments],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def save_exported_program(model, archive_path, example_inputs=None):
    if example_inputs is None:
        example_inputs = (torch.ones(2, 4),)
    exported_program = torch.export.export(model.eval(), example_inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.export.save(exported_program, archive_path)


def archive_entry(entries, marker, suffix=""):
    matches = [
        name for name in entries if marker in name and name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one archive entry for {marker!r}/{suffix!r}, got {matches!r}"
        )
    return matches[0]


def rewrite_archive(source_path, destination_path, mutate):
    with zipfile.ZipFile(source_path, "r") as source:
        entries = {name: source.read(name) for name in source.namelist()}

    mutate(entries)

    with zipfile.ZipFile(
        destination_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as destination:
        for name, data in entries.items():
            destination.writestr(name, data, compress_type=zipfile.ZIP_STORED)


def rewrite_model_json(source_path, destination_path, mutate):
    def mutate_entries(entries):
        model_path = archive_entry(entries, "/models/", ".json")
        document = json.loads(entries[model_path])
        mutate(document)
        entries[model_path] = json.dumps(
            document, separators=(",", ":")
        ).encode()

    rewrite_archive(source_path, destination_path, mutate_entries)


def rewrite_payload_configs(source_path, destination_path, mutate):
    def mutate_entries(entries):
        weights_path = archive_entry(entries, "", "_weights_config.json")
        constants_path = archive_entry(entries, "", "_constants_config.json")
        weights = json.loads(entries[weights_path])
        constants = json.loads(entries[constants_path])
        mutate(weights["config"], constants["config"])
        entries[weights_path] = json.dumps(
            weights, separators=(",", ":")
        ).encode()
        entries[constants_path] = json.dumps(
            constants, separators=(",", ":")
        ).encode()

    rewrite_archive(source_path, destination_path, mutate_entries)


def load_generated_output(work_dir, basename):
    module_path = work_dir / f"{basename}_pnnx.py"
    spec = importlib.util.spec_from_file_location(
        f"test_exported_program_{basename}_pnnx", module_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load generated module {module_path}")

    module = importlib.util.module_from_spec(spec)
    previous_work_dir = Path.cwd()
    try:
        os.chdir(work_dir)
        spec.loader.exec_module(module)
        return module.test_inference()
    finally:
        os.chdir(previous_work_dir)


def load_generated_ncnn_module(work_dir, basename):
    module_path = work_dir / f"{basename}_ncnn.py"
    spec = importlib.util.spec_from_file_location(
        f"test_exported_program_{basename}_ncnn", module_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load generated module {module_path}")

    module = importlib.util.module_from_spec(spec)
    # These tests must reject unsupported inputs before using the native binding.
    with mock.patch.dict(sys.modules, {"ncnn": types.ModuleType("ncnn")}):
        spec.loader.exec_module(module)
    return module


class NonSeekableBuffer(io.BytesIO):
    def seekable(self):
        return False

    def seek(self, *args, **kwargs):
        raise io.UnsupportedOperation("seek")


class ExportedProgramEndToEndTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyModel().eval()

    def assert_nested_close(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor)
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertTrue(
                torch.allclose(expected, actual, rtol=1e-4, atol=1e-4),
                f"generated output mismatch\nexpected={expected}\nactual={actual}",
            )
            return

        self.assertIs(type(actual), type(expected))
        self.assertEqual(len(expected), len(actual))
        for expected_item, actual_item in zip(expected, actual):
            self.assert_nested_close(expected_item, actual_item)

    def assert_conversion_matches(self, work_dir, model_path, expected=None):
        result = run_pnnx(work_dir, model_path)
        self.assertEqual(
            result.returncode, 0, result.stderr.decode(errors="replace")
        )

        if expected is None:
            torch.manual_seed(0)
            expected = self.model(torch.rand(2, 4))
        actual = load_generated_output(work_dir, model_path.stem)
        self.assert_nested_close(expected, actual)

    def assert_conversion_fails(self, work_dir, model_path, *messages):
        result = run_pnnx(work_dir, model_path)
        stderr = result.stderr.decode(errors="replace")
        self.assertGreater(
            result.returncode,
            0,
            "conversion must fail without a signal\n" + stderr,
        )
        for message in messages:
            self.assertIn(message, stderr)

        for suffix in (
            ".pnnx.param",
            ".pnnx.bin",
            "_pnnx.py",
            ".ncnn.param",
            ".ncnn.bin",
            "_ncnn.py",
        ):
            self.assertFalse(
                (work_dir / f"{model_path.stem}{suffix}").exists(),
                f"failed conversion left {model_path.stem}{suffix}",
            )

    def test_tiny_program_and_content_based_routing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "tiny.pt2"
            renamed_path = work_dir / "tiny_renamed.bin"
            torchscript_path = work_dir / "legacy.pt2"

            save_exported_program(self.model, archive_path)
            shutil.copyfile(archive_path, renamed_path)
            self.assert_conversion_matches(work_dir, archive_path)
            self.assert_conversion_matches(work_dir, renamed_path)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traced = torch.jit.trace(self.model, (torch.ones(2, 4),))
            traced.save(str(torchscript_path))
            self.assert_conversion_matches(work_dir, torchscript_path)
            self.assertIn(
                "net.float()", (work_dir / "legacy_pnnx.py").read_text()
            )

    def test_real_producer_omits_default_arguments(self):
        model = CompatibilityModel().eval()
        example_inputs = (torch.ones(1, 1, 5, 5),)
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "producer_defaults.pt2"
            save_exported_program(model, archive_path, example_inputs)

            with zipfile.ZipFile(archive_path) as archive:
                model_path = archive_entry(
                    archive.namelist(), "/models/", ".json"
                )
                document = json.loads(archive.read(model_path))
            self.assertEqual(document["schema_version"]["major"], 8)
            self.assertIsInstance(document["schema_version"]["minor"], int)
            self.assertIsInstance(document["opset_version"]["aten"], int)

            serialized_nodes = {
                node["target"]: [argument["name"] for argument in node["inputs"]]
                for node in document["graph_module"]["graph"]["nodes"]
            }
            self.assertEqual(
                serialized_nodes["torch.ops.aten.add.Tensor"],
                ["self", "other"],
            )
            self.assertEqual(
                serialized_nodes["torch.ops.aten.flatten.using_ints"],
                ["self", "start_dim"],
            )
            self.assertEqual(
                serialized_nodes["torch.ops.aten.conv2d.default"],
                ["input", "weight"],
            )

            torch.manual_seed(0)
            expected = model(torch.rand(1, 1, 5, 5))
            self.assert_conversion_matches(work_dir, archive_path, expected)

    def test_static_weight_norm_broadcast_and_finite_values(self):
        v = torch.arange(1, 19, dtype=torch.float32).reshape(2, 3, 3)
        g = torch.tensor([1., 2., 3.])
        cases = (
            (v, g.reshape(1, 3, 1), 1),
            (v, g, 1),
            (v, g, -2),
            (torch.tensor([[1e10, 1e10]]), torch.tensor([[1e30]]), 0),
            (torch.cat((torch.tensor([256.]), torch.full((32767,), 0.0625))).reshape(1, -1),
             torch.ones(1, 1), 0),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            for index, (weight, scale, dim) in enumerate(cases):
                with self.subTest(case=index):
                    model = StaticWeightNormModel(weight, scale, dim).eval()
                    archive_path = work_dir / f"weight_norm_{index}.pt2"
                    save_exported_program(model, archive_path, (torch.zeros_like(weight),))
                    torch.manual_seed(0)
                    expected = model(torch.rand(weight.shape))
                    self.assertTrue(torch.isfinite(expected).all())
                    self.assert_conversion_matches(work_dir, archive_path, expected)

    def test_static_weight_norm_float32_range(self):
        cases = (
            (1e20, 1.),       # Each squared value overflows.
            (1e19, 1.),       # Only the sum of squares overflows.
            (1e-30, 1e-30),   # The squared values underflow to zero.
            (1e-20, 1e30),    # The scale / norm ratio overflows.
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            for dim in (0, 1, 2, -1):
                shape = [1, 1, 1]
                if dim != -1:
                    shape[dim] = 2
                for index, (value, scale) in enumerate(cases):
                    with self.subTest(dim=dim, case=index):
                        weight = torch.full((2, 2, 2), value)
                        model = StaticWeightNormModel(
                            weight, torch.full(shape, scale), dim
                        ).eval()
                        archive_path = work_dir / f"range_{dim + 1}_{index}.pt2"
                        save_exported_program(model, archive_path, (torch.zeros_like(weight),))
                        torch.manual_seed(0)
                        expected = model(torch.rand(weight.shape))
                        self.assert_conversion_matches(work_dir, archive_path, expected)
                        graph = archive_path.with_suffix(".pnnx.param").read_text()
                        self.assertNotIn("torch._weight_norm ", graph)

    def test_empty_state_view_beyond_storage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "empty_state_view.pt2"
            model = EmptyViewStateModel().eval()
            save_exported_program(model, archive_path, (torch.ones(8),))
            torch.manual_seed(0)
            self.assert_conversion_matches(work_dir, archive_path, model(torch.rand(8)))

    def test_scalar_numpy_input_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "scalar_numpy.pt2"
            model = ScalarInputModel().eval()
            save_exported_program(model, archive_path, (torch.tensor(2.0),))
            np.save(work_dir / "scalar.npy", np.array(2.0, dtype=np.float32))
            result = run_pnnx(work_dir, archive_path, "input=scalar.npy")
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            torch.manual_seed(0)
            self.assert_nested_close(
                model(torch.rand(())), load_generated_output(work_dir, archive_path.stem)
            )

    def test_operator_returns_are_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            model = OperatorReturnsModel().eval()
            valid_path = work_dir / "valid_returns.pt2"
            save_exported_program(model, valid_path)
            torch.manual_seed(0)
            self.assert_conversion_matches(work_dir, valid_path, model(torch.rand(2, 4)))

            source_path = work_dir / "return_source.pt2"
            save_exported_program(ScalarInputModel(), source_path)
            cases = (
                ([], "return count"),
                ([{"as_tensors": [{"name": "alias_out"}]}], "return 0"),
                ([{"as_none": True}], None),
            )
            for index, (outputs, message) in enumerate(cases):
                with self.subTest(case=index):
                    archive_path = work_dir / f"invalid_returns_{index}.pt2"

                    def add_invalid_alias(document):
                        graph = document["graph_module"]["graph"]
                        input_name = graph["inputs"][0]["as_tensor"]["name"]
                        graph["tensor_values"]["alias_out"] = graph["tensor_values"][input_name]
                        graph["nodes"].insert(0, {
                            "target": "torch.ops.aten.alias.default",
                            "inputs": [{"name": "self", "arg": graph["inputs"][0], "kind": 1}],
                            "outputs": outputs,
                            "metadata": {},
                        })

                    rewrite_model_json(source_path, archive_path, add_invalid_alias)
                    if message is None:
                        torch.manual_seed(0)
                        expected = ScalarInputModel()(torch.rand(2, 4))
                        self.assert_conversion_matches(work_dir, archive_path, expected)
                    else:
                        self.assert_conversion_fails(work_dir, archive_path, "aten.alias.default", message)

    def test_unused_operator_return_slots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            for model in (StaticScalarShapeModel(), MaxIndicesModel()):
                with self.subTest(model=type(model).__name__):
                    source_path = work_dir / (type(model).__name__ + ".pt2")
                    archive_path = work_dir / (type(model).__name__ + "_unused.pt2")
                    save_exported_program(model, source_path)

                    def mark_unused_return(document):
                        for node in document["graph_module"]["graph"]["nodes"]:
                            if node["target"] == "torch.ops.aten.max.dim":
                                node["outputs"][0] = {"as_none": True}

                    rewrite_model_json(source_path, archive_path, mark_unused_return)
                    torch.manual_seed(0)
                    inputs = (torch.rand(2, 4),)
                    expected = model(*inputs)
                    if isinstance(model, MaxIndicesModel):
                        loaded = torch.export.load(archive_path)
                        self.assert_nested_close(loaded.module()(*inputs), expected)
                    self.assert_conversion_matches(work_dir, archive_path, expected)

    def test_torchvision_operator_schema_contracts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "torchvision_contract_source.pt2"
            save_exported_program(ScalarInputModel().eval(), source_path)

            def replace_with_roi_align(document):
                graph = document["graph_module"]["graph"]
                node = graph["nodes"][0]
                tensor = node["inputs"][0]["arg"]
                node["target"] = "torch.ops.torchvision.roi_align.default"
                node["inputs"] = [
                    {"name": "input", "arg": tensor, "kind": 1},
                    {"name": "rois", "arg": tensor, "kind": 1},
                    {"name": "spatial_scale", "arg": {"as_float": 0.25}, "kind": 1},
                    {"name": "pooled_height", "arg": {"as_int": 3}, "kind": 1},
                    {"name": "pooled_width", "arg": {"as_int": 3}, "kind": 1},
                    {"name": "sampling_ratio", "arg": {"as_int": 2}, "kind": 1},
                    {"name": "aligned", "arg": {"as_bool": False}, "kind": 1},
                ]

            cases = (
                (
                    "return_count",
                    lambda node: node.update(outputs=[]),
                    ("torch.ops.torchvision.roi_align.default", "return count"),
                ),
                (
                    "return_type",
                    lambda node: node.update(outputs=[{"as_int": 1}]),
                    ("torch.ops.torchvision.roi_align.default", "return 0", "Tensor"),
                ),
                (
                    "argument_type",
                    lambda node: node["inputs"][2].update(arg={"as_int": 1}),
                    ("argument spatial_scale", "float"),
                ),
                (
                    "missing_argument",
                    lambda node: node["inputs"].pop(),
                    ("missing required argument aligned",),
                ),
                (
                    "duplicate_argument",
                    lambda node: node["inputs"].append(dict(node["inputs"][-1])),
                    ("duplicate argument aligned",),
                ),
                (
                    "argument_kind",
                    lambda node: node["inputs"][0].update(kind=0),
                    ("unknown argument kind for input",),
                ),
                (
                    "unknown_operator",
                    lambda node: node.update(target="torch.ops.torchvision.unknown.default"),
                    ("unsupported exported operator",),
                ),
                (
                    "non_default_overload",
                    lambda node: node.update(target="torch.ops.torchvision.roi_align.special"),
                    ("unsupported exported operator",),
                ),
            )
            for name, mutate, messages in cases:
                with self.subTest(contract=name):
                    archive_path = work_dir / f"torchvision_contract_{name}.pt2"

                    def mutate_document(document):
                        replace_with_roi_align(document)
                        mutate(document["graph_module"]["graph"]["nodes"][0])

                    rewrite_model_json(source_path, archive_path, mutate_document)
                    self.assert_conversion_fails(work_dir, archive_path, *messages)

    def test_input_shape_overrides_are_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "two_inputs.pt2"
            save_exported_program(
                TwoInputModel().eval(),
                source_path,
                (torch.ones(2, 4), torch.ones(2, 4)),
            )

            invalid_arguments = (
                (
                    "inputshape=[2,4]",
                    "input_shape expect 2 tensors but got 1",
                ),
                (
                    "inputshape=[2,4],[2,4],[2,4]",
                    "input_shape expect 2 tensors but got 3",
                ),
                (
                    "inputshape=[1,4],[2,4]",
                    "input_shapes[0] expect [2,4]f32 but got [1,4]f32",
                ),
                (
                    "inputshape=[2,4]i32,[2,4]",
                    "input_shapes[0] expect [2,4]f32 but got [2,4]i32",
                ),
            )
            for index, (argument, message) in enumerate(invalid_arguments):
                with self.subTest(argument=argument):
                    archive_path = work_dir / f"invalid_input_{index}.pt2"
                    shutil.copyfile(source_path, archive_path)
                    result = run_pnnx(work_dir, archive_path, argument)
                    stderr = result.stderr.decode(errors="replace")
                    self.assertGreater(result.returncode, 0, stderr)
                    self.assertIn(message, stderr)

            second_shape_path = work_dir / "second_input_shape.pt2"
            shutil.copyfile(source_path, second_shape_path)
            result = run_pnnx(
                work_dir,
                second_shape_path,
                "inputshape=[2,4],[2,4]",
                "inputshape2=[3,4],[3,4]",
            )
            stderr = result.stderr.decode(errors="replace")
            self.assertGreater(result.returncode, 0, stderr)
            self.assertIn(
                "inputshape2 and input2 are unsupported for exported program",
                stderr,
            )

            valid_path = work_dir / "valid_input_shape.pt2"
            shutil.copyfile(source_path, valid_path)
            result = run_pnnx(
                work_dir, valid_path, "inputshape=[2,4],[2,4]"
            )
            self.assertEqual(
                result.returncode, 0, result.stderr.decode(errors="replace")
            )

    def test_generated_ncnn_helper_rejects_unsupported_inputs(self):
        cases = (
            ("bool", torch.bool, (2, 3), BoolInputModel()),
            ("bfloat16", torch.bfloat16, (2, 3), ScalarInputModel()),
            ("scalar", torch.float32, (), ScalarInputModel()),
            ("complex", torch.complex32, (2, 3), ScalarInputModel()),
            ("complex", torch.complex64, (2, 3), ScalarInputModel()),
            ("complex", torch.complex128, (2, 3), ScalarInputModel()),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            for index, (kind, dtype, shape, model) in enumerate(cases):
                with self.subTest(kind=kind, dtype=dtype):
                    archive_path = work_dir / f"unsupported_input_{index}.pt2"
                    save_exported_program(
                        model, archive_path, (torch.ones(shape, dtype=dtype),)
                    )
                    result = run_pnnx(work_dir, archive_path)
                    self.assertEqual(
                        result.returncode, 0, result.stderr.decode(errors="replace")
                    )
                    torch.manual_seed(0)
                    example = (
                        torch.randint(0, 2, shape, dtype=dtype)
                        if dtype == torch.bool else torch.rand(shape, dtype=dtype)
                    )
                    self.assert_nested_close(
                        model(example), load_generated_output(work_dir, archive_path.stem)
                    )
                    module = load_generated_ncnn_module(work_dir, archive_path.stem)
                    with self.assertRaisesRegex(
                        RuntimeError, f"ncnn inference does not support {kind} input in0"
                    ):
                        module.test_inference()

    def test_output_comparison_checks_structure_shape_and_dtype(self):
        expected = (torch.ones(2, 1), torch.zeros(2))
        for actual in (
            expected[:1], list(expected),
            (torch.ones(2), expected[1]),
            (torch.ones(2, 1, dtype=torch.float64), expected[1]),
        ):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                self.assert_nested_close(expected, actual)

    def test_torchscript_extra_archive_format_is_not_pt2_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            torchscript_path = work_dir / "legacy_extra.pt"
            example_inputs = (torch.ones(2, 4),)
            traced = torch.jit.trace(self.model, example_inputs)
            torch.jit.save(
                traced,
                str(torchscript_path),
                _extra_files={"archive_format": "pt2"},
            )

            with zipfile.ZipFile(torchscript_path, "r") as archive:
                self.assertTrue(
                    any(
                        name.endswith("/extra/archive_format")
                        for name in archive.namelist()
                    )
                )

            torch.manual_seed(0)
            self.assert_conversion_matches(
                work_dir, torchscript_path, self.model(torch.rand(2, 4))
            )

    def test_unrepresentable_integer_parameter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)

            int64_path = work_dir / "int64_max.pt2"
            save_exported_program(Int64MaxFillModel().eval(), int64_path)
            self.assert_conversion_fails(
                work_dir,
                int64_path,
                "integer value 9223372036854775807 does not fit pnnx integer parameter",
            )

    def test_unrepresentable_float_parameter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            float64_path = work_dir / "float64_overflow.pt2"
            save_exported_program(
                Float64OverflowModel().eval(),
                float64_path,
                (torch.ones(2, dtype=torch.float64),),
            )
            self.assert_conversion_fails(
                work_dir,
                float64_path,
                "floating-point value 1e+100 does not fit pnnx float parameter",
            )

    def test_infinite_float_parameter_uses_pnnx_sentinel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            model = NegativeInfinityFillModel().eval()
            archive_path = work_dir / "negative_infinity_fill.pt2"
            example_inputs = (torch.ones(2, 3),)
            save_exported_program(model, archive_path, example_inputs)
            torch.manual_seed(0)
            self.assert_conversion_matches(
                work_dir, archive_path, model(torch.rand(2, 3))
            )

    def test_open_ended_slice_converts_with_pnnx_sentinel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            model = OpenEndedSliceModel().eval()
            archive_path = work_dir / "open_ended_slice.pt2"
            save_exported_program(model, archive_path, (torch.ones(2, 3),))
            torch.manual_seed(0)
            self.assert_conversion_matches(
                work_dir, archive_path, model(torch.rand(2, 3))
            )

    def test_input_and_output_trees(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)

            nested_model = NestedInputModel().eval()
            nested_path = work_dir / "nested_inputs.pt2"
            example_inputs = (
                torch.ones(2, 4),
                (
                    torch.full((2, 4), 2.0),
                    [torch.full((2, 4), 3.0), torch.full((2, 4), 4.0)],
                ),
            )
            save_exported_program(nested_model, nested_path, example_inputs)
            torch.manual_seed(0)
            expected = nested_model(
                torch.rand(2, 4),
                (torch.rand(2, 4), [torch.rand(2, 4), torch.rand(2, 4)]),
            )
            self.assert_conversion_matches(work_dir, nested_path, expected)

            tuple_model = SingleTupleModel().eval()
            tuple_path = work_dir / "single_tuple.pt2"
            save_exported_program(tuple_model, tuple_path)
            torch.manual_seed(0)
            expected = tuple_model(torch.rand(2, 4))
            self.assert_conversion_matches(work_dir, tuple_path, expected)

            leaf = {"type": None, "context": None, "children_spec": []}
            cases = (
                ("unsupported_type", {"type": "builtins.dict", "context": "null", "children_spec": [leaf]},
                 "unsupported output treespec type builtins.dict"),
                ("leaf_children", {"type": None, "context": None, "children_spec": [leaf]},
                 "leaf must not have children"),
                ("leaf_count", {"type": "builtins.tuple", "context": "null", "children_spec": [leaf, leaf]},
                 "output treespec leaf count does not match graph outputs"),
            )
            for label, tree, message in cases:
                with self.subTest(output_tree=label):
                    archive_path = work_dir / (label + ".pt2")

                    def replace_tree(document):
                        root = next(entry for entry in document["graph_module"]["module_call_graph"]
                                    if entry["fqn"] == "")
                        root["signature"]["out_spec"] = json.dumps([1, tree])

                    rewrite_model_json(tuple_path, archive_path, replace_tree)
                    self.assert_conversion_fails(work_dir, archive_path,
                                                 "invalid exported program", message)

    def test_keyword_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            archive_path = work_dir / "keyword_input.pt2"
            exported_program = torch.export.export(
                self.model, (), {"x": torch.ones(2, 4)}
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.export.save(exported_program, archive_path)
            self.assert_conversion_fails(
                work_dir,
                archive_path,
                "keyword inputs are unsupported",
            )

    def test_parameter_buffer_and_constant_payloads(self):
        cases = (
            ("parameter", self.model),
            ("buffer", BufferLinearModel().eval()),
            ("non_persistent_buffer", NonPersistentBufferLinearModel().eval()),
            ("constant", TensorConstantLinearModel().eval()),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            for name, model in cases:
                with self.subTest(kind=name):
                    archive_path = work_dir / f"state_{name}.pt2"
                    save_exported_program(model, archive_path)
                    torch.manual_seed(0)
                    expected = model(torch.rand(2, 4))
                    self.assert_conversion_matches(
                        work_dir, archive_path, expected
                    )

    def test_state_names_remain_distinct_after_python_sanitization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            model = StateNameCollisionModel().eval()
            archive_path = work_dir / "state_name_collision.pt2"
            save_exported_program(model, archive_path)
            torch.manual_seed(0)
            expected = model(torch.rand(2, 4))
            self.assert_conversion_matches(
                work_dir, archive_path, expected
            )

    def test_dtype_stride_and_shared_storage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)

            int64_model = Int64BufferAddModel().eval()
            int64_path = work_dir / "int64_buffer.pt2"
            save_exported_program(
                int64_model, int64_path, (torch.ones(2, 3),)
            )
            torch.manual_seed(0)
            self.assert_conversion_matches(
                work_dir,
                int64_path,
                int64_model(torch.rand(2, 3)),
            )

            strided_model = StridedWeightLinearModel().eval()
            self.assertFalse(strided_model.transposed_weight.is_contiguous())
            self.assertGreater(strided_model.offset_weight.storage_offset(), 0)
            strided_path = work_dir / "strided_weight.pt2"
            save_exported_program(
                strided_model, strided_path, (torch.ones(2, 3),)
            )
            torch.manual_seed(0)
            self.assert_conversion_matches(
                work_dir,
                strided_path,
                strided_model(torch.rand(2, 3)),
            )

            shared_model = SharedStorageLinearModel().eval()
            shared_path = work_dir / "shared_storage.pt2"
            save_exported_program(shared_model, shared_path)
            with zipfile.ZipFile(shared_path, "r") as archive:
                weights_path = archive_entry(
                    archive.namelist(), "", "_weights_config.json"
                )
                config = json.loads(archive.read(weights_path))["config"]
            self.assertEqual(
                config["first_weight"]["path_name"],
                config["second_weight"]["path_name"],
            )
            torch.manual_seed(0)
            self.assert_conversion_matches(
                work_dir,
                shared_path,
                shared_model(torch.rand(2, 4)),
            )

    def test_einsum_normalization_and_scalar_rejection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            model = SpacedEinsumModel().eval()
            archive_path = work_dir / "spaced_einsum.pt2"
            example_inputs = (torch.ones(2, 3, 4), torch.ones(2, 5, 4))
            save_exported_program(model, archive_path, example_inputs)
            torch.manual_seed(0)
            expected = model(torch.rand(2, 3, 4), torch.rand(2, 5, 4))
            self.assert_conversion_matches(work_dir, archive_path, expected)
            generated = (work_dir / "spaced_einsum_pnnx.py").read_text()
            einsum_line = next(
                line for line in generated.splitlines() if "torch.einsum(" in line
            )
            equation = einsum_line.split("torch.einsum('", 1)[1].split("'", 1)[0]
            self.assertNotIn(" ", equation)

            scalar_path = work_dir / "scalar_einsum.pt2"
            save_exported_program(
                ScalarEinsumModel().eval(),
                scalar_path,
                (torch.tensor(2.0), torch.tensor(3.0)),
            )
            self.assert_conversion_fails(
                work_dir,
                scalar_path,
                "scalar einsum operands are unsupported",
            )

    def test_unsafe_scalar_strings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "string_source.pt2"
            save_exported_program(StringArgumentModel().eval(), source_path)

            for index, unsafe_value in enumerate(
                ("two words", "a'b", "torch.float32")
            ):
                with self.subTest(value=unsafe_value):
                    archive_path = work_dir / f"string_unsafe_{index}.pt2"

                    def replace_approximate(document):
                        node = next(
                            node
                            for node in document["graph_module"]["graph"][
                                "nodes"
                            ]
                            if node["target"]
                            == "torch.ops.aten.gelu.default"
                        )
                        approximate = next(
                            value
                            for value in node["inputs"]
                            if value["name"] == "approximate"
                        )
                        approximate["arg"] = {"as_string": unsafe_value}

                    rewrite_model_json(
                        source_path, archive_path, replace_approximate
                    )
                    self.assert_conversion_fails(
                        work_dir,
                        archive_path,
                        "argument approximate",
                        "string is not safely representable by pnnx Parameter",
                    )

    def test_static_symbol_arguments_convert(self):
        model = StaticSymbolArgumentModel().eval()
        example_inputs = (torch.ones(1, 1, 4, 4),)
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "static_symbols_source.pt2"
            archive_path = work_dir / "static_symbols.pt2"
            save_exported_program(model, source_path, example_inputs)

            def use_static_symbol_tags(document):
                nodes = document["graph_module"]["graph"]["nodes"]

                def argument(target, name):
                    node = next(node for node in nodes if node["target"] == target)
                    return next(
                        value for value in node["inputs"] if value["name"] == name
                    )

                argument(
                    "torch.ops.aten.avg_pool2d.default", "ceil_mode"
                )["arg"] = {"as_sym_bool": {"as_bool": True}}
                argument("torch.ops.aten.reshape.default", "shape")["arg"] = {
                    "as_sym_ints": [{"as_int": 1}, {"as_int": 4}]
                }
                argument(
                    "torch.ops.aten.leaky_relu.default", "negative_slope"
                )["arg"] = {"as_sym_float": {"as_float": 0.25}}
                argument(
                    "torch.ops.aten.flatten.using_ints", "start_dim"
                )["arg"] = {"as_sym_int": {"as_int": 1}}

            rewrite_model_json(source_path, archive_path, use_static_symbol_tags)
            torch.manual_seed(0)
            self.assert_conversion_matches(
                work_dir, archive_path, model(torch.rand(1, 1, 4, 4))
            )

    def test_dynamic_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "dynamic_source.pt2"
            save_exported_program(self.model, source_path)

            def first_tensor_meta(document):
                return next(
                    iter(
                        document["graph_module"]["graph"][
                            "tensor_values"
                        ].values()
                    )
                )

            def use_symbolic_size(document):
                first_tensor_meta(document)["sizes"][0] = {
                    "as_expr": {
                        "expr_str": "s0",
                        "hint": {"as_int": 2},
                    }
                }

            def use_symbolic_stride(document):
                first_tensor_meta(document)["strides"][0] = {
                    "as_expr": {"expr_str": "s0"}
                }

            def use_symbolic_storage_offset(document):
                first_tensor_meta(document)["storage_offset"] = {
                    "as_expr": {"expr_str": "s0"}
                }

            def add_symbol_value(document):
                document["graph_module"]["graph"]["sym_int_values"][
                    "s0"
                ] = {"as_int": 2}

            def add_range_constraint(document):
                document["range_constraints"]["s0"] = {
                    "min_val": 1,
                    "max_val": 4,
                }

            def use_named_operator_symint(document):
                document["graph_module"]["graph"]["nodes"][0][
                    "inputs"
                ][0]["arg"] = {"as_sym_int": {"as_name": "s0"}}

            cases = (
                (
                    "size_expression_with_hint",
                    use_symbolic_size,
                    (".sizes[0].as_expr", "dynamic tensor shapes are unsupported"),
                ),
                (
                    "stride_expression",
                    use_symbolic_stride,
                    (".strides[0].as_expr", "dynamic tensor shapes are unsupported"),
                ),
                (
                    "storage_offset_expression",
                    use_symbolic_storage_offset,
                    (
                        ".storage_offset.as_expr",
                        "dynamic tensor shapes are unsupported",
                    ),
                ),
                (
                    "symbol_value",
                    add_symbol_value,
                    (".sym_int_values", "dynamic symbolic values are unsupported"),
                ),
                (
                    "range_constraint",
                    add_range_constraint,
                    (
                        "$.range_constraints",
                        "dynamic range constraints are unsupported",
                    ),
                ),
                (
                    "named_operator_symint",
                    use_named_operator_symint,
                    ("unsupported serialized argument as_sym_int",),
                ),
            )
            for label, mutate, messages in cases:
                with self.subTest(dynamic=label):
                    archive_path = work_dir / f"dynamic_{label}.pt2"
                    rewrite_model_json(source_path, archive_path, mutate)
                    self.assert_conversion_fails(
                        work_dir, archive_path, *messages
                    )

    def test_float_arguments_accept_integer_json_numbers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "float_argument.pt2"
            save_exported_program(FloatArgumentModel(), source_path)
            for tag in ("as_float", "as_sym_float"):
                for index, value in enumerate((0, -1, 1, 2**63)):
                    with self.subTest(tag=tag, value=value):
                        archive_path = work_dir / f"{tag}_{index}.pt2"

                        def replace_slope(document):
                            node = next(
                                node for node in document["graph_module"]["graph"]["nodes"]
                                if node["target"] == "torch.ops.aten.leaky_relu.default"
                            )
                            argument = next(
                                arg for arg in node["inputs"] if arg["name"] == "negative_slope"
                            )
                            argument["arg"] = {
                                tag: value if tag == "as_float" else {"as_float": value}
                            }

                        rewrite_model_json(source_path, archive_path, replace_slope)
                        torch.manual_seed(0)
                        expected = (torch.rand(2, 4) - 1) * float(value)
                        self.assert_conversion_matches(work_dir, archive_path, expected)

    def test_schema_and_opset_contracts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "contract_source.pt2"
            save_exported_program(self.model, source_path)

            def omit_device_indices(document):
                for meta in document["graph_module"]["graph"][
                    "tensor_values"
                ].values():
                    meta["device"].pop("index", None)

            def replace_device_indices(document, index):
                for meta in document["graph_module"]["graph"][
                    "tensor_values"
                ].values():
                    meta["device"]["index"] = index

            def omit_node_metadata(document):
                for node in document["graph_module"]["graph"]["nodes"]:
                    node.pop("metadata", None)

            def replace_node_metadata(document):
                for node in document["graph_module"]["graph"]["nodes"]:
                    node["metadata"] = ["ignored", {"future": True}]

            with zipfile.ZipFile(source_path, "r") as archive:
                model_path = archive_entry(
                    archive.namelist(), "/models/", ".json"
                )
                document = json.loads(archive.read(model_path))
            self.assertIsInstance(document["schema_version"]["major"], int)
            self.assertIsInstance(document["schema_version"]["minor"], int)
            self.assertIsInstance(document["opset_version"]["aten"], int)
            self.assert_conversion_matches(work_dir, source_path)

            for label, mutate in (
                ("producer_default", lambda doc: doc.pop("torch_version", None)),
                ("device_default", omit_device_indices),
                ("metadata_absent", omit_node_metadata),
                ("metadata_ignored", replace_node_metadata),
            ):
                with self.subTest(optional=label):
                    candidate = work_dir / (label + ".pt2")
                    rewrite_model_json(source_path, candidate, mutate)
                    self.assert_conversion_matches(work_dir, candidate)

            cases = (
                (
                    "torch_version_type",
                    lambda value: value.update(torch_version=42),
                    "$.torch_version: expected string",
                ),
                (
                    "device_index_type",
                    lambda value: replace_device_indices(value, "0"),
                    ".device.index: expected integer",
                ),
                (
                    "device_index_negative",
                    lambda value: replace_device_indices(value, -1),
                    ".device.index: device index must be non-negative",
                ),
                (
                    "node_target_required",
                    lambda value: value["graph_module"]["graph"]["nodes"][
                        0
                    ].pop("target"),
                    ".nodes[0].target: missing required field",
                ),
                (
                    "node_inputs_required",
                    lambda value: value["graph_module"]["graph"]["nodes"][
                        0
                    ].pop("inputs"),
                    ".nodes[0].inputs: missing required field",
                ),
                (
                    "node_outputs_required",
                    lambda value: value["graph_module"]["graph"]["nodes"][
                        0
                    ].pop("outputs"),
                    ".nodes[0].outputs: missing required field",
                ),
                (
                    "tensor_values_required",
                    lambda value: value["graph_module"]["graph"].pop(
                        "tensor_values"
                    ),
                    ".graph.tensor_values: missing required field",
                ),
                (
                    "schema",
                    lambda value: value["schema_version"].update(minor=999),
                    "unsupported schema minor",
                ),
                (
                    "opset",
                    lambda value: value["opset_version"].update(
                        aten=value["opset_version"]["aten"] + 1
                    ),
                    "does not match linked libtorch opset",
                ),
            )
            for name, mutate, message in cases:
                with self.subTest(contract=name):
                    archive_path = work_dir / f"contract_{name}.pt2"
                    rewrite_model_json(source_path, archive_path, mutate)
                    self.assert_conversion_fails(
                        work_dir, archive_path, message
                    )

    def test_dispatcher_argument_type_is_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "type_source.pt2"
            archive_path = work_dir / "type_invalid.pt2"
            save_exported_program(self.model, source_path)

            def replace_relu_input(document):
                node = next(
                    node
                    for node in document["graph_module"]["graph"]["nodes"]
                    if node["target"] == "torch.ops.aten.relu.default"
                )
                node["inputs"][0]["arg"] = {"as_int": 1}

            rewrite_model_json(source_path, archive_path, replace_relu_input)
            self.assert_conversion_fails(
                work_dir,
                archive_path,
                "torch.ops.aten.relu.default",
                "argument self has serialized type int incompatible with "
                "dispatcher schema Tensor",
            )

    def test_disabled_higher_order_wrappers_and_enabled_rejections(self):
        def inject_wrapper(document, kind, enabled):
            graph = document["graph_module"]["graph"]
            output_name = graph["outputs"][0]["as_tensor"]["name"]
            tensor_meta = graph["tensor_values"]["x"]

            def tensor(name):
                return {"as_tensor": {"name": name}}

            def positional(argument):
                return {"name": "", "arg": argument, "kind": 1}

            subgraph = {
                "inputs": [tensor("captured")],
                "outputs": [tensor("sub_output")],
                "nodes": [
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [
                            {
                                "name": "self",
                                "arg": tensor("captured"),
                                "kind": 1,
                            }
                        ],
                        "outputs": [tensor("sub_output")],
                        "metadata": {},
                        "is_hop_single_tensor_return": None,
                    }
                ],
                "tensor_values": {
                    "captured": tensor_meta,
                    "sub_output": tensor_meta,
                },
                "sym_int_values": {},
                "sym_bool_values": {},
                "sym_float_values": {},
                "is_single_tensor_return": False,
                "custom_obj_values": {},
            }
            graph_argument = positional(
                {"as_graph": {"name": f"{kind}_subgraph", "graph": subgraph}}
            )
            if kind == "set_grad":
                target = "torch.ops.higher_order.wrap_with_set_grad_enabled"
                inputs = [
                    positional({"as_bool": enabled}),
                    graph_argument,
                    positional(tensor("x")),
                ]
            else:
                target = "torch.ops.higher_order.wrap_with_autocast"
                inputs = [
                    positional({"as_string": "cpu"}),
                    positional({"as_scalar_type": 13}),
                    positional({"as_bool": enabled}),
                    positional({"as_bool": False}),
                    graph_argument,
                    positional(tensor("x")),
                ]
            graph["nodes"] = [
                {
                    "name": kind,
                    "target": target,
                    "inputs": inputs,
                    "outputs": [tensor(output_name)],
                    "metadata": {},
                    "is_hop_single_tensor_return": None,
                }
            ]

        model = SingleTupleModel().eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "higher_order_source.pt2"
            save_exported_program(model, source_path)
            torch.manual_seed(0)
            expected = model(torch.rand(2, 4))
            for kind, label in (("set_grad", "set-grad"), ("autocast", "autocast")):
                with self.subTest(wrapper=kind):
                    positive_path = work_dir / f"{kind}_disabled.pt2"
                    negative_path = work_dir / f"{kind}_enabled.pt2"
                    rewrite_model_json(
                        source_path,
                        positive_path,
                        lambda document, kind=kind: inject_wrapper(
                            document, kind, False
                        ),
                    )
                    self.assert_conversion_matches(
                        work_dir, positive_path, expected
                    )
                    rewrite_model_json(
                        source_path,
                        negative_path,
                        lambda document, kind=kind: inject_wrapper(
                            document, kind, True
                        ),
                    )
                    self.assert_conversion_fails(
                        work_dir,
                        negative_path,
                        f"enabled {label} higher-order graph is unsupported",
                    )

    def test_json_parser_rejects_unsafe_boundaries(self):
        cases = (
            (b'{"graph_module":null,', "duplicate key graph_module"),
            (b'{"invalid":"\\ud800",', "unpaired high surrogate"),
            (b'{"invalid":"\x80",', "utf-8"),
            (b'{"invalid":18446744073709551616,', "integer overflow"),
            (
                b'{"deep":' + b"[" * 257 + b"null" + b"]" * 257 + b",",
                "depth limit",
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "json_source.pt2"
            save_exported_program(self.model, source_path)
            for index, (prefix, detail) in enumerate(cases):
                with self.subTest(boundary=detail):
                    archive_path = work_dir / f"json_invalid_{index}.pt2"
                    model_entry = ""

                    def corrupt_model(entries):
                        nonlocal model_entry
                        model_entry = archive_entry(
                            entries, "/models/", ".json"
                        )
                        entries[model_entry] = prefix + entries[model_entry][1:]

                    rewrite_archive(source_path, archive_path, corrupt_model)
                    self.assert_conversion_fails(
                        work_dir,
                        archive_path,
                        "invalid json",
                        model_entry,
                        detail,
                    )

    def test_data_descriptors_and_central_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "descriptor_source.pt2"
            descriptor_path = work_dir / "descriptor.pt2"
            crc_path = work_dir / "crc_invalid.pt2"
            corrupt_path = work_dir / "central_invalid.pt2"
            save_exported_program(self.model, source_path)

            with zipfile.ZipFile(source_path, "r") as source:
                entries = [
                    (info.filename, source.read(info))
                    for info in source.infolist()
                ]
            stream = NonSeekableBuffer()
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_STORED
            ) as destination:
                for name, data in entries:
                    destination.writestr(name, data)
            descriptor_path.write_bytes(stream.getvalue())

            with zipfile.ZipFile(descriptor_path, "r") as archive:
                self.assertTrue(
                    all(info.flag_bits & 0x08 for info in archive.infolist())
                )
            self.assert_conversion_matches(work_dir, descriptor_path)

            with zipfile.ZipFile(source_path, "r") as archive:
                payload = next(
                    info
                    for info in archive.infolist()
                    if "/data/weights/" in info.filename
                    and not info.filename.endswith(".json")
                )
            data = bytearray(source_path.read_bytes())
            name_size = int.from_bytes(
                data[payload.header_offset + 26 : payload.header_offset + 28],
                "little",
            )
            extra_size = int.from_bytes(
                data[payload.header_offset + 28 : payload.header_offset + 30],
                "little",
            )
            payload_offset = payload.header_offset + 30 + name_size + extra_size
            data[payload_offset] ^= 1
            crc_path.write_bytes(data)
            self.assert_conversion_fails(
                work_dir,
                crc_path,
                "zip entry crc mismatch",
                "load exported program failed",
            )

            data = bytearray(source_path.read_bytes())
            eocd = data.rfind(b"PK\x05\x06")
            self.assertGreaterEqual(eocd, 0)
            central_offset = int.from_bytes(
                data[eocd + 16 : eocd + 20], "little"
            )
            self.assertEqual(
                data[central_offset : central_offset + 4], b"PK\x01\x02"
            )
            data[central_offset : central_offset + 4] = b"BAD!"
            corrupt_path.write_bytes(data)
            self.assert_conversion_fails(
                work_dir,
                corrupt_path,
                "invalid zip central directory file header",
                "detect model format failed",
            )

    def test_zip64_local_headers_convert(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "zip64_source.pt2"
            archive_path = work_dir / "zip64.pt2"
            save_exported_program(self.model, source_path)
            with zipfile.ZipFile(source_path, "r") as source:
                entries = [
                    (info.filename, source.read(info))
                    for info in source.infolist()
                ]
            with zipfile.ZipFile(
                archive_path,
                "w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as destination:
                for name, data in entries:
                    with destination.open(name, "w", force_zip64=True) as entry:
                        entry.write(data)
            self.assert_conversion_matches(work_dir, archive_path)

    def test_zip_entry_count_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "entry_count_source.pt2"
            save_exported_program(self.model, source_path)
            with zipfile.ZipFile(source_path) as source:
                entries = [(info.filename, source.read(info)) for info in source.infolist()]
            root = entries[0][0].split("/", 1)[0]

            for count in (65534, 65535, 65536):
                with self.subTest(entries=count):
                    archive_path = work_dir / f"entry_count_{count}.pt2"
                    with zipfile.ZipFile(archive_path, "w") as archive:
                        for name, data in entries:
                            archive.writestr(name, data)
                        for index in range(count - len(entries)):
                            archive.writestr(f"{root}/extra/filler_{index}", b"")
                    data = bytearray(archive_path.read_bytes())
                    eocd = data.rfind(b"PK\x05\x06")
                    self.assertEqual(data[eocd - 20:eocd - 16] == b"PK\x06\x07", count > 65535)
                    self.assert_conversion_matches(work_dir, archive_path)

                    if count > 65535:
                        corrupt_path = work_dir / "missing_zip64_locator.pt2"
                        data[eocd - 20:eocd - 16] = b"BAD!"
                        corrupt_path.write_bytes(data)
                        self.assert_conversion_fails(work_dir, corrupt_path, "detect model format failed")

    def test_compressed_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "compressed_source.pt2"
            save_exported_program(self.model, source_path)
            with zipfile.ZipFile(source_path, "r") as source:
                entries = [
                    (info.filename, source.read(info))
                    for info in source.infolist()
                ]
            root = entries[0][0].split("/", 1)[0]
            cases = (
                (
                    "unused_attachment",
                    f"{root}/extra/note.txt",
                    b"unused attachment",
                    True,
                ),
                (
                    "model_json",
                    archive_entry(dict(entries), "/models/", ".json"),
                    None,
                    False,
                ),
                (
                    "weights_config",
                    archive_entry(dict(entries), "", "_weights_config.json"),
                    None,
                    False,
                ),
                (
                    "weight_blob",
                    next(
                        name
                        for name, _ in entries
                        if "/data/weights/" in name
                        and not name.endswith(".json")
                    ),
                    None,
                    False,
                ),
            )
            for label, target, attachment, converts in cases:
                with self.subTest(entry=label):
                    archive_path = work_dir / f"compressed_{label}.pt2"
                    archive_entries = list(entries)
                    if attachment is not None:
                        archive_entries.append((target, attachment))
                    with zipfile.ZipFile(archive_path, "w") as destination:
                        for name, data in archive_entries:
                            compression = (
                                zipfile.ZIP_DEFLATED
                                if name == target
                                else zipfile.ZIP_STORED
                            )
                            destination.writestr(
                                name, data, compress_type=compression
                            )
                    if converts:
                        self.assert_conversion_matches(work_dir, archive_path)
                    else:
                        self.assert_conversion_fails(
                            work_dir,
                            archive_path,
                            "compressed pt2 entry is unsupported",
                        )

    def test_legacy_pickled_payload_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "legacy_layout_source.pt2"
            archive_path = work_dir / "legacy_layout.pt2"
            save_exported_program(self.model, source_path)

            def replace_configs_with_legacy_entries(entries):
                weights_path = archive_entry(
                    entries, "", "_weights_config.json"
                )
                constants_path = archive_entry(
                    entries, "", "_constants_config.json"
                )
                del entries[weights_path]
                del entries[constants_path]
                entries[
                    weights_path[: -len("_weights_config.json")] + ".pt"
                ] = b"synthetic legacy weights"
                entries[
                    constants_path[: -len("_constants_config.json")] + ".pt"
                ] = b"synthetic legacy constants"

            rewrite_archive(
                source_path,
                archive_path,
                replace_configs_with_legacy_entries,
            )
            self.assert_conversion_fails(
                work_dir,
                archive_path,
                "PyTorch 2.8 legacy pickled-payload PT2 is unsupported",
            )

    def test_missing_and_truncated_payloads_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "payload_source.pt2"
            save_exported_program(self.model, source_path)

            for mode in ("missing", "truncated"):
                with self.subTest(mode=mode):
                    archive_path = work_dir / f"payload_{mode}.pt2"
                    payload_path = ""

                    def mutate(entries):
                        nonlocal payload_path
                        config_path = archive_entry(
                            entries, "", "_weights_config.json"
                        )
                        config = json.loads(entries[config_path])["config"]
                        path_name = config["linear.weight"]["path_name"]
                        root = config_path.split("/data/weights/", 1)[0]
                        payload_path = f"{root}/data/weights/{path_name}"
                        if mode == "missing":
                            del entries[payload_path]
                        else:
                            entries[payload_path] = entries[payload_path][:4]

                    rewrite_archive(source_path, archive_path, mutate)
                    messages = (
                        (payload_path, "is missing")
                        if mode == "missing"
                        else (
                            "linear.weight",
                            payload_path,
                            "tensor view exceeds storage",
                        )
                    )
                    self.assert_conversion_fails(
                        work_dir, archive_path, *messages
                    )

    def test_payload_kind_and_shape_are_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "payload_contract_source.pt2"
            save_exported_program(self.model, source_path)

            wrong_kind_path = work_dir / "payload_wrong_kind.pt2"

            def duplicate_parameter_into_constants(weights, constants):
                constants["linear.weight"] = dict(weights["linear.weight"])

            rewrite_payload_configs(
                source_path,
                wrong_kind_path,
                duplicate_parameter_into_constants,
            )
            self.assert_conversion_fails(
                work_dir,
                wrong_kind_path,
                "parameter linear.weight is present in constants config",
            )

            wrong_shape_path = work_dir / "payload_wrong_shape.pt2"

            def transpose_parameter_shape(weights, constants):
                tensor_meta = weights["linear.weight"]["tensor_meta"]
                tensor_meta["sizes"] = [{"as_int": 4}, {"as_int": 3}]
                tensor_meta["strides"] = [{"as_int": 3}, {"as_int": 1}]

            rewrite_payload_configs(
                source_path, wrong_shape_path, transpose_parameter_shape
            )
            self.assert_conversion_fails(
                work_dir,
                wrong_shape_path,
                "parameter linear.weight shape does not match tensor metadata",
            )


if __name__ == "__main__":
    if TORCH_VERSION < (2, 9):
        print("modern exported program tests require PyTorch 2.9 or newer")
        sys.exit(77)
    unittest.main()
