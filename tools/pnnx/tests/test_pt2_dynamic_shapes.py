# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import copy
import importlib.util
import json
import os
import pathlib
import subprocess
import zipfile

import torch

from pnnx_test_utils import _check_output, _run_ncnn


class Model(torch.nn.Module):
    def forward(self, x):
        return x[:x.shape[0] - 1]


def load_model(path):
    spec = importlib.util.spec_from_file_location("pnnx_dynamic_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
        print("PT2 test skipped: PyTorch 2.6 or newer is required")
        return 77

    args.workdir.mkdir(parents=True, exist_ok=True)
    model = Model().eval()
    exported = torch.export.export(
        model,
        (torch.randn(4, 3),),
        dynamic_shapes=({0: torch.export.Dim("batch", min=3, max=8)},),
    )
    torch.export.save(exported, args.workdir / "model.pt2")

    command = [
        str(pathlib.Path(args.pnnx).resolve()),
        "model.pt2",
        "inputshape=[6,3]",
        "inputshape2=[4,3]",
    ]
    process = subprocess.run(command, cwd=args.workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        print(process.stdout)
        return 1

    previous_workdir = pathlib.Path.cwd()
    try:
        os.chdir(args.workdir)
        converted = load_model(args.workdir / "model_pnnx.py").eval()
        for shape in ((6, 3), (4, 3)):
            x = torch.randn(shape)
            expected = model(x)
            _check_output(expected, converted(x))
            actual = _run_ncnn(args.workdir, (x,), 1)
            _check_output(expected, actual, 1e-3, 1e-3, True)
    finally:
        os.chdir(previous_workdir)

    invalid = subprocess.run(
        [str(pathlib.Path(args.pnnx).resolve()), "model.pt2", "inputshape=[9,3]"],
        cwd=args.workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if invalid.returncode == 0 or "outside the exported range" not in invalid.stdout:
        print(invalid.stdout)
        return 1

    invalid = subprocess.run(
        [str(pathlib.Path(args.pnnx).resolve()), "model.pt2", "inputshape=[6,3]f16"],
        cwd=args.workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if invalid.returncode == 0 or "type must be f32" not in invalid.stdout:
        print(invalid.stdout)
        return 1

    records = {}
    with zipfile.ZipFile(args.workdir / "model.pt2") as archive:
        for info in archive.infolist():
            records[info.filename] = archive.read(info)
    model_record = next(name for name in records if name.endswith("models/model.json") or name.endswith("serialized_exported_program.json"))
    program = json.loads(records[model_record])
    input_name = program["graph_module"]["graph"]["inputs"][0]["as_tensor"]["name"]
    input_size = program["graph_module"]["graph"]["tensor_values"][input_name]["sizes"][0]
    derived_name = None
    derived_index = None
    for name, tensor in program["graph_module"]["graph"]["tensor_values"].items():
        if name == input_name:
            continue
        for i, size in enumerate(tensor["sizes"]):
            if "as_expr" in size:
                derived_name = name
                derived_index = i
                break
        if derived_name is not None:
            break
    if derived_name is None:
        return 1

    symbol = input_size["as_expr"]["expr_str"]
    deep = symbol
    for _ in range(64):
        deep = "Add(Integer(1), %s)" % deep
    cases = [
        ("unsupported", False, "Unsupported()", "unsupported symbolic expression Unsupported()"),
        ("unknown_assumption", True, symbol.replace("positive=True", "unknown=True"), "unsupported symbolic input dimension"),
        ("empty_symbol", True, "Symbol('', positive=True, integer=True)", "unsupported symbolic input dimension"),
        ("deep", False, deep, "unsupported symbolic expression"),
        ("negative", False, "Integer(-1)", "symbolic expression evaluates to a negative dimension"),
    ]
    for name, mutate_input, expression, expected in cases:
        mutated = copy.deepcopy(program)
        tensors = mutated["graph_module"]["graph"]["tensor_values"]
        size = tensors[input_name]["sizes"][0] if mutate_input else tensors[derived_name]["sizes"][derived_index]
        size["as_expr"]["expr_str"] = expression
        mutated_records = dict(records)
        mutated_records[model_record] = json.dumps(mutated, separators=(",", ":")).encode("utf-8")
        path = name + ".pt2"
        with zipfile.ZipFile(args.workdir / path, "w") as archive:
            for record_name, data in mutated_records.items():
                archive.writestr(record_name, data)
        invalid = subprocess.run(
            [str(pathlib.Path(args.pnnx).resolve()), path, "inputshape=[6,3]"],
            cwd=args.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if invalid.returncode == 0 or expected not in invalid.stdout:
            print(invalid.stdout)
            return 1

    guard_cases = [
        ("guard_supported", "L['%s'].size()[0] != max(1, L['%s'].size()[0] // 2)" % (input_name, input_name), True, ""),
        ("guard_false", "L['%s'].size()[0] < 6" % input_name, False, "violates runtime guard"),
        ("guard_unsupported", "L['%s'].stride()[0] == 3" % input_name, False, "unsupported runtime guard"),
    ]
    for name, guard, valid, expected in guard_cases:
        mutated = copy.deepcopy(program)
        mutated["guards_code"] = [guard]
        mutated_records = dict(records)
        mutated_records[model_record] = json.dumps(mutated, separators=(",", ":")).encode("utf-8")
        path = name + ".pt2"
        with zipfile.ZipFile(args.workdir / path, "w") as archive:
            for record_name, data in mutated_records.items():
                archive.writestr(record_name, data)
        invalid = subprocess.run(
            [str(pathlib.Path(args.pnnx).resolve()), path, "inputshape=[6,3]"],
            cwd=args.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if valid != (invalid.returncode == 0) or (expected and expected not in invalid.stdout):
            print(invalid.stdout)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
