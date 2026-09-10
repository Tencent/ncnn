# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""PT2 Python input contracts, not native ncnn runtime guard support.

Parent registration (inside PNNX_TEST_HAS_PT2): add_test named
test_pnnx_exported_program_guards invoking Python3_EXECUTABLE on this script
with --pnnx-executable $<TARGET_FILE:pnnx>. Use labels
"pt2;pt2_frontend;python;robustness" and PNNX_TEST_CASE_TIMEOUT.

Every conversion uses the real CLI through run_pnnx; only executable discovery
is mocked. Artifacts live in a temporary directory with POSIX-form paths.
Saved Input parameters are checked, including scalar/empty-vector encodings.
The CLI has no Graph::load entry point: native save/load roundtrip execution
belongs in the parent's C++ IR test, not a Python imitation of Graph::load.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import re
import tempfile
from unittest import mock
import zipfile

import torch
from torch import nn

import pnnx_test_utils


class Add(nn.Module):
    def forward(self, x, y):
        return x + y


class Twice(nn.Module):
    def forward(self, x):
        return x + x


def require(condition, message):
    # Keep the test itself effective when invoked under python -O/-OO.
    if not condition:
        raise AssertionError(message)


def check_output(actual, expected):
    require(isinstance(actual, torch.Tensor), type(actual))
    require(actual.dtype == expected.dtype and actual.shape == expected.shape,
            (actual.dtype, actual.shape, expected.dtype, expected.shape))
    require(torch.equal(actual, expected), "generated output differs from Torch")


def reject(model, inputs, needle):
    try:
        model(*inputs)
    except ValueError as error:
        require("PT2 input" in str(error) and needle in str(error), str(error))
    else:
        raise AssertionError("missing runtime rejection: " + needle)


def convert(path, prefix, optlevel, timeout, arguments=(), failure=None):
    result = pnnx_test_utils.run_pnnx(
        path.as_posix(), prefix.as_posix(),
        ("optlevel=" + str(optlevel), *arguments),
        capture_output=True, timeout=timeout,
    )
    diagnostic = result.stdout + "\n" + result.stderr
    if failure is not None:
        require(result.returncode != 0 and failure in diagnostic, diagnostic)
        require(not Path(prefix.as_posix() + "_pnnx.py").exists(),
                "failed conversion retained generated Python")
        return
    require(result.returncode == 0, diagnostic)
    require(Path(prefix.as_posix() + "_pnnx.py").is_file(), diagnostic)


def generated_models(prefix):
    path = Path(prefix.as_posix() + "_pnnx.py")
    source = path.read_text(encoding="utf-8")
    for optimization in (0, 1, 2):
        # Compiling the entire generated module with optimize=1/2 exercises
        # -O/-OO semantics without another process or bytecode-cache reuse.
        namespace = {"__name__": "pnnx_guard_fixture", "__file__": path.as_posix()}
        exec(compile(source, path.as_posix(), "exec", optimize=optimization), namespace)
        yield namespace["Model"]().eval()


def saved_contracts(prefix):
    text = Path(prefix.as_posix() + ".pnnx.param").read_text(encoding="utf-8")
    contracts = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != "pnnx.Input":
            continue
        contract = {}
        for field in fields:
            if not field.startswith("__pt2_input_"):
                continue
            key, value = field.split("=", 1)
            if key == "__pt2_input_type":
                contract[key] = int(value)
            else:
                require(value.startswith("(") and value.endswith(")"), (key, value))
                values = value[1:-1].split(",") if value != "()" else []
                if key == "__pt2_input_symbols":
                    require(all(v == "-" or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", v)
                                for v in values), values)
                    contract[key] = tuple("" if v == "-" else v for v in values)
                else:
                    contract[key] = tuple(map(int, values))
        contracts.append(contract)
    native = Path(prefix.as_posix() + ".ncnn.param").read_text(encoding="utf-8")
    require("__pt2_input_" not in native, "Python contract leaked into native ncnn params")
    return contracts


def check_common_rejections(model, inputs):
    # Test each input independently; meta requires no accelerator, and sparse
    # COO distinguishes layout from the (still CPU) device check.
    for index, tensor in enumerate(inputs):
        variants = (
            (tensor.double(), "dtype"),
            (tensor.unsqueeze(0), "rank"),
            (torch.empty(tensor.shape, device="meta", dtype=tensor.dtype), "CPU device"),
            (tensor.to_sparse(), "strided layout"),
            (None, "must be a tensor"),
        )
        for invalid, needle in variants:
            candidate = list(inputs)
            candidate[index] = invalid
            reject(model, tuple(candidate), needle)


def test_real_dynamic(directory, timeout):
    eager = Add().eval()
    sample = (torch.randn(3, 3), torch.randn(3, 3))
    batch = torch.export.Dim("batch", min=2, max=8)
    exported = torch.export.export(eager, sample, dynamic_shapes={"x": {0: batch}, "y": {0: batch}})
    reference = exported.module()
    path = directory / "shared_batch.pt2"
    torch.export.save(exported, path.as_posix())
    for optlevel in (0, 1, 2):
        for specialized in (False, True):
            prefix = directory / ("shared_%d_%d" % (optlevel, specialized))
            arguments = ("inputshape=[5,3],[5,3]",) if specialized else ()
            convert(path, prefix, optlevel, timeout, arguments)
            contracts = saved_contracts(prefix)
            require(len(contracts) == 2, contracts)
            for contract in contracts:
                require(contract["__pt2_input_type"] == 1, contract)
                require(contract["__pt2_input_shape"] == (-233, 3), contract)
                require(contract["__pt2_input_min"] == (2, 3), contract)
                require(contract["__pt2_input_max"] == (8, 3), contract)
                require(contract["__pt2_input_symbols"][1] == "", contract)
                require("__pt2_input_stride" not in contract, contract)
            require(contracts[0]["__pt2_input_symbols"][0]
                    == contracts[1]["__pt2_input_symbols"][0], contracts)
            for model in generated_models(prefix):
                # Fresh values each call prove shared symbols are call-local,
                # rather than cached at the first/example/specialization size.
                for size in (2, 4, 8, 3, 2):
                    inputs = (torch.randn(size, 3), torch.randn(size, 3))
                    check_output(model(*inputs), reference(*inputs))
                noncontiguous = (torch.randn(3, 4).t(), torch.randn(4, 6)[:, ::2])
                require(all(not t.is_contiguous() for t in noncontiguous), "bad stride fixture")
                check_output(model(*noncontiguous), eager(*noncontiguous))
                for size in (0, 1, 9):
                    reject(model, (torch.randn(size, 3), torch.randn(size, 3)), "allowed range is [2, 8]")
                reject(model, (torch.randn(2, 3), torch.randn(3, 3)), "shared symbol mismatch")
                for index in (0, 1):
                    inputs = list(sample)
                    inputs[index] = torch.randn(3, 4)
                    reject(model, tuple(inputs), "dimension 1 must be 3")
                check_common_rejections(model, sample)


def test_static_and_torchscript(directory, timeout):
    eager = Twice().eval()
    sample = (torch.randn(2, 3),)
    path = directory / "static.pt2"
    torch.export.save(torch.export.export(eager, sample), path.as_posix())
    traced = directory / "static.pt"
    # This control verifies only that TS does not acquire PT2 input contracts.
    # Keep it independent of raw ATen keyword binding at optlevel=0.
    traced_eager = nn.Identity().eval()
    torch.jit.trace(traced_eager, sample).save(traced.as_posix())
    for optlevel in (0, 1, 2):
        prefix = directory / ("static_%d" % optlevel)
        convert(path, prefix, optlevel, timeout, ("inputshape=[2,3]",))
        contract, = saved_contracts(prefix)
        require(contract["__pt2_input_shape"] == (2, 3), contract)
        require(contract["__pt2_input_symbols"] == ("", ""), contract)
        require("__pt2_input_stride" not in contract, contract)
        for model in generated_models(prefix):
            check_output(model(*sample), eager(*sample))
            alternate = torch.randn(3, 2).t()
            check_output(model(alternate), eager(alternate))
            reject(model, (torch.randn(4, 3),), "dimension 0 must be 2")
            reject(model, (torch.randn(2, 4),), "dimension 1 must be 3")
            check_common_rejections(model, sample)
        convert(path, directory / ("static_bad_%d" % optlevel), optlevel, timeout,
                ("inputshape=[4,3]",), failure="expected 2")

        prefix = directory / ("torchscript_%d" % optlevel)
        convert(traced, prefix, optlevel, timeout, ("inputshape=[2,3]",))
        require(saved_contracts(prefix) == [{}], "TorchScript acquired a PT2 contract")
        for model in generated_models(prefix):
            alternate = torch.randn(4, 3, dtype=torch.float64)
            check_output(model(alternate), traced_eager(alternate))


def root_document(shape=(2, 3)):
    tensor = {"as_tensor": {"name": "x"}}
    return {
        "graph_module": {
            "graph": {
                "inputs": [copy.deepcopy(tensor)], "outputs": [copy.deepcopy(tensor)],
                "nodes": [], "sym_int_values": {}, "is_single_tensor_return": True,
                "tensor_values": {"x": {
                    "dtype": 7, "sizes": [{"as_int": n} for n in shape],
                    "strides": [{"as_int": n} for n in torch.empty(shape).stride()],
                    "requires_grad": False, "device": {"type": "cpu", "index": None},
                    "storage_offset": {"as_int": 0}, "layout": 7,
                }},
            },
            "signature": {
                "input_specs": [{"user_input": {"arg": copy.deepcopy(tensor)}}],
                "output_specs": [{"user_output": {"arg": copy.deepcopy(tensor)}}],
            },
            "module_call_graph": [],
        },
        "schema_version": {"major": 8, "minor": 20},
        "opset_version": {"aten": torch._C._get_max_operator_version()},
        "range_constraints": {},
    }


def save_root(path, document):
    # No wrapper directory, producer pickle or optional extra records: exercise
    # the public root-schema archive path independently of export guard folding.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive_format", "pt2")
        archive.writestr("archive_version", "0")
        archive.writestr("models/model.json", json.dumps(document))
        archive.writestr("data/weights/model_weights_config.json", '{"config":{}}')
        archive.writestr("data/constants/model_constants_config.json", '{"config":{}}')


def metadata_guard(stride, void_none):
    return {
        "name": "check_x", "target": "torch.ops.aten._assert_tensor_metadata.default",
        # Deliberately non-schema order: append/default normalization runs twice.
        "inputs": [
            {"name": "dtype", "arg": {"as_scalar_type": 7}},
            {"name": "stride", "arg": {"as_none": True} if stride is None else {"as_ints": stride}},
            {"name": "a", "arg": {"as_tensor": {"name": "x"}}},
            {"name": "size", "arg": {"as_ints": [2, 3]}},
            {"name": "device", "arg": {"as_device": {"type": "cpu", "index": None}}},
            {"name": "layout", "arg": {"as_layout": 7}},
        ],
        "outputs": [{"as_none": True}] if void_none else [], "metadata": {},
    }


def test_root_metadata_guards(directory, timeout):
    for optlevel in (0, 1, 2):
        for mode in ("explicit", "none", "omitted"):
            for void_none in (False, True):
                document = root_document()
                guard = metadata_guard([3, 1] if mode == "explicit" else None, void_none)
                if mode == "omitted":
                    guard["inputs"] = [a for a in guard["inputs"] if a["name"] != "stride"]
                document["graph_module"]["graph"]["nodes"] = [guard]
                name = "guard_%s_%d_%d" % (mode, void_none, optlevel)
                path, prefix = directory / (name + ".pt2"), directory / name
                save_root(path, document)
                convert(path, prefix, optlevel, timeout, ("inputshape=[2,3]",))
                contract, = saved_contracts(prefix)
                require(("__pt2_input_stride" in contract) == (mode == "explicit"), contract)
                if mode == "explicit":
                    require(contract["__pt2_input_stride"] == (3, 1), contract)
                source = Path(prefix.as_posix() + "_pnnx.py").read_text(encoding="utf-8")
                require("_assert_tensor_metadata" not in source, "guard was not discharged")
                for model in generated_models(prefix):
                    tensor = torch.randn(2, 3)
                    check_output(model(tensor), tensor)
                    check_common_rejections(model, (tensor,))
                    reject(model, (torch.randn(2, 4),), "dimension 1 must be 3")
                    # Storage offset is not guarded, even when stride is explicit.
                    offset = torch.randn(3, 3)[1:]
                    check_output(model(offset), offset)
                    transposed = torch.randn(3, 2).t()
                    if mode == "explicit":
                        reject(model, (transposed,), "stride")
                    else:
                        check_output(model(transposed), transposed)

        document = root_document()
        document["graph_module"]["graph"]["nodes"] = [metadata_guard([1, 2], False)]
        path = directory / "false_stride.pt2"
        save_root(path, document)
        convert(path, directory / ("false_stride_%d" % optlevel), optlevel, timeout,
                failure="metadata guard is false or unsupported")

        # Rank zero must serialize each contract vector as (), with no loss of
        # dtype/rank validation. Native Graph::load restores reserved types.
        path, prefix = directory / "scalar.pt2", directory / ("scalar_%d" % optlevel)
        save_root(path, root_document(()))
        convert(path, prefix, optlevel, timeout)
        contract, = saved_contracts(prefix)
        for key in ("shape", "symbols", "min", "max"):
            require(contract["__pt2_input_" + key] == (), contract)
        for model in generated_models(prefix):
            tensor = torch.tensor(1.25)
            check_output(model(tensor), tensor)
            reject(model, (tensor.reshape(1),), "rank")
            reject(model, (tensor.double(),), "dtype")


def test_root_symbols_and_unknown_guards(directory, timeout):
    document = root_document((3, 3))
    meta = document["graph_module"]["graph"]["tensor_values"]["x"]
    meta["sizes"] = [
        {"as_expr": {"expr_str": "Symbol('s0', integer=True, positive=True)", "hint": {"as_int": 3}}},
        {"as_expr": {"expr_str": "s0", "hint": {"as_int": 3}}},
    ]
    document["range_constraints"] = {"s0": {"min_val": 0, "max_val": 2**50}}
    for optlevel in (0, 1, 2):
        path, prefix = directory / "symbols.pt2", directory / ("symbols_%d" % optlevel)
        save_root(path, document)
        convert(path, prefix, optlevel, timeout)
        contract, = saved_contracts(prefix)
        require(contract["__pt2_input_symbols"] == ("s0", "s0"), contract)
        require(contract["__pt2_input_min"] == (1, 0), contract)
        require(contract["__pt2_input_max"] == (2**31 - 1, 2**31 - 1), contract)
        for model in generated_models(prefix):
            for size in (1, 2, 5):
                tensor = torch.randn(size, size)
                check_output(model(tensor), tensor)
            reject(model, (torch.randn(2, 3),), "shared symbol mismatch")
            reject(model, (torch.empty(0, 0),), "allowed range")
            # A zero-stride view tests the int cap without allocating GiBs.
            reject(model, (torch.zeros(1, 3).expand(2**31, 3),), "allowed range")

        for name, expression in (
            ("derived", "Add(Symbol('s0'), Integer(1))"),
            ("injection", "Symbol('s0');__import__('builtins').print('PNNX_GUARD_INJECTION')"),
        ):
            invalid = copy.deepcopy(document)
            invalid["graph_module"]["graph"]["tensor_values"]["x"]["sizes"][0]["as_expr"]["expr_str"] = expression
            path = directory / (name + ".pt2")
            save_root(path, invalid)
            convert(path, directory / ("%s_%d" % (name, optlevel)), optlevel, timeout,
                    failure="unvalidated derived input expression")

        invalid = copy.deepcopy(document)
        invalid["range_constraints"]["s0"] = {"min_val": 2**31, "max_val": 2**50}
        path = directory / "impossible_range.pt2"
        save_root(path, invalid)
        convert(path, directory / ("impossible_%d" % optlevel), optlevel, timeout,
                failure="allowed range has no supported pnnx dimension")

        invalid = root_document()
        invalid["graph_module"]["graph"]["nodes"] = [{
            "name": "unknown_guard", "target": "torch.ops.aten._assert_async.msg",
            "inputs": [{"name": "self", "arg": {"as_tensor": {"name": "x"}}},
                       {"name": "assert_msg", "arg": {"as_string": "not a supported guard"}}],
            "outputs": [], "metadata": {},
        }]
        path = directory / "unknown_guard.pt2"
        save_root(path, invalid)
        convert(path, directory / ("unknown_%d" % optlevel), optlevel, timeout,
                failure="unsupported guard")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnnx-executable", default=os.environ.get("PNNX_EXECUTABLE"))
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()
    if not pnnx_test_utils.has_exported_program():
        print("SKIP: torch.export.save is unavailable in torch " + torch.__version__)
        return
    executable = Path(args.pnnx_executable or pnnx_test_utils.find_pnnx()).resolve()
    require(executable.is_file(), executable)
    torch.manual_seed(2026)
    with tempfile.TemporaryDirectory(prefix="pnnx_guards_") as temporary:
        directory = Path(temporary)
        with mock.patch.object(pnnx_test_utils, "find_pnnx", return_value=executable.as_posix()), torch.no_grad():
            test_real_dynamic(directory, args.timeout)
            test_static_and_torchscript(directory, args.timeout)
            test_root_metadata_guards(directory, args.timeout)
            test_root_symbols_and_unknown_guards(directory, args.timeout)
    print("PASS: PT2 Python input guards at pnnx optlevel=0/1/2 and Python optimize=0/1/2")


if __name__ == "__main__":
    main()