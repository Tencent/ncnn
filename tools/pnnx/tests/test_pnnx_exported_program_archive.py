# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""Public PT2 archive contract regressions (current raw serde, not full PyTree).

Run manually with --pnnx-executable, or set PNNX_EXECUTABLE. All converter
invocations use run_pnnx's timeout/crash checks, including expected failures.
The parent CMake registration must pass the built pnnx target explicitly.
No exporter/version/operator failure is treated as a skip.
"""

import argparse
from collections import namedtuple
import copy
import json
import os
from pathlib import Path
import struct
import tempfile
from unittest import mock
import warnings
import zipfile

import torch
from torch import nn
from torch.utils import _pytree

import pnnx_test_utils


class Simple(nn.Module):
    def forward(self, x, y):
        return x + y


class Buffers(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        self.register_buffer("bias", torch.tensor([.5, 1., 1.5]))
        self.register_buffer("scratch", torch.tensor([2., 3., 4.]), persistent=False)

    def forward(self, x):
        return x * self.weight + self.bias + self.scratch, self.scratch


class FlatOutputs(nn.Module):
    def forward(self, x):
        return x + 1, x * 2, 7, 1.5, True


class Singleton(nn.Module):
    def forward(self, x):
        return (x + 1,)


class KeywordInputs(nn.Module):
    def forward(self, x, *, y):
        return x + y


class DictInputs(nn.Module):
    def forward(self, values):
        return values["x"] + values["y"]


class NestedInputs(nn.Module):
    def forward(self, values):
        return values[0] + values[1][0]


class ScalarInputs(nn.Module):
    def forward(self, x, scale):
        return x * scale


class DictOutputs(nn.Module):
    def forward(self, x):
        return {"value": x + 1}


class ListOutputs(nn.Module):
    def forward(self, x):
        return [x + 1, x * 2]


class NestedOutputs(nn.Module):
    def forward(self, x):
        return x + 1, (x * 2,)


ArchivePair = namedtuple("ArchivePair", ("first", "second"))


class NamedTupleOutputs(nn.Module):
    def forward(self, x):
        return ArchivePair(x + 1, x * 2)


class NamedTupleInputs(nn.Module):
    def forward(self, values):
        return values.first + values.second


def check_outputs(actual, expected):
    # Only the existing top-level singleton tuple collapse is intentional.
    # Never flatten nested trees or equate a list/dict with a tuple.
    if type(expected) is tuple and len(expected) == 1 and torch.is_tensor(actual):
        expected = expected[0]
    if type(expected) is tuple:
        assert type(actual) is tuple and len(actual) == len(expected), (actual, expected)
        for a, b in zip(actual, expected):
            check_outputs(a, b)
    elif torch.is_tensor(expected):
        assert torch.is_tensor(actual), type(actual)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected), (actual, expected)
    else:
        assert type(actual) is type(expected) and actual == expected, (actual, expected)


def save_export(directory, name, model, inputs, kwargs=None):
    model.eval()
    exported = torch.export.export(model, inputs, kwargs=kwargs)
    path = directory / (name + ".pt2")
    # Forward slashes also keep absolute generated Python binary paths valid
    # on Windows, where an unescaped backslash can otherwise change a literal.
    torch.export.save(exported, path.as_posix())
    return path, exported


def conversion(path, prefix, timeout):
    return pnnx_test_utils.run_pnnx(
        path.as_posix(), prefix.as_posix(), arguments=("optlevel=0",),
        capture_output=True, timeout=timeout,
    )


def check_success(path, prefix, model, inputs, timeout):
    result = conversion(path, prefix, timeout)
    diagnostic = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, diagnostic
    generated_path = prefix.as_posix() + "_pnnx.py"
    assert Path(generated_path).is_file(), diagnostic
    generated = pnnx_test_utils.import_model(generated_path)
    with torch.no_grad():
        check_outputs(generated(*inputs), model(*inputs))


def check_failure(path, prefix, needle, timeout):
    result = conversion(path, prefix, timeout)
    diagnostic = result.stdout + "\n" + result.stderr
    # run_pnnx already rejects crashes/signals/timeouts, so a diagnostic cannot
    # hide a crashing process. Do not accept a generic nonzero status alone.
    assert result.returncode != 0, "unexpectedly accepted " + path.name
    assert needle in diagnostic, (needle, diagnostic)
    print("EXPECTED REJECTION:", path.name, needle)


def records_from(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), names
        return {name: archive.read(name) for name in names}


def write_records(path, records, compressed=None, force_zip64=False):
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in records.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED if name == compressed else zipfile.ZIP_STORED
            if force_zip64:
                with archive.open(info, "w", force_zip64=True) as entry:
                    entry.write(payload)
            else:
                archive.writestr(info, payload)


def one_name(records, suffix):
    matches = [name for name in records if name.endswith(suffix)]
    assert len(matches) == 1, (suffix, matches)
    return matches[0]


def config_record(records, name, mutate):
    result = dict(records)
    config = json.loads(result[name])
    mutate(config["config"])
    result[name] = json.dumps(config).encode("utf-8")
    return result


def record_offsets(path, record):
    # Mutation inputs are small stdlib-written archives, not an untrusted ZIP
    # parser. The public converter, not zipfile, must diagnose the mutated ZIP.
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(record)
        central = archive.start_dir
    data = bytearray(path.read_bytes())
    local = info.header_offset
    assert data[local:local + 4] == b"PK\x03\x04"
    name_length, extra_length = struct.unpack_from("<HH", data, local + 26)
    payload = local + 30 + name_length + extra_length
    while data[central:central + 4] == b"PK\x01\x02":
        name_size, extra_size, comment_size = struct.unpack_from("<HHH", data, central + 28)
        name = bytes(data[central + 46:central + 46 + name_size]).decode("utf-8")
        if name == record:
            return data, local, central, payload
        central += 46 + name_size + extra_size + comment_size
    raise AssertionError("central record not found: " + record)


def test_call_structures(directory, x, y, timeout):
    for name, model, inputs in (
        ("simple", Simple(), (x, y)),
        ("flat_outputs", FlatOutputs(), (x,)),
        ("singleton", Singleton(), (x,)),
    ):
        path, _ = save_export(directory, name, model, inputs)
        check_success(path, directory / (name + "_converted"), model, inputs, timeout)

    _pytree._register_namedtuple(ArchivePair, serialized_type_name="pnnx.archive.ArchivePair")
    cases = (
        ("kwargs", KeywordInputs(), (x,), {"y": y}, "unsupported PyTree kwargs"),
        ("dict_inputs", DictInputs(), ({"x": x, "y": y},), None, "unsupported PyTree nested input"),
        ("nested_inputs", NestedInputs(), ((x, (y,)),), None, "unsupported PyTree nested input"),
        ("namedtuple_inputs", NamedTupleInputs(), (ArchivePair(x, y),), None, "unsupported PyTree nested input"),
        ("scalar_inputs", ScalarInputs(), (x, 2), None, "unsupported PyTree input leaf"),
        ("dict_outputs", DictOutputs(), (x,), None, "unsupported PyTree output"),
        ("list_outputs", ListOutputs(), (x,), None, "unsupported PyTree output"),
        ("nested_outputs", NestedOutputs(), (x,), None, "unsupported PyTree nested output"),
        ("namedtuple_outputs", NamedTupleOutputs(), (x,), None, "unsupported PyTree output"),
    )
    for name, model, inputs, kwargs, needle in cases:
        path, _ = save_export(directory, name, model, inputs, kwargs)
        check_failure(path, directory / (name + "_converted"), needle, timeout)


def test_archive_contract(directory, x, timeout):
    model = Buffers().eval()
    path, exported = save_export(directory, "buffers", model, (x,))
    assert "scratch" not in exported.state_dict
    assert "scratch" in exported.constants
    check_success(path, directory / "buffers_converted", model, (x,), timeout)
    records = records_from(path)
    weights_name = one_name(records, "/model_weights_config.json")
    constants_name = one_name(records, "/model_constants_config.json")
    model_name = one_name(records, "/models/model.json")
    root = model_name[:-len("models/model.json")]
    byteorder_name = root + "byteorder"
    assert records[byteorder_name] == b"little", "current raw tests require a little-endian torch producer"
    weights = json.loads(records[weights_name])["config"]
    constants = json.loads(records[constants_name])["config"]
    assert weights["weight"]["use_pickle"] is False
    assert constants["scratch"]["use_pickle"] is False
    payload_name = root + "data/weights/" + weights["weight"]["path_name"]
    assert len(records[payload_name]) > 0

    # Default byte order is not an arbitrarily mandatory metadata field. Test
    # both explicit little and absent marks with actual payload value checks.
    for name, candidate, force_zip64 in (
        ("stored", records, False),
        ("zip64", records, True),
        ("unmarked", {k: v for k, v in records.items() if k != byteorder_name}, False),
    ):
        target = directory / (name + ".pt2")
        write_records(target, candidate, force_zip64=force_zip64)
        check_success(target, directory / (name + "_converted"), model, (x,), timeout)

    # A harmless, unused attachment may be DEFLATED. The STORE-only reader
    # must leave it unopened/undecoded and preserve exact inference results.
    attachment_name = root + "extra/notes.txt"
    candidate = dict(records)
    assert attachment_name not in candidate
    candidate[attachment_name] = b"Unused export attachment.\n" * 64
    target = directory / "compressed_attachment.pt2"
    write_records(target, candidate, compressed=attachment_name)
    with zipfile.ZipFile(target) as archive:
        info = archive.getinfo(attachment_name)
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert info.compress_size < info.file_size
    check_success(target, directory / "compressed_attachment_converted", model, (x,), timeout)

    def rejected(name, candidate, needle, compressed=None):
        target = directory / (name + ".pt2")
        write_records(target, candidate, compressed=compressed)
        check_failure(target, directory / (name + "_converted"), needle, timeout)

    for name, field, value, needle in (
        ("dtype_float8", "dtype", 14, "unsupported tensor scalar type 14"),
        ("dtype_unknown", "dtype", 999, "unsupported tensor scalar type 999"),
        ("layout_sparse", "layout", 1, "Strided layout"),
        ("device_cuda", "device", {"type": "cuda", "index": 0}, "CPU device"),
        ("device_meta", "device", {"type": "meta", "index": None}, "CPU device"),
        ("device_cpu_index", "device", {"type": "cpu", "index": 1}, "CPU device"),
    ):
        def mutate(config, field=field, value=value):
            config["weight"]["tensor_meta"][field] = value
        rejected(name, config_record(records, weights_name, mutate), needle)

    def invalid_constant(config):
        config["scratch"]["tensor_meta"]["dtype"] = 999
    rejected("constant_dtype", config_record(records, constants_name, invalid_constant), "unsupported tensor scalar type 999")

    def pickled_weight(config):
        config["weight"]["use_pickle"] = True
    rejected("pickled_weight", config_record(records, weights_name, pickled_weight), "pickled tensor payload")

    for name, mark in (("byteorder_big", b"big"), ("byteorder_unknown", b"middle"),
                       ("byteorder_empty", b""), ("byteorder_newline", b"little\n")):
        candidate = dict(records)
        candidate[byteorder_name] = mark
        rejected(name, candidate, "byteorder:")

    def broadcast(meta):
        meta["sizes"] = [{"as_int": 67108864}]
        meta["strides"] = [{"as_int": 0}]
        meta["storage_offset"] = {"as_int": 0}

    def broadcast_weight(config):
        broadcast(config["weight"]["tensor_meta"])

    candidate = config_record(records, weights_name, broadcast_weight)
    candidate = config_record(candidate, constants_name, lambda config: broadcast(config["scratch"]["tensor_meta"]))
    rejected("aggregate_cross_dictionary", candidate, "aggregate 512 MiB")

    def aliases(config):
        broadcast_weight(config)
        config["weight_alias"] = copy.deepcopy(config["weight"])
    rejected("aggregate_shared_storage_views", config_record(records, weights_name, aliases), "aggregate 512 MiB")

    # Unknown protocols/malformed serialized trees must produce a call-spec
    # diagnostic, rather than accidentally passing after graph flattening.
    document = json.loads(records[model_name])
    root_call = next(call for call in document["graph_module"]["module_call_graph"] if call["fqn"] == "")
    in_spec = json.loads(root_call["signature"]["in_spec"])
    assert in_spec[0] == 1
    in_spec[0] = 2
    root_call["signature"]["in_spec"] = json.dumps(in_spec)
    candidate = dict(records)
    candidate[model_name] = json.dumps(document).encode("utf-8")
    rejected("treespec_protocol", candidate, "unsupported PyTree TreeSpec protocol")

    rejected("compressed_payload", records, "unsupported ZIP compression", payload_name)
    rejected("compressed_model", records, "unsupported ZIP compression", model_name)
    target = directory / "compressed_huge_declared.pt2"
    write_records(target, records, compressed=payload_name)
    data, local, central, _ = record_offsets(target, payload_name)
    # A tiny physical compressed record claiming 512 MiB must be refused before
    # allocating its advertised output buffer, even though the view needs 24 B.
    struct.pack_into("<I", data, local + 22, 512 * 1024 * 1024)
    struct.pack_into("<I", data, central + 24, 512 * 1024 * 1024)
    target.write_bytes(data)
    check_failure(target, directory / "compressed_huge_converted", "unsupported ZIP compression", timeout)

    for name, mutation, record, needle in (
        ("payload_crc", "crc", payload_name, "CRC mismatch"),
        ("record_flags", "flags", payload_name, "unsupported ZIP record flags"),
        ("payload_reserved_flags", "reserved_flags", payload_name, "unsupported ZIP record flags"),
        ("model_reserved_flags", "reserved_flags", model_name, "unsupported ZIP record flags"),
        ("config_reserved_flags", "reserved_flags", weights_name, "unsupported ZIP record flags"),
        ("encrypted_record", "encrypted", payload_name, "invalid or unsupported zip archive"),
        ("forged_store_size", "size", payload_name, "invalid or unsupported zip archive"),
    ):
        target = directory / (name + ".pt2")
        write_records(target, records)
        data, local, central, payload = record_offsets(target, record)
        if mutation == "crc":
            data[payload] ^= 1
        elif mutation in ("flags", "reserved_flags", "encrypted"):
            flag = {"flags": 0x20, "reserved_flags": 0x4000, "encrypted": 1}[mutation]
            struct.pack_into("<H", data, local + 6, flag)
            struct.pack_into("<H", data, central + 8, flag)
        else:
            struct.pack_into("<I", data, local + 22, 512 * 1024 * 1024)
            struct.pack_into("<I", data, central + 24, 512 * 1024 * 1024)
        target.write_bytes(data)
        check_failure(target, directory / (name + "_converted"), needle, timeout)

    def empty_weight(config):
        meta = config["weight"]["tensor_meta"]
        meta["sizes"] = [{"as_int": 0}]
        meta["strides"] = [{"as_int": 1}]
        meta["storage_offset"] = {"as_int": 0}
    empty = config_record(records, weights_name, empty_weight)
    empty[payload_name] = b""
    rejected("compressed_empty", empty, "unsupported ZIP compression", payload_name)
    target = directory / "empty_crc.pt2"
    write_records(target, empty)
    data, local, central, _ = record_offsets(target, payload_name)
    struct.pack_into("<I", data, local + 14, 1)
    struct.pack_into("<I", data, central + 16, 1)
    target.write_bytes(data)
    check_failure(target, directory / "empty_crc_converted", "CRC mismatch", timeout)

    target = directory / "duplicate_record.pt2"
    write_records(target, records)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
        with zipfile.ZipFile(target, "a") as archive:
            archive.writestr(payload_name, records[payload_name])
    check_failure(target, directory / "duplicate_converted", "invalid or unsupported zip archive", timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnnx-executable", default=os.environ.get("PNNX_EXECUTABLE"))
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-conversion seconds; defaults to PNNX_TEST_TIMEOUT or 300")
    args = parser.parse_args()
    if not pnnx_test_utils.has_exported_program():
        raise RuntimeError("archive regressions require torch.export.save, got " + torch.__version__)
    executable = Path(args.pnnx_executable or pnnx_test_utils.find_pnnx()).resolve()
    if not executable.is_file():
        raise RuntimeError("pnnx executable does not exist: " + str(executable))
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    y = torch.ones_like(x)
    with tempfile.TemporaryDirectory(prefix="pnnx_archive_") as temporary:
        directory = Path(temporary)
        with mock.patch.object(pnnx_test_utils, "find_pnnx", return_value=executable.as_posix()):
            test_call_structures(directory, x, y, args.timeout)
            test_archive_contract(directory, x, args.timeout)
    print("PT2 archive contract regressions passed")


if __name__ == "__main__":
    main()