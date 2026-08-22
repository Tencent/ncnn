#!/usr/bin/env python3
# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import io
import json
import pathlib
import struct
import subprocess
import tempfile
import zipfile


ELEMENT_SIZES = [0, 1, 1, 2, 4, 8, 2, 4, 8, 4, 8, 16, 1, 2]


def read_zip(path_or_bytes):
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with zipfile.ZipFile(source) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


def zip_bytes(records):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, data in records.items():
            archive.writestr(name, data)
    return output.getvalue()


def write_zip(path, records):
    path.write_bytes(zip_bytes(records))


def find_record(records, suffix):
    names = [name for name in records if name.endswith(suffix)]
    if len(names) != 1:
        raise AssertionError(f"expected one {suffix!r} record, got {names}")
    return names[0]


def tensor_meta(dtype, shape, strides=None, offset=0):
    if strides is None:
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.append(stride)
            stride *= size
        strides.reverse()
    return {
        "dtype": dtype,
        "sizes": [{"as_int": value} for value in shape],
        "requires_grad": True,
        "device": {"type": "cpu", "index": None},
        "strides": [{"as_int": value} for value in strides],
        "storage_offset": {"as_int": offset},
        "layout": 7,
    }


def make_raw_archive(entries, byteorder="little"):
    graph_inputs = []
    input_specs = []
    tensor_values = {}
    config = {}
    records = {
        "model/archive_format": b"pt2",
        "model/archive_version": b"0",
        "model/byteorder": byteorder.encode(),
    }
    for name, dtype, shape, payload in entries:
        argument_name = "p_" + name
        meta = tensor_meta(dtype, shape)
        graph_inputs.append({"as_tensor": {"name": argument_name}})
        input_specs.append(
            {"parameter": {"arg": {"name": argument_name}, "parameter_name": name}}
        )
        tensor_values[argument_name] = meta
        config[name] = {
            "path_name": name,
            "is_param": True,
            "use_pickle": False,
            "tensor_meta": meta,
        }
        records["model/data/weights/" + name] = payload

    program = {
        "graph_module": {
            "graph": {
                "inputs": graph_inputs,
                "outputs": [],
                "nodes": [],
                "tensor_values": tensor_values,
                "sym_int_values": {},
                "sym_bool_values": {},
                "sym_float_values": {},
                "custom_obj_values": {},
                "is_single_tensor_return": False,
            },
            "signature": {"input_specs": input_specs, "output_specs": []},
            "module_call_graph": [],
            "metadata": {},
            "treespec_namedtuple_fields": {},
        },
        "opset_version": {"aten": 10},
        "range_constraints": {},
        "schema_version": {"major": 8, "minor": 20},
        "torch_version": "2.12.1",
        "verifiers": [],
    }
    records["model/models/model.json"] = json.dumps(program, separators=(",", ":")).encode()
    records["model/data/weights/model_weights_config.json"] = json.dumps(
        {"config": config}, separators=(",", ":")
    ).encode()
    records["model/data/constants/model_constants_config.json"] = b'{"config":{}}'
    return records


def run(tester, path, case, valid, expected=""):
    process = subprocess.run(
        [str(tester), "--weights-archive", str(path), case],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if valid != (process.returncode == 0) or (expected and expected not in process.stdout):
        raise AssertionError(
            f"{path.name}: valid={valid} expected={expected!r} rc={process.returncode}\n{process.stdout}"
        )


def mutate_raw(source, root, name, mutation, expected, tester):
    records = read_zip(source)
    mutation(records)
    path = root / f"raw_{name}.pt2"
    write_zip(path, records)
    run(tester, path, "state_and_constants", False, expected)


def mutate_legacy(source, root, name, mutation, expected, tester):
    records = read_zip(source)
    state_name = find_record(records, "serialized_state_dict.pt")
    nested = read_zip(records[state_name])
    mutation(nested)
    records[state_name] = zip_bytes(nested)
    path = root / f"legacy_{name}.pt2"
    write_zip(path, records)
    run(tester, path, "state_and_constants", False, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tester", type=pathlib.Path, required=True)
    parser.add_argument("--legacy", type=pathlib.Path, required=True)
    parser.add_argument("--raw", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="pnnx-pt2-weights-") as directory:
        root = pathlib.Path(directory)

        entries = []
        for dtype in range(1, 14):
            entries.append(
                (f"dtype_{dtype}", dtype, [2], bytes(range(ELEMENT_SIZES[dtype] * 2)))
            )
        entries.extend(
            [
                ("scalar", 7, [], bytes(range(4))),
                ("empty", 7, [0], b""),
            ]
        )
        scalar_types = root / "scalar_types.pt2"
        write_zip(scalar_types, make_raw_archive(entries))
        run(args.tester, scalar_types, "scalar_types", True)

        big_endian = root / "big_endian.pt2"
        write_zip(big_endian, make_raw_archive([("value", 7, [1], struct.pack(">f", 1.0))], "big"))
        run(args.tester, big_endian, "big_endian", True)

        oversized_tensor = root / "oversized_tensor.pt2"
        write_zip(oversized_tensor, make_raw_archive([("value", 7, [1 << 62], b"")]))
        run(args.tester, oversized_tensor, "oversized_tensor", False, "symbolic, negative, or oversized tensor view metadata")

        def unsafe_path(records):
            name = find_record(records, "model_weights_config.json")
            config = json.loads(records[name])
            config["config"]["weight"]["path_name"] = "../weight_0"
            records[name] = json.dumps(config).encode()

        mutate_raw(args.raw, root, "unsafe_path", unsafe_path, "unsafe payload record name", args.tester)

        def pickled_payload(records):
            name = find_record(records, "model_weights_config.json")
            config = json.loads(records[name])
            config["config"]["weight"]["use_pickle"] = True
            records[name] = json.dumps(config).encode()

        mutate_raw(args.raw, root, "pickled", pickled_payload, "pickled raw payload is unsupported", args.tester)

        def unknown_config_field(records):
            name = find_record(records, "model_weights_config.json")
            config = json.loads(records[name])
            config["config"]["weight"]["future_required"] = True
            records[name] = json.dumps(config).encode()

        mutate_raw(args.raw, root, "unknown_config_field", unknown_config_field, "future_required", args.tester)

        def truncated_storage(records):
            name = find_record(records, "data/weights/weight_0")
            records[name] = records[name][:-4]

        mutate_raw(args.raw, root, "truncated", truncated_storage, "tensor view is outside its storage", args.tester)

        def missing_payload(records):
            name = find_record(records, "model_weights_config.json")
            config = json.loads(records[name])
            del config["config"]["bias"]
            records[name] = json.dumps(config).encode()

        mutate_raw(args.raw, root, "missing", missing_payload, "missing payload for bias", args.tester)

        def unknown_global(records):
            name = find_record(records, "data.pkl")
            records[name] = records[name].replace(
                b"collections\nOrderedDict\n", b"collections\nOrderedDicx\n", 1
            )

        mutate_legacy(args.legacy, root, "global", unknown_global, "unsupported GLOBAL", args.tester)

        def unsupported_opcode(records):
            name = find_record(records, "data.pkl")
            data = bytearray(records[name])
            data[2] = 0xFF
            records[name] = bytes(data)

        mutate_legacy(args.legacy, root, "opcode", unsupported_opcode, "unsupported pickle opcode", args.tester)

        def empty_string_at_eof(records):
            name = find_record(records, "data.pkl")
            records[name] = b"\x80\x02X\x00\x00\x00\x00"

        mutate_legacy(args.legacy, root, "empty_string", empty_string_at_eof, "pickle has no STOP opcode", args.tester)

        def short_storage(records):
            name = find_record(records, "data/0")
            records[name] = records[name][:-4]

        mutate_legacy(args.legacy, root, "storage", short_storage, "storage byte size does not match", args.tester)


if __name__ == "__main__":
    main()
