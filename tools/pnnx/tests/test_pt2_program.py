#!/usr/bin/env python3

import argparse
import copy
import json
import pathlib
import subprocess
import tempfile


def run_case(tester, root, name, mode, data, valid, expected=()):
    path = root / f"{name}.json"
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    process = subprocess.run(
        [str(tester), mode, str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if valid != (process.returncode == 0):
        raise AssertionError(f"{name}: rc={process.returncode}\n{process.stdout}")
    for text in expected:
        if text not in process.stdout:
            raise AssertionError(f"{name}: missing {text!r}\n{process.stdout}")


def tensor():
    return {
        "dtype": 7,
        "sizes": [{"as_int": 2}, {"as_int": 3}],
        "requires_grad": False,
        "device": {"type": "cpu", "index": None},
        "strides": [{"as_int": 3}, {"as_int": 1}],
        "storage_offset": {"as_int": 0},
        "layout": 7,
    }


def program():
    return {
        "graph_module": {
            "graph": {
                "inputs": [{"as_tensor": {"name": "x"}}],
                "outputs": [{"as_tensor": {"name": "y"}}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.relu.default",
                        "inputs": [
                            {
                                "name": "self",
                                "arg": {"as_tensor": {"name": "x"}},
                                "kind": 1,
                            }
                        ],
                        "outputs": [{"as_tensor": {"name": "y"}}],
                        "metadata": {},
                    }
                ],
                "tensor_values": {"x": tensor(), "y": tensor()},
                "sym_int_values": {},
                "sym_bool_values": {},
                "sym_float_values": {},
                "custom_obj_values": {},
                "is_single_tensor_return": False,
            },
            "signature": {
                "input_specs": [
                    {"user_input": {"arg": {"as_tensor": {"name": "x"}}}}
                ],
                "output_specs": [
                    {"user_output": {"arg": {"as_tensor": {"name": "y"}}}}
                ],
            },
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


def encoded(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tester", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="pnnx-pt2-json-") as temporary_directory:
        root = pathlib.Path(temporary_directory)

        run_case(args.tester, root, "json_valid", "--json", '[null,true,false,-1,2.5e3,"\\u4e2d\\ud83d\\ude00"]', True)
        run_case(args.tester, root, "json_truncated", "--json", '{"x":[1,2}', False, ("json offset",))
        run_case(args.tester, root, "json_duplicate", "--json", '{"x":1,"x":2}', False, ("duplicate object key",))
        run_case(args.tester, root, "json_surrogate", "--json", '"\\ud800"', False, ("missing low surrogate",))
        run_case(args.tester, root, "json_utf8", "--json", b'"\xc0\xaf"', False, ("invalid utf-8",))
        run_case(args.tester, root, "json_deep", "--json", "[" * 130 + "0" + "]" * 130, False, ("nesting is too deep",))

        base = program()
        run_case(args.tester, root, "program_valid", "--program-json", encoded(base), True)

        value = copy.deepcopy(base)
        value["graph_module"]["graph"]["nodes"][0]["inputs"].extend(
            [
                {"name": "optional", "arg": {"as_optional_tensor": {"as_none": True}}},
                {
                    "name": "optionals",
                    "arg": {
                        "as_optional_tensors": [
                            {"as_tensor": {"name": "x"}},
                            {"as_none": True},
                        ]
                    },
                },
            ]
        )
        run_case(args.tester, root, "program_optional_tensors", "--program-json", encoded(value), True)

        cases = []

        value = copy.deepcopy(base)
        value["schema_version"]["major"] = 7
        cases.append(("schema_major", value, "$.schema_version.major"))

        value = copy.deepcopy(base)
        value["schema_version"]["minor"] = 14
        cases.append(("schema_minor", value, "$.schema_version.minor"))

        value = copy.deepcopy(base)
        del value["opset_version"]["aten"]
        cases.append(("missing_opset", value, "missing aten opset"))

        value = copy.deepcopy(base)
        value["graph_module"]["graph"]["inputs"][0] = {"as_graph": {}}
        cases.append(("unknown_argument", value, "unsupported Argument union tag as_graph"))

        value = copy.deepcopy(base)
        del value["graph_module"]["graph"]["tensor_values"]["x"]["dtype"]
        cases.append(("missing_tensor_field", value, ".tensor_values.x.dtype"))

        value = copy.deepcopy(base)
        value["graph_module"]["graph"]["nodes"][0]["inputs"][0]["arg"]["as_tensor"]["name"] = "missing"
        cases.append(("unknown_value", value, "unknown tensor value missing"))

        value = copy.deepcopy(base)
        value["graph_module"]["signature"]["output_specs"][0] = {
            "buffer_mutation": {"arg": {"as_tensor": {"name": "y"}}, "buffer_name": "state"}
        }
        cases.append(("mutation", value, "training or mutation output is unsupported"))

        value = copy.deepcopy(base)
        value["graph_module"]["graph"]["tensor_values"]["x"]["device"]["type"] = "cuda"
        cases.append(("device", value, "only dense strided CPU tensors are supported"))

        value = copy.deepcopy(base)
        value["range_constraints"] = {"s0": {"min_val": 4, "max_val": 2}}
        cases.append(("range", value, "minimum exceeds maximum"))

        for name, value, message in cases:
            run_case(args.tester, root, name, "--program-json", encoded(value), False, ("json offset", message))


if __name__ == "__main__":
    main()
