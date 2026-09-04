# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import struct
import subprocess
import time
import zipfile

import torch
import torch.nn as nn

from pnnx_test_utils import has_torch_export


class Model(nn.Module):
    def forward(self, x):
        return x + 1


class DictOutputModel(nn.Module):
    def forward(self, x):
        return {"output": x + 1}


class DictInputModel(nn.Module):
    def forward(self, values):
        return values["input"] + 1


class BufferMutationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("value", torch.zeros(1))

    def forward(self, x):
        self.value.add_(1)
        return x + self.value


class UserInputMutationModel(nn.Module):
    def forward(self, x):
        x.add_(1)
        return x * 2


def json_member(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    return next(name for name in names if name.endswith("serialized_exported_program.json") or name.endswith("models/model.json"))


def rewrite_archive(source, destination, transform=None, compression=zipfile.ZIP_STORED):
    target_json = json_member(source)
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(destination, "w", compression=compression, allowZip64=True) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == target_json and transform is not None:
                data = transform(data)
            dst.writestr(info.filename, data)


def run_pnnx(path):
    pnnx = os.path.join("..", "src", "pnnx")
    environment = os.environ.copy()
    torch_library_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    environment["PATH"] = torch_library_path + os.pathsep + environment.get("PATH", "")
    for _ in range(3):
        result = subprocess.run([pnnx, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
        # A few Windows hosts sporadically fail to start LibTorch before main().
        if result.returncode not in (-1073741515, 3221225781):
            return result
        time.sleep(2)
    return result


def expect_failure(path, diagnostic):
    result = run_pnnx(path)
    output = result.stdout + result.stderr
    if result.returncode == 0 or diagnostic not in output:
        print("expected failure containing:", diagnostic)
        print("return code:", result.returncode)
        print(output)
        return False
    return True


def test():
    if not has_torch_export():
        return True

    x = torch.rand(2, 3)
    valid = "test_exported_program_invalid_valid.pt2"
    torch.export.save(torch.export.export(Model().eval(), (x,)), valid)

    compressed = "test_exported_program_invalid_compressed.pt2"
    rewrite_archive(valid, compressed, compression=zipfile.ZIP_DEFLATED)
    compressed_renamed = "test_exported_program_invalid_compressed.pt"
    rewrite_archive(valid, compressed_renamed, compression=zipfile.ZIP_DEFLATED)

    malformed_json = "test_exported_program_invalid_json.pt2"
    rewrite_archive(valid, malformed_json, lambda _: b"{")

    duplicate_key = "test_exported_program_invalid_duplicate_key.pt2"
    def duplicate_schema(data):
        text = data.decode("utf-8")
        marker = '"schema_version"'
        position = text.find(marker)
        colon = text.find(":", position)
        decoder = json.JSONDecoder()
        value, end = decoder.raw_decode(text[colon + 1:].lstrip())
        value_start = colon + 1 + len(text[colon + 1:]) - len(text[colon + 1:].lstrip())
        value_end = value_start + end
        return (text[:value_end] + ',"schema_version":' + json.dumps(value) + text[value_end:]).encode("utf-8")
    rewrite_archive(valid, duplicate_key, duplicate_schema)

    unsupported_schema = "test_exported_program_invalid_schema.pt2"
    def replace_schema(data):
        document = json.loads(data)
        if isinstance(document["schema_version"], dict):
            document["schema_version"]["major"] = 99
        else:
            document["schema_version"] = 99
        return json.dumps(document, separators=(",", ":")).encode("utf-8")
    rewrite_archive(valid, unsupported_schema, replace_schema)

    unsupported_output = "test_exported_program_invalid_output_kind.pt2"
    def replace_output_kind(data):
        document = json.loads(data)
        output_spec = document["graph_module"]["signature"]["output_specs"][0]
        output_spec["unsupported_output"] = output_spec.pop("user_output")
        return json.dumps(document, separators=(",", ":")).encode("utf-8")
    rewrite_archive(valid, unsupported_output, replace_output_kind)

    corrupt_crc = "test_exported_program_invalid_crc.pt2"
    rewrite_archive(valid, corrupt_crc)
    member = json_member(corrupt_crc)
    with zipfile.ZipFile(corrupt_crc) as archive:
        info = archive.getinfo(member)
        with open(corrupt_crc, "r+b") as stream:
            stream.seek(info.header_offset + 26)
            name_length, extra_length = struct.unpack("<HH", stream.read(4))
            data_offset = info.header_offset + 30 + name_length + extra_length
            stream.seek(data_offset)
            first = stream.read(1)
            stream.seek(data_offset)
            stream.write(bytes([first[0] ^ 1]))

    dict_output = "test_exported_program_invalid_dict_output.pt2"
    torch.export.save(torch.export.export(DictOutputModel().eval(), (x,)), dict_output)

    dict_input = "test_exported_program_invalid_dict_input.pt2"
    torch.export.save(torch.export.export(DictInputModel().eval(), ({"input": x},)), dict_input)

    buffer_mutation = "test_exported_program_invalid_buffer_mutation.pt2"
    torch.export.save(torch.export.export(BufferMutationModel().eval(), (x,)), buffer_mutation)

    user_input_mutation = "test_exported_program_invalid_user_input_mutation.pt2"
    torch.export.save(torch.export.export(UserInputMutationModel().eval(), (x,)), user_input_mutation)

    return all((
        expect_failure(compressed, "compressed zip entry is not supported"),
        expect_failure(compressed_renamed, "compressed zip entry is not supported"),
        expect_failure(malformed_json, "parse exported program json failed"),
        expect_failure(duplicate_key, "duplicate object key"),
        expect_failure(unsupported_schema, "unsupported pt2 exported program schema"),
        expect_failure(unsupported_output, "unsupported pt2 graph output kind"),
        expect_failure(corrupt_crc, "zip crc mismatch"),
        expect_failure(dict_output, "unsupported or invalid pt2 output pytree"),
        expect_failure(dict_input, "unsupported or invalid pt2 input pytree"),
        expect_failure(buffer_mutation, "unsupported pt2 graph output kind"),
        expect_failure(user_input_mutation, "user input mutation is not supported"),
    ))


if __name__ == "__main__":
    raise SystemExit(0 if test() else 1)
