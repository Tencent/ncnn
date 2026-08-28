#!/usr/bin/env python3
# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pathlib
import struct
import subprocess
import tempfile
import zipfile

from test_program import encoded, program


def write_zip(path, records, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression, allowZip64=True) as archive:
        for name, data in records:
            archive.writestr(name, data)


def probe(tester, path, expected):
    process = subprocess.run(
        [str(tester), str(path), expected],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.returncode != 0:
        raise AssertionError(
            f"probe failed for {path.name}: expected={expected} rc={process.returncode}\n{process.stdout}"
        )


def patch_eocd_cd_offset(path):
    data = bytearray(path.read_bytes())
    eocd = data.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise AssertionError("generated ZIP has no EOCD")
    struct.pack_into("<I", data, eocd + 16, len(data) + 4096)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tester", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="pnnx-pt2-zip-") as temporary_directory:
        root = pathlib.Path(temporary_directory)

        plain = root / "model.onnx"
        plain.write_bytes(b"not-a-zip")
        probe(args.tester, plain, "other")

        torchscript = root / "torchscript.pt"
        write_zip(
            torchscript,
            [
                ("module/data.pkl", b"pickle"),
                ("module/constants.pkl", b"pickle"),
                ("module/code/__torch__.py", b"class Module:\n    pass\n"),
                ("module/.data/version", b"3\n"),
            ],
        )
        probe(args.tester, torchscript, "torchscript")

        compressed_torchscript = root / "compressed_torchscript.pt"
        write_zip(
            compressed_torchscript,
            [
                ("module/data.pkl", b"pickle"),
                ("module/constants.pkl", b"pickle"),
                ("module/code/__torch__.py", b"class Module:\n    pass\n"),
                ("module/version", b"3\n"),
            ],
            compression=zipfile.ZIP_DEFLATED,
        )
        probe(args.tester, compressed_torchscript, "torchscript")

        pt2 = root / "archive.pt2"
        write_zip(
            pt2,
            [
                ("model/archive_format", b"pt2"),
                ("model/archive_version", b"0"),
                ("model/models/model.json", b"{}"),
            ],
        )
        probe(args.tester, pt2, "pt2-archive")

        legacy = root / "legacy.pt2"
        write_zip(
            legacy,
            [
                ("serialized_exported_program.json", b"{}"),
                ("version", b"8.2"),
            ],
        )
        probe(args.tester, legacy, "pt2-legacy-exported-program")

        empty_legacy_version = root / "empty_legacy_version.pt2"
        write_zip(empty_legacy_version, [("serialized_exported_program.json", b"{}"), ("version", b"\n")])
        probe(args.tester, empty_legacy_version, "invalid-zip")

        mismatched_legacy_version = root / "mismatched_legacy_version.pt2"
        write_zip(mismatched_legacy_version, [("serialized_exported_program.json", encoded(program()).encode()), ("version", b"8.7")])
        process = subprocess.run(
            [str(args.tester), "--program-archive", str(mismatched_legacy_version)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if process.returncode == 0 or "schema version does not match legacy archive version 8.7" not in process.stdout:
            raise AssertionError(f"legacy schema mismatch was accepted\n{process.stdout}")

        unknown = root / "unknown.zip"
        write_zip(unknown, [("payload", b"data")])
        probe(args.tester, unknown, "unknown-zip")

        wrong_marker = root / "wrong_marker.zip"
        write_zip(wrong_marker, [("archive_format", b"not-pt2"), ("payload", b"data")])
        probe(args.tester, wrong_marker, "unknown-zip")

        missing_version = root / "missing_version.pt2"
        write_zip(missing_version, [("archive_format", b"pt2"), ("models/model.json", b"{}")])
        probe(args.tester, missing_version, "invalid-zip")

        unsupported_version = root / "unsupported_version.pt2"
        write_zip(unsupported_version, [("archive_format", b"pt2"), ("archive_version", b"1"), ("models/model.json", b"{}")])
        probe(args.tester, unsupported_version, "invalid-zip")

        missing_model = root / "missing_model.pt2"
        write_zip(missing_model, [("archive_format", b"pt2"), ("archive_version", b"0")])
        probe(args.tester, missing_model, "invalid-zip")

        traversal = root / "traversal.zip"
        write_zip(traversal, [("../archive_format", b"pt2")])
        probe(args.tester, traversal, "invalid-zip")

        duplicate = root / "duplicate.zip"
        write_zip(duplicate, [("root/record", b"a"), ("root/./record", b"b")])
        probe(args.tester, duplicate, "invalid-zip")

        compressed = root / "compressed.zip"
        write_zip(compressed, [("record", b"data")], compression=zipfile.ZIP_DEFLATED)
        probe(args.tester, compressed, "invalid-zip")

        truncated = root / "truncated.zip"
        write_zip(truncated, [("record", b"data")])
        truncated.write_bytes(truncated.read_bytes()[:-10])
        probe(args.tester, truncated, "invalid-zip")

        invalid_offset = root / "invalid_offset.zip"
        write_zip(invalid_offset, [("record", b"data")])
        patch_eocd_cd_offset(invalid_offset)
        probe(args.tester, invalid_offset, "invalid-zip")

        corrupt_crc = root / "corrupt_crc.pt2"
        write_zip(
            corrupt_crc,
            [("archive_format", b"pt2"), ("archive_version", b"0"), ("models/model.json", b"{}")],
        )
        data = bytearray(corrupt_crc.read_bytes())
        marker = data.find(b"pt2")
        if marker < 0:
            raise AssertionError("generated PT2 marker was not found")
        data[marker] ^= 1
        corrupt_crc.write_bytes(data)
        probe(args.tester, corrupt_crc, "invalid-zip")


if __name__ == "__main__":
    main()
