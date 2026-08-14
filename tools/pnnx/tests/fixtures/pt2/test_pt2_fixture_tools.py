#!/usr/bin/env python3
# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from pt2_fixture_tools import (
    build_manifest,
    extract_record,
    inspect_archive,
    normalize_torch_version,
    semantic_sha256_archive,
    verify_manifest,
    write_manifest,
)


def _program_json(minor: int, graph_id: int | None = None) -> bytes:
    program = {
        "schema_version": {"major": 8, "minor": minor},
        "opset_version": {"aten": 1},
        "torch_version": "synthetic",
        "graph_module": {
            "graph": {
                "inputs": [{"name": "x"}],
                "nodes": [
                    {
                        "target": "torch.ops.aten.add.Tensor",
                        "metadata": {"from_node": "graph_id=%s" % graph_id},
                    }
                ],
                "outputs": [{"name": "add"}],
            }
        },
    }
    return json.dumps(program, sort_keys=True).encode("utf-8")


def _write_stored(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    archive.writestr(info, data)


def _write_legacy(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        _write_stored(archive, "version", b"8.7")
        _write_stored(archive, "serialized_exported_program.json", _program_json(7))
        _write_stored(archive, "serialized_state_dict.pt", b"state-dict")
        _write_stored(archive, "serialized_constants.pt", b"constants")
        _write_stored(archive, "serialized_example_inputs.pt", b"inputs")


def _write_archive(path: Path, graph_id: int | None = None) -> None:
    root = "synthetic/"
    with zipfile.ZipFile(path, "w") as archive:
        _write_stored(archive, root + "archive_format", b"pt2")
        _write_stored(archive, root + "archive_version", b"1")
        _write_stored(archive, root + "byteorder", b"little")
        _write_stored(archive, root + ".data/version", b"6")
        _write_stored(archive, root + ".data/serialization_id", str(graph_id).encode())
        _write_stored(archive, root + "models/model.json", _program_json(8, graph_id))
        config = {
            "config": {
                "weight": {
                    "path_name": "weight_0",
                    "is_param": True,
                    "use_pickle": False,
                    "tensor_meta": {},
                }
            }
        }
        _write_stored(
            archive,
            root + "data/weights/model_weights_config.json",
            json.dumps(config).encode("utf-8"),
        )
        _write_stored(archive, root + "data/weights/weight_0", b"\x00\x01\x02\x03")


class Pt2FixtureToolsTest(unittest.TestCase):
    def test_normalize_torch_version(self) -> None:
        self.assertEqual(normalize_torch_version("2.12.1+cpu"), "2.12.1")
        self.assertEqual(normalize_torch_version("2.8.0.dev20250101"), "2.8.0")

    def test_inspect_legacy_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt2"
            _write_legacy(path)
            result = inspect_archive(path)
            self.assertEqual(result["container"], "legacy_exported_program_zip")
            self.assertEqual(result["root_prefix"], "")
            self.assertEqual(result["schema_version"], {"major": 8, "minor": 7})
            self.assertEqual(result["payload_styles"], ["nested_torch_save"])
            self.assertEqual(result["graph"], {"inputs": 1, "nodes": 1, "outputs": 1})

    def test_inspect_rooted_pt2_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.pt2"
            _write_archive(path)
            result = inspect_archive(path)
            self.assertEqual(result["container"], "pt2_archive")
            self.assertEqual(result["root_prefix"], "synthetic")
            self.assertEqual(result["schema_version"], {"major": 8, "minor": 8})
            self.assertEqual(result["payload_styles"], ["raw_tensor_with_payload_config"])

    def test_manifest_hash_and_inspection_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "state_and_constants.pt2"
            _write_legacy(fixture)
            manifest = build_manifest(
                root,
                [fixture],
                {"torch_version_normalized": "2.7.0"},
                {"version": 1},
            )
            manifest_path = root / "manifest.json"
            write_manifest(manifest_path, manifest)
            self.assertEqual(verify_manifest(manifest_path), [])

            with fixture.open("ab") as stream:
                stream.write(b"tamper")
            errors = verify_manifest(manifest_path)
            self.assertTrue(any("size mismatch" in error for error in errors))
            self.assertTrue(any("SHA-256 mismatch" in error for error in errors))

    def test_semantic_hash_ignores_debug_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.pt2"
            second = root / "second.pt2"
            _write_archive(first, graph_id=123)
            _write_archive(second, graph_id=456)
            self.assertNotEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                semantic_sha256_archive(first), semantic_sha256_archive(second)
            )

    def test_extract_logical_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive.pt2"
            output = root / "model.json"
            _write_archive(archive)
            extract_record(archive, "models/model.json", output, force=False)
            self.assertEqual(output.read_bytes(), _program_json(8))


if __name__ == "__main__":
    unittest.main()
