#!/usr/bin/env python3
# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect and verify version-pinned torch.export .pt2 fixtures.

This module deliberately depends only on the Python standard library so that
archive/container checks can run without importing the producer's PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import zipfile


MANIFEST_VERSION = 1
LEGACY_JSON_RECORD = "serialized_exported_program.json"
LEGACY_REQUIRED_RECORDS = {
    "version",
    LEGACY_JSON_RECORD,
    "serialized_state_dict.pt",
    "serialized_constants.pt",
    "serialized_example_inputs.pt",
}


class FixtureError(RuntimeError):
    pass


def normalize_torch_version(version: str) -> str:
    """Return X.Y.Z and intentionally discard wheel/build suffixes."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise FixtureError("invalid PyTorch version %r; expected X.Y.Z" % version)
    return ".".join(match.groups())


def version_directory(version: str) -> str:
    return "torch_" + normalize_torch_version(version).replace(".", "_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _strip_nonsemantic_json_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_nonsemantic_json_metadata(child)
            for key, child in value.items()
            if key != "metadata"
        }
    if isinstance(value, list):
        return [_strip_nonsemantic_json_metadata(child) for child in value]
    return value


def _normalized_member_name(name: str) -> str:
    if "\\" in name:
        raise FixtureError("ZIP member uses a backslash: %r" % name)
    if name.startswith("/"):
        raise FixtureError("ZIP member is absolute: %r" % name)

    parts = []
    for part in name.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise FixtureError("ZIP member contains '..': %r" % name)
        parts.append(part)
    return "/".join(parts)


def _classify_record_map(record_map: dict[str, str], archive: zipfile.ZipFile) -> str | None:
    names = set(record_map)
    if "archive_format" in names:
        value = archive.read(record_map["archive_format"])
        if value == b"pt2" and any(
            name.startswith("models/") and name.endswith(".json") for name in names
        ):
            return "pt2_archive"

    if LEGACY_REQUIRED_RECORDS.issubset(names):
        return "legacy_exported_program_zip"

    return None


def _logical_record_map(archive: zipfile.ZipFile) -> tuple[str, dict[str, str], str]:
    physical_names = [
        _normalized_member_name(info.filename)
        for info in archive.infolist()
        if not info.is_dir()
    ]
    if not physical_names:
        raise FixtureError("archive contains no file records")
    if len(physical_names) != len(set(physical_names)):
        raise FixtureError("archive contains duplicate normalized record names")

    raw_map = {name: name for name in physical_names}
    kind = _classify_record_map(raw_map, archive)
    if kind is not None:
        return kind, raw_map, ""

    first_parts = [name.split("/", 1) for name in physical_names]
    if all(len(parts) == 2 for parts in first_parts):
        prefixes = {parts[0] for parts in first_parts}
        if len(prefixes) == 1:
            root_prefix = next(iter(prefixes))
            stripped_map: dict[str, str] = {}
            for physical_name, parts in zip(physical_names, first_parts):
                logical_name = parts[1]
                if logical_name in stripped_map:
                    raise FixtureError(
                        "archive root stripping creates duplicate record %r" % logical_name
                    )
                stripped_map[logical_name] = physical_name
            kind = _classify_record_map(stripped_map, archive)
            if kind is not None:
                return kind, stripped_map, root_prefix

    key_records = sorted(
        name
        for name in physical_names
        if name.endswith(".json") or name.endswith("/version") or name == "version"
    )
    raise FixtureError(
        "unrecognized PT2 archive layout; key records: %s" % ", ".join(key_records[:20])
    )


def semantic_sha256_archive(path: Path | str) -> str:
    """Hash logical PT2 content while excluding documented non-semantic metadata."""
    path = Path(path)
    digest = hashlib.sha256()
    try:
        with zipfile.ZipFile(path, "r") as archive:
            _, record_map, _ = _logical_record_map(archive)
            for logical_name in sorted(record_map):
                if logical_name == ".data/serialization_id":
                    continue
                data = archive.read(record_map[logical_name])
                if logical_name.endswith(".json"):
                    try:
                        value = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise FixtureError(
                            "invalid JSON record %r while computing semantic hash"
                            % logical_name
                        ) from exc
                    value = _strip_nonsemantic_json_metadata(value)
                    data = json.dumps(
                        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                    ).encode("utf-8")
                encoded_name = logical_name.encode("utf-8")
                digest.update(len(encoded_name).to_bytes(8, "little"))
                digest.update(encoded_name)
                digest.update(len(data).to_bytes(8, "little"))
                digest.update(data)
    except zipfile.BadZipFile as exc:
        raise FixtureError("invalid ZIP archive %s: %s" % (path, exc)) from exc
    return digest.hexdigest()


def _read_text(
    archive: zipfile.ZipFile, record_map: dict[str, str], logical_name: str
) -> str:
    try:
        data = archive.read(record_map[logical_name])
    except KeyError as exc:
        raise FixtureError("missing archive record %r" % logical_name) from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FixtureError("archive record %r is not UTF-8" % logical_name) from exc


def _version_object(value: Any) -> dict[str, Any] | None:
    major: int | None = None
    minor: int | None = None

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        major = value
        minor = 0
    elif isinstance(value, str):
        match = re.match(r"^\s*(\d+)(?:\.(\d+))?", value)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2) or 0)
    elif isinstance(value, (list, tuple)) and value:
        if isinstance(value[0], int) and not isinstance(value[0], bool):
            major = value[0]
            if len(value) > 1 and isinstance(value[1], int) and not isinstance(value[1], bool):
                minor = value[1]
            else:
                minor = 0
    elif isinstance(value, dict):
        for major_key in ("major", "major_version"):
            candidate = value.get(major_key)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                major = candidate
                break
        for minor_key in ("minor", "minor_version"):
            candidate = value.get(minor_key)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                minor = candidate
                break
        if major is not None and minor is None:
            minor = 0

    if major is None or minor is None:
        return None
    return {"major": major, "minor": minor}


def _graph_summary(program: dict[str, Any]) -> dict[str, int | None]:
    graph_module = program.get("graph_module")
    graph = graph_module.get("graph") if isinstance(graph_module, dict) else None
    if not isinstance(graph, dict):
        return {"inputs": None, "nodes": None, "outputs": None}

    def count(name: str) -> int | None:
        value = graph.get(name)
        return len(value) if isinstance(value, list) else None

    return {"inputs": count("inputs"), "nodes": count("nodes"), "outputs": count("outputs")}


def _find_use_pickle(value: Any, result: set[bool]) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("use_pickle"), bool):
            result.add(value["use_pickle"])
        for child in value.values():
            _find_use_pickle(child, result)
    elif isinstance(value, list):
        for child in value:
            _find_use_pickle(child, result)


def _has_payload_path(value: Any) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("path_name"), str):
            return True
        return any(_has_payload_path(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_payload_path(child) for child in value)
    return False


def _payload_styles(
    archive: zipfile.ZipFile, record_map: dict[str, str], kind: str
) -> list[str]:
    if kind == "legacy_exported_program_zip":
        return ["nested_torch_save"]

    config_names = sorted(
        name
        for name in record_map
        if name.endswith("_config.json")
        and ("weight" in name or "constant" in name)
    )
    styles: set[str] = set()
    has_config_payload = False
    for config_name in config_names:
        try:
            config = json.loads(_read_text(archive, record_map, config_name))
        except json.JSONDecodeError as exc:
            raise FixtureError("invalid payload config %r: %s" % (config_name, exc)) from exc
        use_pickle: set[bool] = set()
        has_config_payload = has_config_payload or _has_payload_path(config)
        _find_use_pickle(config, use_pickle)
        if False in use_pickle:
            styles.add("raw_tensor_with_payload_config")
        if True in use_pickle:
            styles.add("nested_torch_save_with_payload_config")

    if not styles and any(
        name.endswith(".pt") and ("weights/" in name or "constants/" in name)
        for name in record_map
    ):
        styles.add("nested_torch_save")
    if not styles and config_names and not has_config_payload:
        styles.add("none")
    elif not styles:
        styles.add("unknown")
    return sorted(styles)


def inspect_archive(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FixtureError("fixture does not exist: %s" % path)

    try:
        with zipfile.ZipFile(path, "r") as archive:
            bad_record = archive.testzip()
            if bad_record is not None:
                raise FixtureError("CRC check failed for %r" % bad_record)

            kind, record_map, root_prefix = _logical_record_map(archive)
            model_records = sorted(
                name
                for name in record_map
                if (kind == "pt2_archive" and name.startswith("models/") and name.endswith(".json"))
                or (kind == "legacy_exported_program_zip" and name == LEGACY_JSON_RECORD)
            )
            if len(model_records) != 1:
                raise FixtureError(
                    "P0 fixtures require exactly one model JSON, found %d" % len(model_records)
                )

            try:
                program = json.loads(_read_text(archive, record_map, model_records[0]))
            except json.JSONDecodeError as exc:
                raise FixtureError("invalid model JSON: %s" % exc) from exc
            if not isinstance(program, dict):
                raise FixtureError("model JSON root must be an object")

            schema_version = _version_object(program.get("schema_version"))
            if schema_version is None and kind == "legacy_exported_program_zip":
                schema_version = _version_object(_read_text(archive, record_map, "version"))
            if schema_version is None:
                raise FixtureError("cannot determine Export schema version")

            opset = program.get("opset_version", {})
            if not isinstance(opset, dict):
                raise FixtureError("opset_version must be an object")

            torch_version = program.get("torch_version")
            if torch_version is not None and not isinstance(torch_version, str):
                torch_version = str(torch_version)

            record_details = []
            for logical_name in sorted(record_map):
                info = archive.getinfo(record_map[logical_name])
                record_details.append(
                    {
                        "name": logical_name,
                        "size": info.file_size,
                        "compressed_size": info.compress_size,
                        "compression": info.compress_type,
                        "flag_bits": info.flag_bits,
                        "crc32": "%08x" % info.CRC,
                    }
                )

            archive_version = None
            if "archive_version" in record_map:
                archive_version = _read_text(archive, record_map, "archive_version").strip()

            return {
                "container": kind,
                "root_prefix": root_prefix,
                "archive_version": archive_version,
                "schema_version": schema_version,
                "opset_version": opset,
                "producer_torch_version_in_model": torch_version,
                "model_record": model_records[0],
                "graph": _graph_summary(program),
                "payload_styles": _payload_styles(archive, record_map, kind),
                "records": record_details,
            }
    except zipfile.BadZipFile as exc:
        raise FixtureError("invalid ZIP archive %s: %s" % (path, exc)) from exc


def build_manifest(
    output_directory: Path,
    fixture_paths: list[Path],
    producer: dict[str, Any],
    generator: dict[str, Any],
) -> dict[str, Any]:
    fixtures = []
    for path in sorted(fixture_paths):
        inspection = inspect_archive(path)
        fixtures.append(
            {
                "case": path.stem,
                "file": os.path.relpath(path, output_directory),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "semantic_sha256": semantic_sha256_archive(path),
                "inspection": inspection,
            }
        )
    return {
        "manifest_version": MANIFEST_VERSION,
        "producer": producer,
        "generator": generator,
        "fixtures": fixtures,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, path)


def verify_manifest(path: Path | str) -> list[str]:
    path = Path(path)
    errors: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ["cannot read manifest %s: %s" % (path, exc)]

    if manifest.get("manifest_version") != MANIFEST_VERSION:
        errors.append("unsupported manifest_version in %s" % path)
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        errors.append("manifest has no fixtures: %s" % path)
        return errors

    seen_cases: set[str] = set()
    for expected in fixtures:
        if not isinstance(expected, dict):
            errors.append("manifest fixture entry is not an object")
            continue
        case = expected.get("case")
        relative_file = expected.get("file")
        if not isinstance(case, str) or not isinstance(relative_file, str):
            errors.append("manifest fixture lacks string case/file")
            continue
        if case in seen_cases:
            errors.append("duplicate fixture case %r" % case)
        seen_cases.add(case)

        fixture_path = path.parent / relative_file
        if not fixture_path.is_file():
            errors.append("missing fixture %s" % fixture_path)
            continue
        actual_size = fixture_path.stat().st_size
        actual_sha256 = sha256_file(fixture_path)
        actual_semantic_sha256 = semantic_sha256_archive(fixture_path)
        if expected.get("size") != actual_size:
            errors.append("size mismatch for %s" % fixture_path)
        if expected.get("sha256") != actual_sha256:
            errors.append("SHA-256 mismatch for %s" % fixture_path)
        if expected.get("semantic_sha256") != actual_semantic_sha256:
            errors.append("semantic SHA-256 mismatch for %s" % fixture_path)

        try:
            actual_inspection = inspect_archive(fixture_path)
        except FixtureError as exc:
            errors.append(str(exc))
            continue
        if expected.get("inspection") != actual_inspection:
            errors.append("archive inspection mismatch for %s" % fixture_path)
    return errors


def verify_matrix(root: Path | str) -> list[str]:
    root = Path(root)
    required_path = root / "required_producers.json"
    try:
        required = json.loads(required_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ["cannot read %s: %s" % (required_path, exc)]

    errors: list[str] = []
    required_cases = set(required.get("required_cases", []))
    for producer_requirement in required.get("producers", []):
        version = producer_requirement["torch_version"]
        fixture_directory = root / "data" / version_directory(version)
        manifest_path = fixture_directory / "manifest.json"
        if not manifest_path.is_file():
            errors.append("missing producer manifest %s" % manifest_path)
            continue
        errors.extend(verify_manifest(manifest_path))
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        actual_version = manifest.get("producer", {}).get("torch_version_normalized")
        if actual_version != normalize_torch_version(version):
            errors.append(
                "producer version mismatch in %s: expected %s, got %s"
                % (manifest_path, version, actual_version)
            )
        fixtures = manifest.get("fixtures", [])
        actual_cases = {entry.get("case") for entry in fixtures if isinstance(entry, dict)}
        missing_cases = required_cases - actual_cases
        if missing_cases:
            errors.append(
                "missing cases for torch %s: %s" % (version, ", ".join(sorted(missing_cases)))
            )

        for fixture in fixtures:
            if not isinstance(fixture, dict):
                continue
            inspection = fixture.get("inspection", {})
            if inspection.get("container") != producer_requirement["expected_container"]:
                errors.append(
                    "container mismatch for torch %s case %s" % (version, fixture.get("case"))
                )
            schema_version = inspection.get("schema_version", {})
            if schema_version.get("major") != producer_requirement["expected_schema_major"]:
                errors.append(
                    "schema-major mismatch for torch %s case %s"
                    % (version, fixture.get("case"))
                )
    return errors


def extract_record(
    archive_path: Path, logical_name: str, output_path: Path, force: bool
) -> None:
    if output_path.exists() and not force:
        raise FixtureError("output already exists; pass --force to replace it: %s" % output_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        _, record_map, _ = _logical_record_map(archive)
        if logical_name not in record_map:
            raise FixtureError("archive has no logical record %r" % logical_name)
        data = archive.read(record_map[logical_name])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_bytes(data)
    os.replace(temporary_path, output_path)


def _print_errors(errors: list[str]) -> int:
    if not errors:
        print("PT2 fixture verification passed")
        return 0
    for error in errors:
        print("error: " + error, file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect one PT2 archive")
    inspect_parser.add_argument("archive", type=Path)

    verify_parser = subparsers.add_parser("verify", help="verify one generated manifest")
    verify_parser.add_argument("manifest", type=Path)

    matrix_parser = subparsers.add_parser(
        "verify-matrix", help="verify all required producer fixture sets"
    )
    matrix_parser.add_argument(
        "root", type=Path, nargs="?", default=Path(__file__).resolve().parent
    )

    extract_parser = subparsers.add_parser(
        "extract", help="extract a normalized logical record"
    )
    extract_parser.add_argument("archive", type=Path)
    extract_parser.add_argument("record")
    extract_parser.add_argument("output", type=Path)
    extract_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_archive(args.archive), indent=2, sort_keys=True))
            return 0
        if args.command == "verify":
            return _print_errors(verify_manifest(args.manifest))
        if args.command == "verify-matrix":
            return _print_errors(verify_matrix(args.root))
        if args.command == "extract":
            extract_record(args.archive, args.record, args.output, args.force)
            return 0
    except (FixtureError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
