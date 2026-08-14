#!/usr/bin/env python3
# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""Generate deterministic, version-pinned torch.export PT2 fixtures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

from pt2_fixture_tools import (
    FixtureError,
    build_manifest,
    normalize_torch_version,
    semantic_sha256_archive,
    sha256_file,
    version_directory,
    write_manifest,
)


GENERATOR_VERSION = 1


def _load_torch(expected_version: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise FixtureError(
            "PyTorch is required to generate fixtures; install the exact producer wheel first"
        ) from exc

    actual = normalize_torch_version(torch.__version__)
    expected = normalize_torch_version(expected_version)
    if actual != expected:
        raise FixtureError(
            "producer mismatch: --expected-torch=%s but imported torch is %s"
            % (expected, torch.__version__)
        )
    if not hasattr(torch, "export") or not hasattr(torch.export, "save"):
        raise FixtureError("imported PyTorch does not provide torch.export.save")
    return torch


def _cases(torch: Any) -> list[tuple[str, Any, tuple[Any, ...], dict[str, Any]]]:
    class StateAndConstants(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weight = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 16.0
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(torch.arange(4, dtype=torch.float32) / 8.0)
            self.register_buffer(
                "persistent_buffer",
                torch.tensor([0.25, -0.5, 0.75, -1.0], dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "non_persistent_buffer",
                torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float32),
                persistent=False,
            )
            self.tensor_constant = torch.tensor(
                [0.125, 0.25, 0.375, 0.5], dtype=torch.float32
            )

        def forward(self, x: Any) -> Any:
            y = torch.nn.functional.linear(x, self.weight, self.bias)
            return (
                y
                + self.persistent_buffer
                + self.non_persistent_buffer
                + self.tensor_constant
            )

    class StridedTensors(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weight_base = torch.arange(30, dtype=torch.float32).reshape(5, 6)
            self.weight = torch.nn.Parameter(weight_base.transpose(0, 1))

            shared = torch.arange(20, dtype=torch.float32)
            self.register_buffer("offset_view", shared[3:8], persistent=True)
            self.register_buffer("strided_view", shared[3:13:2], persistent=True)

        def forward(self, x: Any) -> Any:
            return torch.matmul(x, self.weight) + self.offset_view + self.strided_view

    class StructuredIo(torch.nn.Module):
        def forward(self, x: Any, y: Any, scale: int = 3) -> Any:
            value = (x + y) * scale
            return {"value": value, "summary": (value.mean(), value.sum(dim=1))}

    state_input = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10.0
    strided_input = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 10.0
    structured_x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    structured_y = torch.arange(8, 16, dtype=torch.float32).reshape(2, 4)

    return [
        ("state_and_constants", StateAndConstants(), (state_input,), {}),
        ("strided_tensors", StridedTensors(), (strided_input,), {}),
        (
            "structured_io",
            StructuredIo(),
            (structured_x, structured_y),
            {"scale": 3},
        ),
    ]


def _export_case(
    torch: Any,
    model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    path: Path,
) -> None:
    model.eval()
    exported_program = torch.export.export(model, args, kwargs=kwargs, strict=True)
    # PyTorch derives the archive root prefix from the output basename. Export
    # to the final basename in a temporary directory so an atomic move does not
    # accidentally leave a misleading ".tmp" prefix inside the fixture.
    with tempfile.TemporaryDirectory(prefix=".pnnx-pt2-export-", dir=path.parent) as directory:
        temporary_path = Path(directory) / path.name
        torch.export.save(exported_program, temporary_path)
        os.replace(temporary_path, path)


def _assert_same_output(torch: Any, expected: Any, actual: Any, path: str = "output") -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise FixtureError("%s type mismatch: expected Tensor, got %s" % (path, type(actual)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or list(actual.keys()) != list(expected.keys()):
            raise FixtureError("%s dictionary structure mismatch" % path)
        for key in expected:
            _assert_same_output(torch, expected[key], actual[key], path + "." + str(key))
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise FixtureError("%s sequence structure mismatch" % path)
        for index, (expected_child, actual_child) in enumerate(zip(expected, actual)):
            _assert_same_output(
                torch, expected_child, actual_child, "%s[%d]" % (path, index)
            )
        return
    if expected != actual:
        raise FixtureError("%s value mismatch: %r != %r" % (path, expected, actual))


def _check_roundtrip(
    torch: Any,
    model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    path: Path,
) -> None:
    with torch.no_grad():
        expected = model(*args, **kwargs)
        loaded = torch.export.load(path)
        actual = loaded.module()(*args, **kwargs)
    _assert_same_output(torch, expected, actual)


def _check_determinism(
    torch: Any,
    model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    first_path: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="pnnx-pt2-determinism-") as directory:
        second_path = Path(directory) / first_path.name
        _export_case(torch, model, args, kwargs, second_path)
        if semantic_sha256_archive(first_path) != semantic_sha256_archive(second_path):
            raise FixtureError(
                "fixture is not semantically deterministic across two exports: %s"
                % first_path.name
            )
        if sha256_file(first_path) != sha256_file(second_path):
            print(
                "note: %s differs only in excluded non-semantic archive metadata"
                % first_path.name
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-torch", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory; defaults to data/torch_X_Y_Z beside this script",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-determinism", action="store_true")
    args = parser.parse_args()

    try:
        torch = _load_torch(args.expected_torch)
        normalized_version = normalize_torch_version(torch.__version__)
        script_directory = Path(__file__).resolve().parent
        output_directory = args.output or (
            script_directory / "data" / version_directory(normalized_version)
        )
        output_directory = output_directory.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        cases = _cases(torch)
        target_paths = [output_directory / (name + ".pt2") for name, _, _, _ in cases]
        target_paths.append(output_directory / "manifest.json")
        existing = [path for path in target_paths if path.exists()]
        if existing and not args.force:
            raise FixtureError(
                "output already exists; pass --force to replace the generated files: %s"
                % ", ".join(str(path) for path in existing)
            )

        torch.manual_seed(0)
        torch.set_num_threads(1)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass

        fixture_paths: list[Path] = []
        for name, model, model_args, model_kwargs in cases:
            fixture_path = output_directory / (name + ".pt2")
            _export_case(torch, model, model_args, model_kwargs, fixture_path)
            _check_roundtrip(torch, model, model_args, model_kwargs, fixture_path)
            if args.check_determinism:
                _check_determinism(
                    torch, model, model_args, model_kwargs, fixture_path
                )
            fixture_paths.append(fixture_path)
            print("generated %s" % fixture_path)

        producer = {
            "torch_version": torch.__version__,
            "torch_version_normalized": normalized_version,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        }
        generator = {
            "version": GENERATOR_VERSION,
            "script": Path(__file__).name,
            "script_sha256": sha256_file(Path(__file__)),
            "strict_export": True,
            "roundtrip_checked": True,
            "semantic_determinism_checked": args.check_determinism,
        }
        manifest = build_manifest(output_directory, fixture_paths, producer, generator)
        manifest_path = output_directory / "manifest.json"
        write_manifest(manifest_path, manifest)
        print("wrote %s" % manifest_path)
        return 0
    except (FixtureError, OSError, RuntimeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
