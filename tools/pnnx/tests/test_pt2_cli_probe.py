#!/usr/bin/env python3
# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pathlib
import shutil
import subprocess
import tempfile
import zipfile


def check_conversion(pnnx, model, expected_container, root):
    case = root / expected_container
    case.mkdir()
    shutil.copyfile(model, case / "model.pt2")
    process = subprocess.run(
        [str(pnnx), "model.pt2"],
        cwd=case,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    expected = f"pt2 container={expected_container}"
    if process.returncode != 0 or expected not in process.stdout or "schema=8." not in process.stdout or "opset=aten:10" not in process.stdout or "producer=torch" not in process.stdout:
        raise AssertionError(
            f"unexpected pnnx PT2 conversion result for {model}: rc={process.returncode}\n{process.stdout}"
        )
    for name in ("model.pnnx.param", "model.pnnx.bin", "model_pnnx.py", "model.ncnn.param", "model.ncnn.bin", "model_ncnn.py"):
        if not (case / name).is_file():
            raise AssertionError(f"missing PT2 conversion output {name}\n{process.stdout}")


def check_failure(pnnx, model, expected):
    process = subprocess.run(
        [str(pnnx), str(model)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.returncode == 0 or expected not in process.stdout:
        raise AssertionError(f"unexpected pnnx failure for {model}: rc={process.returncode}\n{process.stdout}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", type=pathlib.Path, required=True)
    parser.add_argument("--legacy", type=pathlib.Path, required=True)
    parser.add_argument("--archive", type=pathlib.Path, required=True)
    args = parser.parse_args()

    args.pnnx = args.pnnx.resolve()
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        check_conversion(args.pnnx, args.legacy, "legacy-exported-program", root)
        check_conversion(args.pnnx, args.archive, "archive", root)

        damaged = root / "damaged.pt2"
        damaged.write_bytes(b"PK\x03\x04damaged")
        check_failure(args.pnnx, damaged, "model format probe failed")

        unknown = root / "unknown.pt2"
        with zipfile.ZipFile(unknown, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("content.txt", "not a model")
        check_failure(args.pnnx, unknown, "unsupported unknown-zip model")


if __name__ == "__main__":
    main()
