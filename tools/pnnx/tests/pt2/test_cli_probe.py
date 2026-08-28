#!/usr/bin/env python3
# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pathlib
import subprocess
import tempfile
import zipfile

import torch


class UnsupportedOperator(torch.nn.Module):
    def forward(self, x):
        return torch.special.airy_ai(x)


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
    args = parser.parse_args()

    args.pnnx = args.pnnx.resolve()
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        damaged = root / "damaged.pt2"
        damaged.write_bytes(b"PK\x03\x04damaged")
        check_failure(args.pnnx, damaged, "model format probe failed")

        unknown = root / "unknown.pt2"
        with zipfile.ZipFile(unknown, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("content.txt", "not a model")
        check_failure(args.pnnx, unknown, "unsupported unknown-zip model")

        unsupported = root / "unsupported.pt2"
        torch.export.save(torch.export.export(UnsupportedOperator(), (torch.randn(4),)), unsupported)
        check_failure(args.pnnx, unsupported, "unsupported operator aten::special_airy_ai")


if __name__ == "__main__":
    main()
