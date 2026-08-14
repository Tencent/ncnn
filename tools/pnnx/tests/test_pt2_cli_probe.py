#!/usr/bin/env python3

import argparse
import pathlib
import subprocess


def check_probe(pnnx, model, expected_format):
    process = subprocess.run(
        [str(pnnx), str(model)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    expected = f"recognized {expected_format} model"
    if process.returncode == 0 or expected not in process.stdout or "loader is not enabled yet" not in process.stdout:
        raise AssertionError(
            f"unexpected pnnx PT2 probe result for {model}: rc={process.returncode}\n{process.stdout}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnnx", type=pathlib.Path, required=True)
    parser.add_argument("--legacy", type=pathlib.Path, required=True)
    parser.add_argument("--archive", type=pathlib.Path, required=True)
    args = parser.parse_args()

    check_probe(args.pnnx, args.legacy, "pt2-legacy-exported-program")
    check_probe(args.pnnx, args.archive, "pt2-archive")


if __name__ == "__main__":
    main()
