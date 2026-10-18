# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""Fail the experiment if wheel tags, extension ABI, or Mach-O target disagree."""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


def inspect(wheel_dir):
    arch = os.environ["EXPECTED_ARCH"]
    cpu, sdk = arch.rsplit("_", 1)
    expected_platform = "IOS" if sdk == "iphoneos" else "IOSSIMULATOR"
    wheels = list(Path(wheel_dir).glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one experimental iOS wheel")
    wheel = wheels[0]
    if not wheel.name.endswith("-cp314-cp314-ios_14_0_" + arch + ".whl"):
        raise RuntimeError("Unexpected wheel target: " + wheel.name)
    with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(wheel) as archive:
        modules = [n for n in archive.namelist() if n.endswith(".so")]
        expected = "ncnn/ncnn.cpython-314-" + sdk + ".so"
        if modules != [expected]:
            raise RuntimeError("Unexpected Python extension files: " + repr(modules))
        module = Path(archive.extract(expected, temp))
        actual_cpus = subprocess.check_output(["lipo", "-archs", str(module)], text=True).split()
        if actual_cpus != [cpu]:
            raise RuntimeError("Unexpected Mach-O CPU slices: " + repr(actual_cpus))
        build = subprocess.check_output(["xcrun", "vtool", "-show-build", str(module)], text=True)
        print(build)
        if not re.search(r"\bplatform\s+" + expected_platform + r"\b", build):
            raise RuntimeError("Mach-O was not built for " + expected_platform)
        minimum = re.search(r"\bminos\s+(\d+(?:\.\d+){0,2})\b", build)
        if minimum is None:
            raise RuntimeError("Mach-O deployment target is missing")
        parts = tuple(int(part) for part in minimum.group(1).split("."))
        if (parts + (0, 0, 0))[:3] > (14, 0, 0):
            raise RuntimeError("Mach-O requires newer iOS than the wheel tag: " + minimum.group(1))
        links = subprocess.check_output(["otool", "-L", str(module)], text=True)
        print(links)
        if "/opt/homebrew/" in links or "/usr/local/" in links or "MacOSX.sdk" in links:
            raise RuntimeError("Host macOS library leaked into the iOS extension")
    print("Verified iOS CP314 wheel tag, extension suffix, Mach-O CPU and platform")


if __name__ == "__main__":
    inspect(sys.argv[1])
