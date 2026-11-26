# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import subprocess
import sys
import tempfile


def test():
    pnnx = Path("../src/pnnx").resolve()
    cases = (
        ((), "", ""),
        (("inputshape=[1,3,224,224]",), "[1,3,224,224]f32", ""),
        (("inputshape=[]",), "[]f32", ""),
        (("inputshape=[1,1,1920]f32,[]i64",), "[1,1,1920]f32,[]i64", ""),
        (("inputshape=[480],[45304],[],",), "[480]f32,[45304]f32,[]f32", ""),
        (("inputshape=[]i64,[1]f16",), "[]i64,[1]f16", ""),
        (("inputshape=[2]i32,[],[3]f64",), "[2]i32,[]f32,[3]f64", ""),
        (("inputshape=[]bool,[]u8,[]c64",), "[]bool,[]u8,[]c64", ""),
        (("inputshape=[2]f16,[4, 5]i32",), "[2]f16,[4,5]i32", ""),
        (("inputshape2=[]i64",), "", "[]i64"),
        (("inputshape=", "inputshape2="), "", ""),
        (("inputshape=[1,-1]f32,[]i64", "inputshape2=[2,-1]f32,[]i64"),
         "[1,-1]f32,[]i64", "[2,-1]f32,[]i64"),
    )

    # The CLI prints the exact parsed shapes before attempting to open the model.
    # Use a missing model so no frontend or sample tensor can affect this test.
    ok = True
    with tempfile.TemporaryDirectory() as tempdir:
        for args, shape1, shape2 in cases:
            result = subprocess.run(
                [str(pnnx), "does-not-exist.pt", *args],
                cwd=tempdir, stderr=subprocess.PIPE, text=True,
            )
            lines = result.stderr.splitlines()
            expected = ("inputshape = " + shape1, "inputshape2 = " + shape2)
            if not result.returncode or not all(line in lines for line in expected):
                print("args=%r expected %r; got:\n%s" % (args, expected, result.stderr),
                      file=sys.stderr)
                ok = False
    return ok


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
