# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# multi-version raw-payload schema coverage: the 8.20 fixture is downgraded to
# 8.17 / 8.15 / 8.14 (see fixtures/pt2_schema/make_schema_fixtures.py). the
# loader is field-presence driven, so each fixture must convert identically:
# schema_version.minor must not gate acceptance, and the pre-8.15 field set
# (no as_nested_tensors/as_int_lists/as_string_to_argument/as_float_lists)
# must load without errors.

import importlib.util
import os
import subprocess
import sys
import tempfile

import torch  # noqa: F401  (torch.export availability gate)


def _find_pnnx():
    # runner cwd is tools/pnnx/build/tests (or tests/ncnn); the binary lives in
    # the build tree's src dir next to it
    for rel in (os.path.join("..", "src", "pnnx.exe"),
                os.path.join("..", "..", "src", "pnnx.exe"),
                os.path.join("..", "src", "pnnx"),
                os.path.join("..", "..", "src", "pnnx")):
        p = os.path.abspath(rel)
        if os.path.isfile(p):
            return p
    return None


def _eager_reference():
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    class M(nn.Module):
        def __init__(self):
            super(M, self).__init__()
            self.c = nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x):
            return self.c(x).relu() + 1

    m = M().eval()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        return m(x).detach().numpy()


def _case(name, ok, detail):
    print("[schema] %-16s %s" % (name, "PASS" if ok else "FAIL"))
    if not ok:
        print(detail)
    return ok


def test():
    pnnx = _find_pnnx()
    if pnnx is None:
        print("[schema] pnnx binary not found")
        return False

    try:
        import torch  # noqa: F401
    except Exception:
        print("[schema] torch not available, skip")
        return True

    fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "pt2_schema")
    ref = _eager_reference()
    results = []

    with tempfile.TemporaryDirectory() as workdir:
        for minor in (20, 17, 15, 14):
            import shutil
            pt2_src = os.path.join(fixtures, "schema_8_%d.pt2" % minor)
            pt2 = os.path.join(workdir, os.path.basename(pt2_src))
            shutil.copy2(pt2_src, pt2)
            base = os.path.join(workdir, "schema_8_%d" % minor)
            # forward-slash form for the CLI: pnnx embeds these paths (incl.
            # the input-derived default py path) literally in the generated py
            # (windows backslashes would be a \\U escape)
            pt2_arg = pt2.replace("\\", "/")
            base_arg = base.replace("\\", "/")
            r = subprocess.run(
                [pnnx, pt2_arg, "inputshape=[1,3,8,8]f32", "pnnxparam=%s.pnnx.param" % base_arg, "pnnxbin=%s.pnnx.bin" % base_arg],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            if r.returncode != 0:
                results.append(_case("8.%d(convert)" % minor, False, r.stdout.decode("utf-8", "replace")[-600:]))
                continue

            pymod = "%s_pnnx.py" % base
            spec = importlib.util.spec_from_file_location("schema_m%d" % minor, pymod)
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, workdir)
            spec.loader.exec_module(mod)
            out = mod.test_inference().detach().numpy()

            import numpy as np
            ok = np.allclose(out, ref, atol=1e-5)
            results.append(_case("8.%d(convert+infer)" % minor, ok, "maxdiff=%f" % np.abs(out - ref).max()))

    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if test() else 1)
