# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# PT2 -> PNNX -> ONNX white-box channel tests. pnnx itself is built without the
# protobuf/onnx backend here, so it cannot emit the C++-side .pnnx.onnx; the
# generated *_pnnx.py carries an export_onnx() helper that serializes the
# converted Model() via torch.onnx.export instead. These tests pin that the
# channel is sound end to end:
#   * the exported graph is structurally the expected one (white-box node set)
#   * onnx.checker accepts it and the reference evaluator reproduces the exact
#     numbers the pnnx Model() produces (torch seed 0 is used by both sides)
#   * the re-exported program (PT2 -> PNNX -> PT2 -> PNNX) exports to the same
#     numbers too, so the onnx channel tracks the whole roundtrip
#
# onnx comes in via onnxscript (already a test dependency); the reference
# evaluator is pure python and needs no native runtime, so this runs anywhere
# the rest of the pt2 suite runs.

import os
import subprocess
import sys
import tempfile


def _find_pnnx():
    for rel in (os.path.join("..", "src", "pnnx.exe"),
                os.path.join("..", "..", "src", "pnnx.exe"),
                os.path.join("..", "src", "pnnx"),
                os.path.join("..", "..", "src", "pnnx")):
        p = os.path.abspath(rel)
        if os.path.isfile(p):
            return p
    return None


def _export_conv(pt2):
    import torch
    import torch.nn as nn

    torch.manual_seed(7)

    class M(nn.Module):
        def __init__(self):
            super(M, self).__init__()
            self.c = nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x):
            return self.c(x).relu() + 1

    m = M().eval()
    with torch.no_grad():
        ep = torch.export.export(m, (torch.rand(1, 3, 8, 8),))
        torch.export.save(ep, pt2)


def _convert(pnnx, pt2, workdir, base, inputshape="[1,3,8,8]f32"):
    r = subprocess.run(
        [pnnx, pt2, "inputshape=%s" % inputshape, "pnnxparam=%s.pnnx.param" % base, "pnnxbin=%s.pnnx.bin" % base],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    return r.returncode, (r.stdout or b"").decode("utf-8", "replace")


def _emit_onnx(workdir, mod_name):
    # run the generated helper: writes <mod_name>.py.onnx next to the module
    r = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, %r); import %s; %s.export_onnx()"
         % (workdir, mod_name, mod_name)],
        cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
    )
    return r.returncode, (r.stdout or b"").decode("utf-8", "replace")


def _onnx_numeric(workdir, mod_name, onnx_path, input_shape=(1, 3, 8, 8)):
    # run in a subprocess too: importing torch twice in the same process (once
    # for Model(), once for onnx) is fine, but keeping it isolated mirrors the
    # suite layout and avoids polluting the evaluator process
    code = (
        "import os, sys, torch, onnx\n"
        "from onnx.reference import ReferenceEvaluator\n"
        "sys.path.insert(0, %r)\n"
        "import %s as G\n"
        "m = onnx.load(%r)\n"
        "onnx.checker.check_model(m)\n"
        "sess = ReferenceEvaluator(m)\n"
        "torch.manual_seed(0)\n"
        "x = torch.rand(%r)\n"
        "got = torch.tensor(sess.run(None, {'in0': x.numpy()})[0])\n"
        "net = G.Model(); net.float(); net.eval()\n"
        "with torch.no_grad():\n"
        "    want = net(x)\n"
        "err = (got - want).abs().max().item()\n"
        "print('NODES', [n.op_type for n in m.graph.node])\n"
        "print('MAXDIFF', err)\n"
        "sys.exit(0 if err < 1e-5 else 1)\n"
    ) % (workdir, mod_name, onnx_path, input_shape)
    r = subprocess.run([sys.executable, "-c", code], cwd=workdir,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    return r.returncode, (r.stdout or b"").decode("utf-8", "replace")


def _case(name, ok, detail):
    print("[pt2onnx] %-22s %s" % (name, "PASS" if ok else "FAIL"))
    if not ok:
        print(detail)
    return ok


def test():
    pnnx = _find_pnnx()
    if pnnx is None:
        print("[pt2onnx] pnnx binary not found")
        return False

    try:
        import torch  # noqa: F401
    except Exception:
        print("[pt2onnx] torch not available, skip")
        return True

    try:
        import onnx  # noqa: F401
    except Exception:
        print("[pt2onnx] onnx not available, skip")
        return True

    results = []

    with tempfile.TemporaryDirectory() as workdir:
        # ---- static conv: PT2 -> PNNX -> ONNX
        pt2 = os.path.join(workdir, "conv.pt2")
        _export_conv(pt2)
        base = os.path.join(workdir, "conv")
        rc, text = _convert(pnnx, pt2, workdir, base)
        ok1 = rc == 0
        results.append(_case("convert", ok1, text[-600:]))
        if not ok1:
            return False

        # generated helper emits a valid onnx graph
        mod = os.path.basename(base) + "_pnnx"
        rc, text = _emit_onnx(workdir, mod)
        ok2 = rc == 0 and os.path.isfile(os.path.join(workdir, mod + ".py.onnx"))
        results.append(_case("export_onnx", ok2, text[-600:]))
        if not ok2:
            return False

        # white-box numeric + structural check against the pnnx Model
        onnx_path = os.path.join(workdir, mod + ".py.onnx")
        rc, text = _onnx_numeric(workdir, mod, onnx_path)
        ok3 = rc == 0 and "Conv" in text and "Relu" in text and "MAXDIFF" in text
        results.append(_case("onnx_numeric", ok3, text[-600:]))
        if not ok3:
            return False

        # ---- roundtrip: PT2 -> PNNX -> PT2 -> PNNX -> ONNX keeps the numbers
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); import %s; %s.export_exported_program(out_path=%r)"
             % (workdir, mod, mod, os.path.join(workdir, "conv_re.pt2"))],
            cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
        )
        ok4 = r.returncode == 0 and os.path.isfile(os.path.join(workdir, "conv_re.pt2"))
        results.append(_case("reexport", ok4, (r.stdout or b"").decode("utf-8", "replace")[-600:]))
        if not ok4:
            return False

        base2 = os.path.join(workdir, "conv2")
        rc, text = _convert(pnnx, os.path.join(workdir, "conv_re.pt2"), workdir, base2)
        ok5 = rc == 0
        results.append(_case("re_convert", ok5, text[-600:]))
        if not ok5:
            return False

        mod2 = os.path.basename(os.path.join(workdir, "conv_re")) + "_pnnx"
        rc, text = _emit_onnx(workdir, mod2)
        ok6 = rc == 0 and os.path.isfile(os.path.join(workdir, mod2 + ".py.onnx"))
        results.append(_case("re_export_onnx", ok6, text[-600:]))
        if not ok6:
            return False

        rc, text = _onnx_numeric(workdir, mod2, os.path.join(workdir, mod2 + ".py.onnx"))
        ok7 = rc == 0 and "MAXDIFF" in text
        results.append(_case("re_onnx_numeric", ok7, text[-600:]))

    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if test() else 1)
