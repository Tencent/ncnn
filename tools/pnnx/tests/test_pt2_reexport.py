# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# PT2 re-export roundtrip tests:
#  1) static model: the generated *_pnnx.py export_exported_program() helper
#     must re-export the converted model to a fresh .pt2 that converts again to
#     identical inference results (PT2 -> PNNX -> PT2 -> PNNX roundtrip)
#  2) dynamic model: the loader records the symbolic dims (Symbol('sNN') +
#     range_constraints), the helper restores them as torch.export.Dim, and the
#     re-exported program keeps its range constraints and runs correctly at
#     other shapes inside the range.

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import zipfile


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


def _export_dynamic_relu(pt2):
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def forward(self, x):
            return x.relu()

    m = M().eval()
    with torch.no_grad():
        ep = torch.export.export(
            m, (torch.rand(4, 3, 8, 8),),
            dynamic_shapes=(({0: torch.export.Dim("batch", min=2, max=16), 3: torch.export.Dim("h", min=4, max=128)},)),
        )
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


def _import_and_run(workdir, pymod_path, fn="test_inference"):
    spec = importlib.util.spec_from_file_location("gen_" + fn, pymod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, workdir)
    spec.loader.exec_module(mod)
    fn_obj = getattr(mod, fn)
    return fn_obj()


def _case(name, ok, detail):
    print("[reexport] %-20s %s" % (name, "PASS" if ok else "FAIL"))
    if not ok:
        print(detail)
    return ok


def test():
    pnnx = _find_pnnx()
    if pnnx is None:
        print("[reexport] pnnx binary not found")
        return False

    try:
        import torch  # noqa: F401
    except Exception:
        print("[reexport] torch not available, skip")
        return True

    results = []

    with tempfile.TemporaryDirectory() as workdir:
        # ---- case 1: static roundtrip
        pt2_0 = os.path.join(workdir, "conv.pt2")
        _export_conv(pt2_0)
        base1 = os.path.join(workdir, "conv")
        rc, text = _convert(pnnx, pt2_0, workdir, base1)
        ok1 = rc == 0
        results.append(_case("static(convert1)", ok1, text[-600:]))
        if not ok1:
            return False

        out1 = _import_and_run(workdir, base1 + "_pnnx.py")

        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); import %s; %s.export_exported_program(out_path=%r)"
             % (workdir, os.path.basename(base1) + "_pnnx", os.path.basename(base1) + "_pnnx",
                os.path.join(workdir, "conv_re.pt2"))],
            cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
        )
        rc, text = r.returncode, (r.stdout or b"").decode("utf-8", "replace")
        ok2 = rc == 0 and os.path.isfile(os.path.join(workdir, "conv_re.pt2"))
        results.append(_case("static(reexport)", ok2, text[-600:]))
        if not ok2:
            return False

        base2 = os.path.join(workdir, "conv2")
        rc, text = _convert(pnnx, os.path.join(workdir, "conv_re.pt2"), workdir, base2)
        # the second generation's python script is named after the re-exported
        # archive (conv_re_pnnx.py), not the pnnxparam/pnnxbin base
        out2 = _import_and_run(workdir, os.path.join(workdir, "conv_re_pnnx.py"))

        import numpy as np
        ok3 = rc == 0 and np.allclose(out1.detach().numpy(), out2.detach().numpy(), atol=1e-5)
        results.append(_case("static(roundtrip)", ok3, "rc=%r\n%s" % (rc, text[-600:])))

        # ---- case 2: dynamic re-export keeps the sym constraints
        pt2_d = os.path.join(workdir, "dyn.pt2")
        _export_dynamic_relu(pt2_d)
        based = os.path.join(workdir, "dyn")
        rc, text = _convert(pnnx, pt2_d, workdir, based, inputshape="[4,3,8,8]f32")
        ok4 = rc == 0
        results.append(_case("dynamic(convert)", ok4, text[-600:]))
        if not ok4:
            return False

        red_path = os.path.join(workdir, "dyn_re.pt2")
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); import %s; %s.export_exported_program(out_path=%r)"
             % (workdir, os.path.basename(based) + "_pnnx", os.path.basename(based) + "_pnnx", red_path)],
            cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
        )
        ok5 = r.returncode == 0 and os.path.isfile(red_path)
        results.append(_case("dynamic(reexport)", ok5, (r.stdout or b"").decode("utf-8", "replace")[-600:]))
        if not ok5:
            return False

        # the re-exported archive must keep its range constraints
        z = zipfile.ZipFile(red_path)
        mj = json.loads(z.read([n for n in z.namelist() if n.endswith("models/model.json")][0]))
        z.close()
        rc_keys = mj.get("range_constraints", {})
        ok6 = len(rc_keys) >= 2
        results.append(_case("dynamic(constraints)", ok6, "range_constraints=%s" % json.dumps(rc_keys)[:200]))
        if not ok6:
            return False

        # and must run at other shapes inside the range (dim0 batch / dim3 h
        # are the dynamic axes; dim1=3 and dim2=8 stay static)
        import torch
        ep = torch.export.load(red_path)
        for shape in ((6, 3, 8, 12), (2, 3, 8, 4)):
            x = torch.rand(*shape)
            with torch.no_grad():
                got = ep.module()(x)
                want = x.relu()
            ok7 = torch.equal(got, want)
            results.append(_case("dynamic(shape%s)" % (shape,), ok7, ""))
            if not ok7:
                return False

        # ---- case 3: the generated *_pnnx.py validates caller inputs against
        # the recorded range_constraints before torch.export.export, so an
        # out-of-range shape fails fast with a clear message instead of an
        # opaque guard error (and in-range shapes still re-export cleanly)
        dyn_py = os.path.join(workdir, os.path.basename(based) + "_pnnx.py")
        spec = importlib.util.spec_from_file_location("dyn_pnnx_val", dyn_py)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, workdir)
        spec.loader.exec_module(mod)

        raised = None
        try:
            mod.export_exported_program(
                example_inputs=(torch.rand(32, 3, 8, 8),),
                out_path=os.path.join(workdir, "dyn_oob_re.pt2"),
            )
        except ValueError as e:
            raised = str(e)
        ok8 = raised is not None and "must be within" in raised
        results.append(_case("dynamic(validate_oob)", ok8, "raised=%r" % raised))
        if not ok8:
            return False

        # boundary value (dim0=2, dim3=4) is inside the closed range and must
        # re-export normally
        mod.export_exported_program(
            example_inputs=(torch.rand(2, 3, 8, 4),),
            out_path=os.path.join(workdir, "dyn_in_re.pt2"),
        )
        ok9 = os.path.isfile(os.path.join(workdir, "dyn_in_re.pt2"))
        results.append(_case("dynamic(validate_inrange)", ok9, ""))
        if not ok9:
            return False

    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if test() else 1)
