# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import sys
import warnings
import importlib

warnings.filterwarnings("ignore")

import torch  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_warn_always(False)

# Keep the harness independent of the optional packaging dependency.
def _install_packaging_shim():
    import types
    import re as _re
    if "packaging" in sys.modules:
        return
    class _V:
        __slots__ = ("s", "t")
        def __init__(self, s):
            self.s = str(s)
            core = self.s.split("+")[0]
            nums = _re.findall(r"\d+", core)
            self.t = tuple(int(x) for x in nums[:4]) or (0,)
        def _o(self, o):
            return o.t if isinstance(o, _V) else _V(o).t
        def __lt__(self, o): return self.t < self._o(o)
        def __le__(self, o): return self.t <= self._o(o)
        def __gt__(self, o): return self.t > self._o(o)
        def __ge__(self, o): return self.t >= self._o(o)
        def __eq__(self, o): return self.t == self._o(o)
        def __ne__(self, o): return self.t != self._o(o)
        def __hash__(self): return hash(self.t)
        def __repr__(self): return "Version(%r)" % self.s
    pkg = types.ModuleType("packaging")
    ver = types.ModuleType("packaging.version")
    ver.parse = _V
    ver.Version = _V
    pkg.version = ver
    sys.modules["packaging"] = pkg
    sys.modules["packaging.version"] = ver

_install_packaging_shim()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PNNX_PYTHON"] = sys.executable
from pt2_crosscheck import find_pnnx, normalize_param  # noqa: E402

PNNX = find_pnnx()
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

SKIP_PREFIXES = (
    "test_ncnn_",
    "test_pt2_",
    "test_pnnx_",
    "test_torchaudio_",
    "test_transformers_",
)
SKIP_EXACT = {
    "test_resnet18.py",
    "test_convnext_tiny.py", "test_mobilenet_v2.py", "test_mobilenet_v3_small.py",
    "test_shufflenet_v2_x1_0.py", "test_squeezenet1_1.py", "test_swin_t.py", "test_vit_b_32.py",
    "test_torchvision_DeformConv2d.py", "test_torchvision_RoIAlign.py",
}


def list_tests():
    out = []
    for f in sorted(os.listdir(TEST_DIR)):
        if not f.startswith("test_") or not f.endswith(".py"):
            continue
        if f in SKIP_EXACT:
            continue
        if any(f.startswith(p) for p in SKIP_PREFIXES):
            continue
        out.append(f)
    return out


def extract_inputshape(src):
    m = re.search(r'inputshape=([^"\s]+)', src)
    ish = m.group(1) if m else None
    m2 = re.search(r'inputshape2=([^"\s]+)', src)
    ish2 = m2.group(1) if m2 else None
    return ish, ish2


def parse_shapes(ish):
    if not ish:
        return []
    shapes = []
    for part in ish.split("],["):
        part = part.strip("[] ")
        if not part:
            continue
        nums = []
        for tok in part.split(","):
            tok = tok.strip()
            if not tok:
                continue
            m = re.match(r'-?\d+', tok)
            if not m:
                return None
            nums.append(int(m.group()))
        shapes.append(nums)
    return shapes


def run_pnnx_quiet(ptx, ish):
    cmd = f"{PNNX} {ptx} inputshape={ish}" if ish else f"{PNNX} {ptx}"
    p = os.popen(cmd + " 2>&1 >/dev/null")
    err = p.read()
    ret = p.close()
    retcode = 0 if ret is None else (ret if isinstance(ret, int) else 1)
    return retcode, err[-400:]


def export_and_compare(net, inputs, ish, base):
    pt2 = base + ".pt2"
    pt = base + "_ts.pt"

    try:
        torch.export.save(torch.export.export(net, tuple(inputs)), pt2)
    except Exception as e:
        return ("EXPORT_FAIL", f"torch.export: {str(e)[:80]}")
    try:
        torch.jit.trace(net, tuple(inputs)).save(pt)
    except Exception as e:
        return ("EXPORT_FAIL", f"jit.trace: {str(e)[:80]}")

    r1, err1 = run_pnnx_quiet(pt2, ish)
    if r1 != 0:
        if "unsupported aten op" in err1:
            op = re.search(r'unsupported aten op[:\s]*([^\n]+)', err1)
            return ("UNSUPPORTED_OP", op.group(1) if op else err1)
        return ("PNNX_PT2_FAIL", err1[:120])
    r2, err2 = run_pnnx_quiet(pt, ish)
    if r2 != 0:
        return ("PNNX_PT_FAIL", err2[:120])

    p_pt2 = base + ".ncnn.param"
    p_pt = base + "_ts.ncnn.param"
    if not (os.path.exists(p_pt2) and os.path.exists(p_pt)):
        return ("NO_PARAM", "ncnn .param not generated")

    a = normalize_param(p_pt2)
    b = normalize_param(p_pt)
    if a == b:
        return ("PASS", f"{len(a)} lines identical")
    diff = []
    for i, (la, lb) in enumerate(zip(a, b)):
        if la != lb:
            diff.append(f"line{i}: pt2={la[:40]!r} pt={lb[:40]!r}")
            if len(diff) >= 2:
                break
    return ("DIFF", " | ".join(diff) if diff else f"len pt2={len(a)} pt={len(b)}")


def classify(name, src):
    modname = name[:-3]
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        return ("IMPORT_FAIL", f"{type(e).__name__}: {str(e)[:110]}")
    Model = getattr(mod, "Model", None)
    if Model is None:
        return ("SKIP", "no Model class")

    ish, ish2 = extract_inputshape(src)
    shapes = parse_shapes(ish)
    if not shapes:
        return ("SKIP", "no/complex inputshape")

    net = Model().eval()
    base = "sw_" + modname
    status, detail = export_and_compare(net, [torch.rand(*s) for s in shapes], ish, base)
    if status != "PASS" or not ish2:
        return status, detail

    shapes2 = parse_shapes(ish2)
    if not shapes2 or len(shapes2) != len(shapes):
        return ("SKIP", "no/complex inputshape2")

    status2, detail2 = export_and_compare(net, [torch.rand(*s) for s in shapes2], ish2, base + "_inputshape2")
    if status2 != "PASS":
        return status2, f"inputshape2: {detail2}"
    return ("PASS", f"{detail}; inputshape2 {detail2}")


def dump_ts(name, src):
    # Keep TorchScript output as the canonical reference.
    modname = name[:-3]
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        return ("IMPORT_FAIL", f"{type(e).__name__}: {str(e)[:110]}")
    Model = getattr(mod, "Model", None)
    if Model is None:
        return ("SKIP", "no Model class")
    ish, _ = extract_inputshape(src)
    shapes = parse_shapes(ish)
    if not shapes:
        return ("SKIP", "no/complex inputshape")
    inputs = [torch.rand(*s) for s in shapes]
    net = Model().eval()
    base = "sw_" + modname
    pt = base + "_ts.pt"
    try:
        torch.jit.trace(net, tuple(inputs)).save(pt)
    except Exception as e:
        return ("EXPORT_FAIL", f"jit.trace: {str(e)[:80]}")
    r, err = run_pnnx_quiet(pt, ish)
    if r != 0:
        return ("PNNX_PT_FAIL", err[:110])
    pnnx_ir = base + "_ts.pnnx.param"
    if not os.path.exists(pnnx_ir):
        return ("NO_PARAM", "pnnx IR not generated")
    return ("DUMPED", pnnx_ir)


def main():
    dump_mode = "--dump-ts" in sys.argv
    sel = [a for a in sys.argv[1:] if a != "--dump-ts"]
    tests = list_tests()
    if sel:
        selset = set(sel)
        tests = [t for t in tests if any(s in t for s in selset)]
    print(f"# {'dump-ts' if dump_mode else 'sweep'} {len(tests)} tests with pnnx={PNNX}\n")

    from collections import Counter
    results = []
    cnt = Counter()
    for t in tests:
        src = open(os.path.join(TEST_DIR, t)).read()
        try:
            status, detail = (dump_ts if dump_mode else classify)(t, src)
        except Exception as e:
            status, detail = ("ERROR", f"{type(e).__name__}: {str(e)[:80]}")
        results.append((t, status, detail))
        cnt[status] += 1
        mark = {"PASS": "OK", "DUMPED": "OK", "DIFF": "XX", "EXPORT_FAIL": "EE", "UNSUPPORTED_OP": "UU",
                "PNNX_PT2_FAIL": "EE", "PNNX_PT_FAIL": "EE", "NO_PARAM": "EE",
                "SKIP": "..", "IMPORT_FAIL": "EE", "ERROR": "EE"}.get(status, "??")
        print(f"[{mark}] {t:34s} {status:16s} {detail}")

    print("\n==== summary ====")
    for s in ("PASS", "DUMPED", "DIFF", "UNSUPPORTED_OP", "EXPORT_FAIL", "SKIP",
              "ERROR", "PNNX_PT2_FAIL", "PNNX_PT_FAIL", "NO_PARAM", "IMPORT_FAIL"):
        if cnt[s]:
            print(f"  {s}: {cnt[s]}")
    print(f"  TOTAL: {sum(cnt.values())}")

    if not dump_mode:
        unsup = [d for _, s, d in results if s == "UNSUPPORTED_OP"]
        if unsup:
            print("\n==== 需扩的 op（UNSUPPORTED_OP）====")
            for d in sorted(set(unsup)):
                print(f"  - {d}")
    else:
        failed = [t for t, s, _ in results if s not in ("DUMPED",)]
        if failed:
            print("\n==== dump 失败的测试 ====")
            for t in failed:
                print(f"  - {t}")
        print(f"\n# 规范 IR 参考文件位于 CWD: sw_*_ts.pnnx.param")


if __name__ == "__main__":
    main()
