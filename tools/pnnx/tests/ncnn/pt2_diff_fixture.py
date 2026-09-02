# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# M0 夹具:固定 12 个 DIFF 场景的双路径基线产物,作为 M1-M5 改造期间的回归参照(docs/15)。
#
# 用法(在 tools/pnnx/tests/ncnn/ 下运行,需 PNNX_BIN 指向 pnnx 二进制):
#   python pt2_diff_fixture.py                # 采集全部 12 DIFF 场景基线
#   python pt2_diff_fixture.py name1 name2    # 子串选择场景
#   python pt2_diff_fixture.py --verify       # 当前结果 vs 基线,报告回归
#
# 产物目录 pt2_diff_baseline/ 为本地生成物(.git/info/exclude),不进 commit。

import os
import sys
import importlib
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from pt2_sweep import extract_inputshape, parse_shapes, run_pnnx_quiet  # noqa: E402
from pt2_crosscheck import normalize_param  # noqa: E402

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(TEST_DIR, "pt2_diff_baseline")

# N4 收口 12 DIFF 场景(docs/08 2026-09-01 清单;docs/15 M1-M5 对应)
DIFF_SCENARIOS = (
    "test_F_interpolate",        # M1 语义等价编码差
    "test_nn_Upsample",          # M1 语义等价编码差
    "test_nn_Conv3d",            # M1 reflect+groups depthwise 判定
    "test_nn_Linear",            # M1 weight_norm 参数化
    "test_F_local_response_norm",  # M2 LRN 分解链
    "test_nn_LocalResponseNorm",   # M2 LRN 分解链
    "test_torch_stft",           # M3 stft 窗口常量+分解链
    "test_torch_istft",          # M3 istft 分解链
    "test_nn_GRU",               # M4 循环网络分解
    "test_nn_LSTM",              # M4 循环网络分解
    "test_nn_RNN",               # M4 循环网络分解
    "test_Tensor_slice_copy",    # M5 copy_ in-place 链
)

# 每场景收集的产物:pnnx 中间 IR(+权重) + ncnn param/bin + 源模型
ARTIFACTS = (
    "{m}.pnnx.param", "{m}.pnnx.bin",
    "{m}_ts.pnnx.param", "{m}_ts.pnnx.bin",
    "{m}.ncnn.param", "{m}.ncnn.bin",
    "{m}_ts.ncnn.param", "{m}_ts.ncnn.bin",
    "{m}.pt2", "{m}_ts.pt",
)


def run_scenario(modname):
    """导出双路径模型并跑 pnnx,产物落在 CWD。返回 (inputshape, error)。"""
    src = open(os.path.join(TEST_DIR, modname + ".py")).read()
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        return None, f"import fail: {type(e).__name__}: {str(e)[:80]}"
    Model = getattr(mod, "Model", None)
    if Model is None:
        return None, "no Model class"
    ish, _ = extract_inputshape(src)
    shapes = parse_shapes(ish)
    if not shapes:
        return None, "no/complex inputshape"
    inputs = [torch.rand(*s) for s in shapes]
    net = Model().eval()
    m = "sw_" + modname
    try:
        torch.export.save(torch.export.export(net, tuple(inputs)), m + ".pt2")
        torch.jit.trace(net, tuple(inputs)).save(m + "_ts.pt")
    except Exception as e:
        return ish, f"export fail: {str(e)[:80]}"
    r1, err1 = run_pnnx_quiet(m + ".pt2", ish)
    if r1 != 0:
        return ish, f"pt2 path fail: {err1[:80]}"
    r2, err2 = run_pnnx_quiet(m + "_ts.pt", ish)
    if r2 != 0:
        return ish, f"ts path fail: {err2[:80]}"
    return ish, None


def dispose(modname, keep):
    """CWD 场景产物 → keep 时移入基线目录,否则删除。返回 param 结构差异行列表。"""
    m = "sw_" + modname
    d = os.path.join(BASELINE_DIR, modname)
    if keep:
        os.makedirs(d, exist_ok=True)
    # 对比必须在删除/移动之前(keep=False 时文件读后即焚)
    diffs = None
    if os.path.exists(m + ".ncnn.param") and os.path.exists(m + "_ts.ncnn.param"):
        a = normalize_param(m + ".ncnn.param")
        b = normalize_param(m + "_ts.ncnn.param")
        diffs = [f"line{i}: pt2={la!r} pt={lb!r}" for i, (la, lb) in enumerate(zip(a, b)) if la != lb]
    for pat in ARTIFACTS:
        f = pat.format(m=m)
        if not os.path.exists(f):
            continue
        if keep:
            os.replace(f, os.path.join(d, f))
        else:
            os.remove(f)
    return diffs


def capture(mods):
    records = []
    for modname in mods:
        ish, err = run_scenario(modname)
        if err:
            print(f"[EE] {modname:32s} {err}")
            records.append((modname, ish, None, err))
            continue
        diffs = dispose(modname, keep=True)
        status = "IDENTICAL" if diffs == [] else f"{len(diffs)} diff lines"
        print(f"[OK] {modname:32s} ish={ish} {status}")
        records.append((modname, ish, diffs, None))

    path = os.path.join(BASELINE_DIR, "index.md")
    lines = [
        "# pt2 DIFF 基线索引(M0 夹具)",
        "",
        "12 个 N4 收口 DIFF 场景的双路径基线产物;M1-M5 每清零一类,对应场景转 PASS 后更新此表。",
        "",
        "| 场景 | inputshape | 状态 | 首差异行 |",
        "|---|---|---|---|",
    ]
    for modname, ish, diffs, err in records:
        first = (diffs[0][:120] if diffs else ("-" if diffs == [] else err)) or "-"
        lines.append(f"| {modname} | {ish} | {len(diffs) if diffs else 0} diff | {first} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n# baseline -> {BASELINE_DIR}")
    print(f"# index -> {path}")


def verify(mods):
    bad = 0
    for modname in mods:
        base_d = os.path.join(BASELINE_DIR, modname)
        m = "sw_" + modname
        pa = os.path.join(base_d, m + ".ncnn.param")
        pb = os.path.join(base_d, m + "_ts.ncnn.param")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"[??] {modname:32s} no baseline, run capture first")
            bad += 1
            continue
        base_diffs = [
            f"line{i}: pt2={la!r} pt={lb!r}"
            for i, (la, lb) in enumerate(zip(normalize_param(pa), normalize_param(pb)))
            if la != lb
        ]
        ish, err = run_scenario(modname)
        if err:
            print(f"[EE] {modname:32s} {err}")
            bad += 1
            continue
        cur = dispose(modname, keep=False)
        if cur == []:
            print(f"[++]{modname:31s} now PASS — 场景已修复,请更新基线与 docs/08")
        elif cur == base_diffs:
            print(f"[==]{modname:31s} unchanged ({len(base_diffs)} diff lines)")
        else:
            print(f"[!!]{modname:31s} changed: {len(base_diffs)} -> {len(cur)} diff lines")
            bad += 1
    return bad


def main():
    verify_mode = "--verify" in sys.argv
    sel = [a for a in sys.argv[1:] if a != "--verify"]
    mods = [x for x in DIFF_SCENARIOS if not sel or any(s in x for s in sel)]
    print(f"# fixture {'verify' if verify_mode else 'capture'} {len(mods)} scenarios with pnnx\n")
    bad = verify(mods) if verify_mode else 0
    if not verify_mode:
        os.makedirs(BASELINE_DIR, exist_ok=True)
        capture(mods)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
