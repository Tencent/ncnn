"""本地诊断 sweep 的 DIFF：直接比对已落盘的 sw_*.ncnn.param 与 sw_*_ts.ncnn.param。
复用 pt2_crosscheck.py 的 normalize_param，保证与 sweep 判定一致。无需跑 pnnx。"""
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tests", "ncnn"))
from pt2_crosscheck import normalize_param  # noqa: E402

D = HERE  # 与 sweep 的 CWD 一致（tools/pnnx/）

pairs = []
for f in sorted(os.listdir(D)):
    if f.startswith("sw_") and f.endswith("_ts.ncnn.param"):
        ts = f
        pt2 = f.replace("_ts.ncnn.param", ".ncnn.param")
        if os.path.exists(os.path.join(D, pt2)):
            pairs.append((f[len("sw_"):-len("_ts.ncnn.param")], pt2, ts))

print(f"# {len(pairs)} pairs\n")

diffs = []
for name, p2, pts in pairs:
    a = normalize_param(os.path.join(D, p2))
    b = normalize_param(os.path.join(D, pts))
    if a == b:
        continue
    first = None
    for i, (la, lb) in enumerate(zip(a, b)):
        if la != lb:
            first = (i, la, lb)
            break
    if first is None:
        first = (None, f"len pt2={len(a)}", f"len pt={len(b)}")
    diffs.append((name, first, len(a), len(b)))

print(f"# DIFF count = {len(diffs)}\n")
# 按首处差异的 op 类型归类
kind = Counter()
for name, (i, la, lb), _, _ in diffs:
    toks = (la or "").split()
    op = toks[0] if toks else "?"
    if op == "7767517" or re.match(r"^\d+ \d+$", la or ""):
        op = "HEADER(count/len)"
    kind[op] += 1

print("==== DIFF 首处差异所在 op 分布 ====")
for op, c in kind.most_common():
    print(f"  {op}: {c}")

print("\n==== DIFF 明细（首处差异）====")
for name, (i, la, lb), na, nb in diffs:
    print(f"\n[{name}]  pt2={na}L pt={nb}L  line{i}")
    print(f"   pt2: {(la or '')[:150]}")
    print(f"   pt : {(lb or '')[:150]}")
