# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# M4 批量算子交叉对拍工具（无需 ncnn python 绑定，只需 pnnx + torch）。
#
# 对每个算子模型，分别走 .pt2(torch.export) 与 .pt(torchscript) 两条路径经 pnnx 转 ncnn，
# 对比 ncnn .param 结构（归一化 op 名后缀后逐行 diff）。一致即证明 .pt2 路径与
# torchscript 管线等价产出——无 ncnn 绑定下的最强正确性证据。
#
# 用法：python pt2_crosscheck.py [name1 name2 ...]   (不带参数跑全部)

import os
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 让 pnnx 内部 popen 调的 python 就是当前 venv（装了 torch 的）解释器
os.environ["PNNX_PYTHON"] = sys.executable

# ---------- 算子电池 ----------
# 每项: (name, Model 工厂(返回 eval 的 net), inputs, inputshape_str)
BATTERY = []

def case(name, inputs, inputshape_str):
    def deco(cls):
        BATTERY.append((name, cls, inputs, inputshape_str))
        return cls
    return deco

@case("relu", (torch.randn(1, 3, 4, 4),), "[1,3,4,4]")
class M_relu(nn.Module):
    def forward(self, x):
        return F.relu(x)

@case("sigmoid", (torch.randn(1, 3, 4, 4),), "[1,3,4,4]")
class M_sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

@case("tanh", (torch.randn(1, 3, 4, 4),), "[1,3,4,4]")
class M_tanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

@case("silu", (torch.randn(1, 3, 4, 4),), "[1,3,4,4]")
class M_silu(nn.Module):
    def forward(self, x):
        return F.silu(x)

@case("softmax", (torch.randn(1, 3, 4, 4),), "[1,3,4,4]")
class M_softmax(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=1)

@case("flatten", (torch.randn(1, 3, 4, 4),), "[1,3,4,4]")
class M_flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)

@case("cat", (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)), "[1,3,4,4],[1,3,4,4]")
class M_cat(nn.Module):
    def forward(self, x, y):
        return torch.cat((x, y), dim=1)

@case("stack", (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)), "[1,3,4,4],[1,3,4,4]")
class M_stack(nn.Module):
    def forward(self, x, y):
        return torch.stack((x, y), dim=1)

@case("add", (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)), "[1,3,4,4],[1,3,4,4]")
class M_add(nn.Module):
    def forward(self, x, y):
        return x + y

@case("mul", (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)), "[1,3,4,4],[1,3,4,4]")
class M_mul(nn.Module):
    def forward(self, x, y):
        return x * y

@case("view", (torch.randn(1, 3, 4, 4),), "[1,3,4,4]")
class M_view(nn.Module):
    def forward(self, x):
        return x.view(1, -1)

@case("reshape", (torch.randn(1, 3, 4, 4),), "[1,3,4,4]")
class M_reshape(nn.Module):
    def forward(self, x):
        return x.reshape(1, -1)

@case("permute", (torch.randn(1, 3, 4, 4),), "[1,3,4,4]")
class M_permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)

@case("transpose", (torch.randn(1, 3, 4, 4),), "[1,3,4,4]")
class M_transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

# ---- 带权重算子：探测 state_dict->权重加载路径 ----
@case("conv2d", (torch.randn(1, 3, 8, 8),), "[1,3,8,8]")
class M_conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
    def forward(self, x):
        return self.conv(x)

@case("linear", (torch.randn(2, 12),), "[2,12]")
class M_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 8)
    def forward(self, x):
        return self.fc(x)

@case("batchnorm", (torch.randn(1, 3, 8, 8),), "[1,3,8,8]")
class M_batchnorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
    def forward(self, x):
        return self.bn(x)

@case("layernorm", (torch.randn(2, 12),), "[2,12]")
class M_layernorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(12)
    def forward(self, x):
        return self.ln(x)


# ---------- pnnx 二进制探测 ----------
def find_pnnx():
    env_bin = os.environ.get("PNNX_BIN", "")
    if env_bin and os.path.exists(env_bin):
        return env_bin
    for cand in ("../../src/pnnx", "build/src/pnnx"):
        if os.path.exists(cand):
            return cand
    here = os.path.dirname(os.path.abspath(__file__))
    build_pnnx = os.path.normpath(os.path.join(here, "..", "..", "build", "src", "pnnx"))
    if os.path.exists(build_pnnx):
        return build_pnnx
    return "pnnx"

PNNX = find_pnnx()


def run_pnnx(ptx, inputshape_str):
    cmd = f"{PNNX} {ptx} inputshape={inputshape_str}"
    return os.system(f"{cmd} > /dev/null 2>&1")  # 静默


def normalize_param(path):
    """读 ncnn .param，归一化：去掉 op 名后缀 _<digits>，便于两路径比对。"""
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 2 and parts[0] not in ("Input", "Output") and re.match(r"^[A-Za-z].*", parts[0]):
            # 第2 token 是 op 名，去掉 trailing _digits 与 pt2_ 前缀（两路径命名约定不同）
            parts[1] = re.sub(r"_\d+$", "", parts[1])
            parts[1] = re.sub(r"^pt2_", "", parts[1])
        out.append(" ".join(parts))
    return out


def crosscheck(name, Model, inputs, inputshape_str):
    net = Model().eval()
    base_pt2 = f"xc_{name}"
    base_pt = f"xc_{name}_ts"
    pt2_path = base_pt2 + ".pt2"
    pt_path = base_pt + ".pt"

    try:
        torch.export.save(torch.export.export(net, inputs), pt2_path)
        torch.jit.trace(net, inputs).save(pt_path)
    except Exception as e:
        return ("EXPORT_FAIL", f"export error: {e}")

    r1 = run_pnnx(pt2_path, inputshape_str)
    r2 = run_pnnx(pt_path, inputshape_str)
    if r1 != 0:
        return ("PNNX_PT2_FAIL", f"pnnx on .pt2 ret={r1}")
    if r2 != 0:
        return ("PNNX_PT_FAIL", f"pnnx on .pt ret={r2}")

    p_pt2 = base_pt2 + ".ncnn.param"
    p_pt = base_pt + ".ncnn.param"
    if not (os.path.exists(p_pt2) and os.path.exists(p_pt)):
        return ("NO_PARAM", "ncnn .param not generated")

    a = normalize_param(p_pt2)
    b = normalize_param(p_pt)
    if a == b:
        return ("PASS", f"{len(a)} lines identical")
    # 找第一处差异
    diff = []
    for i, (la, lb) in enumerate(zip(a, b)):
        if la != lb:
            diff.append(f"line {i}: pt2={la!r}  pt={lb!r}")
            if len(diff) >= 3:
                break
    extra = f"pt2={len(a)}L pt={len(b)}L; " + " | ".join(diff) if diff else f"length diff pt2={len(a)}L pt={len(b)}L"
    return ("DIFF", extra)


def main():
    sel = sys.argv[1:] or [c[0] for c in BATTERY]
    selset = set(sel)
    results = []
    for name, Model, inputs, ish in BATTERY:
        if name not in selset:
            continue
        status, detail = crosscheck(name, Model, inputs, ish)
        results.append((name, status, detail))
        mark = {"PASS": "OK", "DIFF": "XX", "EXPORT_FAIL": "EE", "PNNX_PT2_FAIL": "EE", "PNNX_PT_FAIL": "EE", "NO_PARAM": "EE"}[status]
        print(f"[{mark}] {name:12s} {status:14s} {detail}")

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n==== {n_pass}/{len(results)} PASS ====")
    fails = [r for r in results if r[1] != "PASS"]
    if fails:
        print("FAIL/差异：")
        for n, s, d in fails:
            print(f"  - {n}: {s} ({d})")


if __name__ == "__main__":
    main()
