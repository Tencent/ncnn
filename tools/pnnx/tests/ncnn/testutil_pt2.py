# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# pt2 对拍公共 helper：把同一模型走 torch.export(.pt2)->pnnx->ncnn，与 torch 参考输出对拍。
# 用法见 test_pt2_*.py。base 名须带后缀（如 test_pt2_smoke），避免与 torchscript 路径产物冲突。

import os
import re
import sys
import torch


def _prepare_ncnn_input(tensor, batch_index):
    """按生成的 ncnn wrapper 语义准备输入，拒绝隐式的任意 reshape。

    batch_index=233 表示 ncnn 不承载 batch 维；仅当 torch 输入首维确实为
    size-1 时才剥离，剩余形状必须与 ncnn 的 (c,h,w)/(c,h) 解释一致。
    """
    import numpy as np

    value = np.ascontiguousarray(tensor.numpy(), dtype=np.float32)
    if batch_index == 233 and value.ndim >= 3 and value.shape[0] == 1:
        return value.reshape(value.shape[1:])
    return value


def _restore_ncnn_output(value, reference, batch_index):
    """将 ncnn 输出恢复为 torch shape，只允许已知的 batch 轴剥离关系。

    batch_index=0（ncnn 承载 batch 维）与 batch_index=233（ncnn 不承载
    batch 维）都只可能出现一种额外形态：torch 参考输出首维为 size-1 且
    ncnn 输出恰好等于去掉该维的形状（batch 轴被折叠/剥除）。仅允许这一
    种还原；错序或结构不同但 numel 相同的形状一律拒绝。
    """
    if value.shape == reference.shape:
        return value
    if (reference.ndim >= 1 and reference.shape[0] == 1
            and value.shape == reference.shape[1:]):
        return value.reshape(reference.shape)
    raise ValueError(
        f"ncnn output shape {value.shape} cannot represent torch shape {reference.shape} "
        f"with batch_index={batch_index}"
    )


def run_pt2_test(net, inputs, inputshape_str, base_name, atol=1e-4, device="cpu"):
    """返回 True 表示 ncnn 输出与 torch 参考一致。

    net          : eval 好的 nn.Module
    inputs       : tuple of torch.Tensor（导出与参考推理用）
    inputshape_str : 传给 pnnx 的 inputshape 字符串，如 "[1,3,4,4],[1,3,4,4]"
    base_name    : 产物基名（不含扩展名），如 "test_pt2_smoke"
    """
    net = net.eval()
    if device == "cpu":
        net = net.cpu()
        inputs = tuple(t.cpu() for t in inputs)

    # 1) torch 参考输出
    with torch.no_grad():
        a = net(*inputs)
    if not isinstance(a, tuple):
        a = (a,)

    # 2) 导出 .pt2
    pt2_path = base_name + ".pt2"
    try:
        ep = torch.export.export(net, inputs)
        torch.export.save(ep, pt2_path)
    except Exception as e:
        print(f"[pt2] export failed for {base_name}: {e}")
        return False

    # 3) pnnx 转换（PNNX_PYTHON 指向当前解释器，确保 popen 调的 python 装了 torch）
    os.environ["PNNX_PYTHON"] = sys.executable

    # 探测 pnnx 二进制：PNNX_BIN > ctest 约定(../../src/pnnx) > 源码 build
    pnnx_bin = os.environ.get("PNNX_BIN", "")
    if not pnnx_bin or not os.path.exists(pnnx_bin):
        cand = os.path.join("../../src/pnnx")  # ctest 工作目录约定
        if os.path.exists(cand):
            pnnx_bin = cand
        else:
            # 从脚本位置推 build/src/pnnx（手动从源码目录跑）
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pnnx_build = os.path.normpath(os.path.join(script_dir, "..", "..", "build", "src", "pnnx"))
            if os.path.exists(pnnx_build):
                pnnx_bin = pnnx_build
            else:
                pnnx_bin = "pnnx"  # 退回 PATH

    cmd = f"{pnnx_bin} {pt2_path} inputshape={inputshape_str}"
    print(f"[pt2] run: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"[pt2] pnnx failed (ret={ret}) for {base_name}")
        return False

    # 4) ncnn 推理：直接驱动 pyncnn 喂同一份 inputs。
    #    不走产物 _ncnn.py 的 test_inference()——它内部 manual_seed(0) 重放输入，
    #    若调用方在构造输入前消耗过 RNG（如模型权重初始化），两边输入会悄然不同。
    #    输入/输出 blob 名经 pass_ncnn 的 convert_input/convert_output 统一为 in0..N / out0..N。
    try:
        import numpy as np
        import ncnn
        with open(base_name + "_ncnn.py", "r", encoding="utf-8") as f:
            src = f.read()
        out_names = re.findall(r'ex\.extract\("([^"]+)"\)', src)
        if not out_names:
            out_names = ["out0"]
        # batch_index 语义是 pnnx 按 Input operand 的 __ncnn_batch_axis 生成的权威值，
        # 直接从产物 _ncnn.py 提取，不自行假设（硬编码错值会丢 batch 维/形状错乱）
        m = re.search(r'ncnn\.Mat\([^)]*batch_index=(\d+)\)', src)
        in_batch_index = int(m.group(1)) if m else 233
        m = re.search(r'numpy\(batch_index=(\d+)\)', src)
        out_batch_index = int(m.group(1)) if m else 233
        outs = []
        with ncnn.Net() as net:
            net.load_param(base_name + ".ncnn.param")
            net.load_model(base_name + ".ncnn.bin")
            with net.create_extractor() as ex:
                for i, t in enumerate(inputs):
                    tnp = _prepare_ncnn_input(t, in_batch_index)
                    ex.input(f"in{i}", ncnn.Mat(tnp, batch_index=in_batch_index).clone())
                for nm in out_names:
                    _, o = ex.extract(nm)
                    raw = o.numpy(batch_index=out_batch_index)
                    outs.append(torch.from_numpy(raw))
        b = tuple(outs)
    except Exception as e:
        print(f"[pt2] ncnn inference failed for {base_name}: {e}")
        return False

    if not isinstance(b, tuple):
        b = (b,)

    # 5) 对拍
    import os as _os
    if _os.environ.get("PNNX_TESTUTIL_DBG"):
        w_dbg = None
        for p_ in net.parameters() if hasattr(net, "parameters") else []:
            w_dbg = p_.detach().flatten()[:3].tolist()
            break
        print(f"[dbg] a[:3]={a[0].flatten()[:3].tolist()} b[:3]={b[0].flatten()[:3].tolist()} net_w[:3]={w_dbg}")
    ok = True
    for i, (a0, b0) in enumerate(zip(a, b)):
        # 只允许 ncnn wrapper 明确声明的 batch_index=0 维度剥离关系。
        b0 = torch.from_numpy(_restore_ncnn_output(b0.numpy(), a0.numpy(), out_batch_index))
        if torch.allclose(a0, b0, atol, atol):
            print(f"[pt2] out[{i}]  shape a={tuple(a0.shape)} b={tuple(b0.shape)}  MATCH")
        else:
            ok = False
            print(f"[pt2] out[{i}]  shape a={tuple(a0.shape)} b={tuple(b0.shape)}  MISMATCH  "
                  f"max|d|={ (a0 - b0).abs().max().item() if a0.shape == b0.shape else 'shape-diff' }")
    return ok
