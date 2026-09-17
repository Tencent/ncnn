# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import subprocess
import sys
import torch


def _as_output_tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _prepare_ncnn_input(tensor, batch_index):
    import numpy as np

    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    value = np.ascontiguousarray(tensor.numpy(), dtype=np.float32)
    if batch_index == 233 and value.ndim >= 3 and value.shape[0] == 1:
        return value.reshape(value.shape[1:])
    return value


def _restore_ncnn_output(value, reference, batch_index):
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
    net = net.eval()
    if device != "cpu":
        raise ValueError(f"unsupported test device: {device}")
    net = net.cpu()
    inputs = tuple(t.cpu() for t in inputs)

    with torch.no_grad():
        a = net(*inputs)
    a = _as_output_tuple(a)

    pt2_path = base_name + ".pt2"
    try:
        ep = torch.export.export(net, inputs)
        torch.export.save(ep, pt2_path)
    except Exception as e:
        print(f"[pt2] export failed for {base_name}: {e}")
        return False

    os.environ["PNNX_PYTHON"] = sys.executable

    pnnx_bin = os.environ.get("PNNX_BIN", "")
    if not pnnx_bin or not os.path.exists(pnnx_bin):
        cand = os.path.join("../../src/pnnx")
        if os.path.exists(cand):
            pnnx_bin = cand
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pnnx_build = os.path.normpath(os.path.join(script_dir, "..", "..", "build", "src", "pnnx"))
            if os.path.exists(pnnx_build):
                pnnx_bin = pnnx_build
            else:
                pnnx_bin = "pnnx"

    cmd = [pnnx_bin, pt2_path, f"inputshape={inputshape_str}"]
    print(f"[pt2] run: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
    except OSError as e:
        print(f"[pt2] pnnx failed to start for {base_name}: {e}")
        return False
    if result.returncode != 0:
        print(f"[pt2] pnnx failed (ret={result.returncode}) for {base_name}")
        return False

    # Drive pyncnn directly because generated inference reseeds inputs.
    try:
        import numpy as np
        import ncnn
        with open(base_name + "_ncnn.py", "r", encoding="utf-8") as f:
            src = f.read()
        out_names = re.findall(r'ex\.extract\("([^"]+)"\)', src)
        if not out_names:
            out_names = ["out0"]
        in_batch_indices = [int(v) for v in re.findall(r'ncnn\.Mat\(.*batch_index=(\d+)\)', src)]
        out_batch_indices = [int(v) for v in re.findall(r'numpy\(batch_index=(\d+)\)', src)]
        if len(in_batch_indices) < len(inputs) or len(out_batch_indices) < len(out_names):
            raise ValueError("generated ncnn wrapper is missing per-operand batch_index")
        outs = []
        with ncnn.Net() as net:
            net.load_param(base_name + ".ncnn.param")
            net.load_model(base_name + ".ncnn.bin")
            with net.create_extractor() as ex:
                for i, t in enumerate(inputs):
                    batch_index = in_batch_indices[i]
                    tnp = _prepare_ncnn_input(t, batch_index)
                    ex.input(f"in{i}", ncnn.Mat(tnp, batch_index=batch_index).clone())
                for i, nm in enumerate(out_names):
                    _, o = ex.extract(nm)
                    raw = o.numpy(batch_index=out_batch_indices[i])
                    outs.append(torch.from_numpy(raw))
        b = tuple(outs)
    except Exception as e:
        print(f"[pt2] ncnn inference failed for {base_name}: {e}")
        return False

    b = _as_output_tuple(b)

    if len(a) != len(b):
        print(f"[pt2] output count mismatch for {base_name}: "
              f"torch={len(a)} ncnn={len(b)}")
        return False

    import os as _os
    if _os.environ.get("PNNX_TESTUTIL_DBG"):
        w_dbg = None
        for p_ in net.parameters() if hasattr(net, "parameters") else []:
            w_dbg = p_.detach().flatten()[:3].tolist()
            break
        print(f"[dbg] a[:3]={a[0].flatten()[:3].tolist()} b[:3]={b[0].flatten()[:3].tolist()} net_w[:3]={w_dbg}")
    ok = True
    for i, (a0, b0) in enumerate(zip(a, b)):
        reference = a0.float() if a0.dtype == torch.bfloat16 else a0
        b0 = torch.from_numpy(_restore_ncnn_output(b0.numpy(), reference.numpy(), out_batch_indices[i]))
        if torch.allclose(reference.float(), b0.float(), atol, atol):
            print(f"[pt2] out[{i}]  shape a={tuple(a0.shape)} b={tuple(b0.shape)}  MATCH")
        else:
            ok = False
            print(f"[pt2] out[{i}]  shape a={tuple(a0.shape)} b={tuple(b0.shape)}  MISMATCH  "
                  f"max|d|={ (reference.float() - b0.float()).abs().max().item() if a0.shape == b0.shape else 'shape-diff' }")
    return ok
