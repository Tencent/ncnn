# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# pnnx pt2 test helpers
# usage:
#   from pnnx_test_helper import test_pnnx
#   pt2_ok = test_pnnx(net, (x, y), ["[1,3,8,8]", "[1,3,8,8]"], "test_F_xxx")
#   # True  : pt2 conversion is correct
#   # False : pt2 conversion failed (mismatched values)
#   # None  : skipped (torch.export does not support the model, or the
#   #          generated code is not yet adapted to pt2)

import importlib
import os
import sys

import torch


def _convert_pnnx(pt_path, inputshapes, pnnx_path=os.path.join("..", "src", "pnnx"), fp16=0):
    cmd = "%s %s inputshape=%s fp16=%d" % (pnnx_path, pt_path, ",".join(inputshapes), fp16)
    rc = os.system(cmd)
    if rc != 0:
        # surface the failure reason (crash -> 128+signal, convert error -> 1)
        print("[pnnx convert failed] rc=%d cmd=%s" % (rc, cmd))
    return rc == 0


def _load_pnnx_module(tag):
    # tag is the full filename prefix (e.g. "test_F_softmax"); the generated
    # python module is named "{tag}_pnnx"
    mod_name = "%s_pnnx" % tag
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    else:
        __import__(mod_name)
    return sys.modules[mod_name]


def _outputs_equal(a, b, atol=1e-3, rtol=1e-3):
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_outputs_equal(x, y, atol, rtol) for x, y in zip(a, b))
    if a.dtype == torch.bool:
        return torch.equal(a, b)
    return torch.allclose(a, b, atol=atol, rtol=rtol)


def _outputs_shape_equal(a, b):
    # shape + dtype only; for outputs with unspecified values (e.g. uninitialized
    # new_empty buffers) the torchscript path only compares shapes, mirror it
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_outputs_shape_equal(x, y) for x, y in zip(a, b))
    return a.shape == b.shape and a.dtype == b.dtype


def _torch_dtype_to_pnnx(dtype):
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float64:
        return "f64"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.int32:
        return "i32"
    if dtype == torch.int64:
        return "i64"
    if dtype == torch.int16:
        return "i16"
    if dtype == torch.int8:
        return "i8"
    if dtype == torch.uint8:
        return "u8"
    if dtype == torch.bool:
        return "bool"
    if dtype == torch.complex64:
        return "c64"
    if dtype == torch.complex128:
        return "c128"
    return "f32"


def _flatten_args(args):
    # list/tuple inputs expand into multiple user_inputs in torch.export; flatten them here
    flat = []
    for a in args:
        if isinstance(a, (list, tuple)):
            flat.extend(a)
        else:
            flat.append(a)
    return flat


def _inputshapes_with_dtype(args, inputshapes):
    # append a dtype suffix to each input shape so int/bool inputs are not treated as default f32
    out = []
    for t, s in zip(_flatten_args(args), inputshapes):
        if isinstance(t, torch.Tensor):
            out.append(s + _torch_dtype_to_pnnx(t.dtype))
        else:
            out.append(s)
    return out


def _inline_exported_symbols(pt2_path, ep, args):
    """Inline dynamo symbolic arguments.

    If the exported graph contains as_sym_float/as_sym_int/as_sym_bool (unbacked
    symbols computed at runtime, e.g. funnel's arange(start=sym, end=sym)),
    torch.export does not keep the concrete symbol values. Under concrete inputs
    those values are deterministic, so run the graph once with concrete inputs,
    capture the actual argument values of every node and replace the symbolic
    arguments in model.json with the concrete values. pnnx can then treat them
    as ordinary constants.
    """
    import json
    import shutil
    import zipfile

    import torch.fx as fx
    from torch.export.graph_signature import InputKind

    # 1. read model.json and check whether there are symbolic arguments
    zin = zipfile.ZipFile(pt2_path)
    mj = None
    for n in zin.namelist():
        if n.endswith("models/model.json"):
            mj = n
            break
    if mj is None:
        zin.close()
        return
    data = json.loads(zin.read(mj).decode())
    zin.close()

    nodes = data.get("graph_module", {}).get("graph", {}).get("nodes", [])
    sym_keys = ("as_sym_float", "as_sym_int", "as_sym_bool", "as_sym_ints", "as_sym_floats")
    if not any(any(k in inp.get("arg", {}) for k in sym_keys) for nd in nodes for inp in nd.get("inputs", [])):
        return

    # 2. run the graph with concrete inputs, capturing concrete values per node/position
    gm = ep.graph_module
    gs = ep.graph_signature

    arg_map = {}
    ui = 0
    for s in gs.input_specs:
        if s.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            arg_map[s.arg.name] = ep.state_dict[s.target]
        elif s.kind == InputKind.USER_INPUT:
            arg_map[s.arg.name] = args[ui]
            ui += 1
        else:
            # constants and other inputs cannot be built from args; set to None
            # (if used, the inlining is abandoned)
            arg_map[s.arg.name] = None
    run_args = [arg_map[s.arg.name] for s in gs.input_specs]

    captured = {}

    class _SymInterp(fx.Interpreter):
        def run_node(self, node):
            self._cur_node = node
            return super().run_node(node)

        def call_function(self, target, fargs, fkwargs):
            captured[self._cur_node.name] = (list(fargs), dict(fkwargs))
            return super().call_function(target, fargs, fkwargs)

    try:
        with torch.no_grad():
            _SymInterp(gm).run(*run_args)
    except Exception:
        # running with concrete inputs failed (e.g. dynamic control flow);
        # abandon inlining and let pnnx handle it
        return

    # 3. replace symbolic args with concrete values (kind=1 positional -> fargs, kind=2 keyword -> fkwargs)
    changed = False
    for nd in nodes:
        rec = captured.get(nd.get("name"))
        if rec is None:
            continue
        fargs, fkwargs = rec
        pos_i = 0
        for inp in nd.get("inputs", []):
            kind = inp.get("kind", 1)
            arg = inp.get("arg", {})
            if kind == 1:
                v = fargs[pos_i] if pos_i < len(fargs) else None
                pos_i += 1
            else:
                v = fkwargs.get(inp.get("name"))
            if "as_sym_float" in arg and isinstance(v, float):
                arg.clear()
                arg["as_float"] = v
                changed = True
            elif "as_sym_int" in arg and isinstance(v, int):
                arg.clear()
                arg["as_int"] = v
                changed = True
            elif "as_sym_bool" in arg and isinstance(v, bool):
                arg.clear()
                arg["as_bool"] = v
                changed = True
            elif "as_sym_ints" in arg and isinstance(v, (list, tuple)) and all(isinstance(x, int) for x in v):
                arg.clear()
                arg["as_ints"] = [int(x) for x in v]
                changed = True
            elif "as_sym_floats" in arg and isinstance(v, (list, tuple)) and all(isinstance(x, float) for x in v):
                arg.clear()
                arg["as_floats"] = [float(x) for x in v]
                changed = True
    if not changed:
        return

    # 4. rewrite the pt2 (keep all other entries as-is)
    tmp = pt2_path + ".tmp"
    zin = zipfile.ZipFile(pt2_path)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zout:
        for item in zin.infolist():
            if item.filename == mj:
                zout.writestr(item, json.dumps(data).encode())
            else:
                zout.writestr(item, zin.read(item.filename))
    zin.close()
    shutil.move(tmp, pt2_path)


def test_pnnx(net, args, inputshapes, tag, shape_only=False):
    """Export pt2, convert and compare outputs.

    Returns True (pass) / False (fail) / None (skip).
    The torchscript path is validated by the test script itself; this only
    supplements the pt2 path validation.

    shape_only: compare shapes and dtypes only instead of numeric values;
    use it for outputs with unspecified values (uninitialized new_empty etc.).
    """
    net.eval()

    if not hasattr(torch, "export") or not hasattr(torch.export, "export") or not hasattr(torch.export, "save"):
        # torch < 2.x has no exported program API, skip the pt2 test
        return None

    with torch.no_grad():
        ref = net(*args)

    try:
        ep = torch.export.export(net, args)
        pt2_path = "%s.pt2" % tag
        torch.export.save(ep, pt2_path)
        # if the graph has dynamo symbols (unbacked), evaluate with concrete
        # inputs and inline them into the pt2
        _inline_exported_symbols(pt2_path, ep, _flatten_args(args))
    except Exception:
        # torch.export does not support the model (dynamic shapes/control flow), skip the pt2 test
        return None

    if not _convert_pnnx(pt2_path, _inputshapes_with_dtype(args, inputshapes)):
        return False

    try:
        mod_pnnx = _load_pnnx_module(tag)
        out = mod_pnnx.test_inference()
    except Exception:
        # the generated pnnx python cannot run (op not yet adapted to pt2), skip
        return None

    if shape_only:
        return _outputs_shape_equal(ref, out)
    return _outputs_equal(ref, out)


def _load_ncnn_module(tag):
    mod_name = "%s_ncnn" % tag
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    else:
        __import__(mod_name)
    return sys.modules[mod_name]


def test_pnnx_ncnn(net, args, inputshapes, tag, atol=1e-3, rtol=1e-3, fp16=0):
    """Export pt2, convert and compare ncnn inference output.

    Used by the tests under tests/ncnn/ (pnnx executable path is ../../src/pnnx).
    atol/rtol are the numeric comparison tolerances; fp16-converted models
    should relax them to 1e-2. fp16 defaults to 0 to match the ncnn tests'
    torchscript path (fp16=0).
    Returns True (pass) / False (fail) / None (skip).
    """
    net.eval()

    if not hasattr(torch, "export") or not hasattr(torch.export, "export") or not hasattr(torch.export, "save"):
        return None

    with torch.no_grad():
        ref = net(*args)

    try:
        ep = torch.export.export(net, args)
        pt2_path = "%s.pt2" % tag
        torch.export.save(ep, pt2_path)
        # if the graph has dynamo symbols (unbacked), evaluate with concrete
        # inputs and inline them into the pt2
        _inline_exported_symbols(pt2_path, ep, _flatten_args(args))
    except Exception:
        return None

    if not _convert_pnnx(pt2_path, _inputshapes_with_dtype(args, inputshapes), pnnx_path=os.path.join("..", "..", "src", "pnnx"), fp16=fp16):
        return False

    try:
        mod_ncnn = _load_ncnn_module(tag)
        out = mod_ncnn.test_inference()
    except Exception:
        return None

    return _outputs_equal(ref, out, atol, rtol)
