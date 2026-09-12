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

# pt2 outcome expectation table: every test's pt2 channel is audited against it
# so a silently dropped test (None) is a visible regression instead of a quiet
# skip. unlisted tags default to "pass"; only torch.export-incompatible models
# are recorded as "export-skip".
try:
    from pt2_expectations import EXPECT as _PT2_EXPECT
except Exception:
    _PT2_EXPECT = {}


def _pt2_expectation(exp):
    # normalize one expectation-table value to (outcome, needle):
    #   absent / "pass"      -> ("pass", None)
    #   "export-skip"        -> ("skip", None)
    #   {"outcome": "skip", "needle": "..."} -> ("skip", needle)
    # needle: when an export-skip is reached, torch.export must reject the model
    # with an error text containing this pinned diagnostic substring, so a
    # model that starts failing for a different reason is a visible regression.
    if isinstance(exp, dict):
        return exp.get("outcome", "skip"), exp.get("needle")
    if exp == "export-skip":
        return "skip", None
    return "pass", None

# optional discovery knobs:
#   PNNX_PT2_RESULT_LOG=<file>  append "tag<TAB>result[<TAB>exporter-error]" per
#                              test_pnnx call; for a skip (None) the exporter
#                              error text is appended so a diagnostic needle can
#                              be picked from it for pt2_expectations.py
#   PNNX_PT2_RECORD_ONLY=1      disable expectation enforcement (only record)
_PT2_RESULT_LOG = os.environ.get("PNNX_PT2_RESULT_LOG", "")
_PT2_RECORD_ONLY = os.environ.get("PNNX_PT2_RECORD_ONLY", "") == "1"


def _record_pt2_result(tag, result, err=""):
    if not _PT2_RESULT_LOG:
        return
    try:
        with open(_PT2_RESULT_LOG, "a") as f:
            f.write("%s\t%s" % (tag, result))
            if result is None:
                # single-line prefix of the exporter error, stable enough to
                # pick a needle from
                flat = " ".join((err or "").split())
                f.write("\t%s" % flat[:200])
            f.write("\n")
    except Exception:
        pass


def _check_pt2_expectation(tag, result, err=""):
    # audit one pt2 outcome against the pinned expectation (default "pass")
    if _PT2_RECORD_ONLY:
        return result
    outcome, needle = _pt2_expectation(_PT2_EXPECT.get(tag))
    if result is None:
        # torch.export rejected the model: allowed only when recorded
        if outcome != "skip":
            print("[pt2-expect] %s: expected %s but torch.export skipped (None) -- "
                  "record it in pt2_expectations.py if deliberate" % (tag, outcome))
            return False
        if needle and needle not in (err or ""):
            print("[pt2-expect] %s: export-skip reached but the exporter error no longer "
                  "matches the pinned diagnostic %r\n  got: %s" % (tag, needle, (err or "")[:500]))
            return False
        if not needle:
            # a skip recorded as a bare "export-skip" has no pinned reason; the
            # whole pt2 channel is audited, so encourage pinning a diagnostic so
            # a model that starts failing for a *different* reason is caught
            print("[pt2-expect] %s: export-skip has no pinned diagnostic needle - "
                  "add one in pt2_expectations.py" % tag)
        return None
    # conversion ran: a previously-recorded export-skip that now passes is a
    # stale table entry (improvement) - keep passing but make it visible
    if outcome == "skip" and result is True:
        print("[pt2-expect] %s: recorded as export-skip but now passes -- "
              "move it to pass in pt2_expectations.py" % tag)
    return result


def _finalize_pt2(tag, result, err=""):
    _record_pt2_result(tag, result, err)
    return _check_pt2_expectation(tag, result, err)


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


def _flatten_leaves(x):
    # flatten a (possibly nested) pytree structure into a flat list of leaves
    #
    # torch.export serializes graph outputs as a flat sequence of tensors in
    # pytree leaf order: a dict return becomes several flat user_outputs, a
    # nested tuple/list is expanded, and a single-element tuple collapses into
    # the bare value. its user_inputs follow the same flattening on the input
    # side. reuse torch.utils._pytree.tree_flatten so the ordering (dict by
    # insertion order, list/tuple by position) matches the exporter exactly.
    if isinstance(x, torch.Tensor):
        return [x]
    try:
        from torch.utils._pytree import tree_flatten

        leaves, _ = tree_flatten(x)
        return list(leaves)
    except Exception:
        # a container pytree does not understand (custom object) - keep it as a
        # single opaque leaf so a length mismatch still surfaces as a failure
        return [x]


def _outputs_equal(a, b, atol=1e-3, rtol=1e-3):
    # the reference `ref` may be a dict / nested pytree while torch.export and
    # pnnx both hand back the flat leaf tensors in the same order; flatten both
    # sides and compare the leaves pairwise
    a_flat = _flatten_leaves(a)
    b_flat = _flatten_leaves(b)
    if len(a_flat) != len(b_flat):
        return False
    for x, y in zip(a_flat, b_flat):
        if x.dtype == torch.bool or y.dtype == torch.bool:
            if not torch.equal(x, y):
                return False
        elif not torch.allclose(x, y, atol=atol, rtol=rtol):
            return False
    return True


def _outputs_shape_equal(a, b):
    # shape + dtype only; for outputs with unspecified values (e.g. uninitialized
    # new_empty buffers) the torchscript path only compares shapes, mirror it;
    # flatten both sides first, see _outputs_equal
    a_flat = _flatten_leaves(a)
    b_flat = _flatten_leaves(b)
    if len(a_flat) != len(b_flat):
        return False
    return all(x.shape == y.shape and x.dtype == y.dtype for x, y in zip(a_flat, b_flat))


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
    # dict/list/tuple inputs (possibly nested) expand into multiple user_inputs
    # in torch.export; flatten them recursively in pytree leaf order so the flat
    # sequence maps 1:1 to the exported user_input specs / inputshapes
    return _flatten_leaves(args)


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
    except Exception as e:
        # torch.export does not support the model (dynamic shapes/control flow);
        # only a deliberate export-skip expectation keeps this green; the error
        # text is forwarded so a pinned diagnostic can be white-box validated
        return _finalize_pt2(tag, None, str(e))

    if not _convert_pnnx(pt2_path, _inputshapes_with_dtype(args, inputshapes)):
        return _finalize_pt2(tag, False)

    try:
        mod_pnnx = _load_pnnx_module(tag)
        out = mod_pnnx.test_inference()
    except Exception:
        # the generated pnnx python cannot be imported or run: that is a real
        # conversion/generation regression, report it as a failure instead of
        # silently skipping (None is reserved for torch.export incompatibility)
        return _finalize_pt2(tag, False)

    if shape_only:
        return _finalize_pt2(tag, _outputs_shape_equal(ref, out))
    return _finalize_pt2(tag, _outputs_equal(ref, out))


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
        # generated ncnn module cannot be imported/run: a real regression
        return False

    return _outputs_equal(ref, out, atol, rtol)
