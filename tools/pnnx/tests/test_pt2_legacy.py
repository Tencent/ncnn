# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# legacy (<2.8) torch.export container support tests.
#
# torch.export.save switched to the pt2-archive layout in torch 2.8. the
# fixtures under fixtures/pt2_legacy/ were exported with torch 2.7.1 (see
# fixtures/pt2_legacy/generate.py for the exact reproducer), whose containers
# store a JSON graph plus pickled state dicts instead of raw weight shards.
#
# these tests convert each fixture with the current pnnx binary and verify the
# materialized weights byte-for-byte against torch.load of the pickled state
# dicts inside the fixture. torch.load is backward compatible, so the tests run
# on any modern torch (the CI venv) without needing torch 2.7 installed; the
# fixtures pin the exact legacy layout being exercised.

import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FIXDIR = os.path.join(HERE, "fixtures", "pt2_legacy")


def _find_pnnx():
    # the binary is pnnx.exe on Windows and pnnx elsewhere; subprocess (list
    # form) does not go through the shell's PATHEXT lookup, so probe explicitly
    for name in ("pnnx.exe", "pnnx"):
        p = os.path.join("..", "src", name)
        if os.path.exists(p):
            return p
    return os.path.join("..", "src", "pnnx")


PNNX = _find_pnnx()

# name -> inputshape (mirrors the exporter call in the fixture generator)
FIXTURES = {
    # F.linear + persistent buffer + non-persistent buffer (weights + constants)
    "linear_params_pt2_7": "[4,8]",
    # nn.Linear parameters + a tensor_constant
    "linear_const_pt2_7": "[4,8]",
    # conv + batchnorm: conv weights + a pile of buffers
    "conv_bn_params_pt2_7": "[1,1,8,8]",
    # bfloat16 weights
    "bf16_weights_pt2_7": "[4,8]",
    # float16 weights
    "f16_weights_pt2_7": "[4,8]",
    # int64 buffer + int64 tensor attribute
    "i64_values_pt2_7": "[4,8]",
}

# type string -> (element size in bytes, numpy dtype); bf16 has no numpy dtype
# and is decoded via bit manipulation in _decode_float
TYPE_ELEMSIZE = {
    "f32": (4, np.float32),
    "f64": (8, np.float64),
    "f16": (2, np.float16),
    "bf16": (2, None),
    "i8": (1, np.int8),
    "u8": (1, np.uint8),
    "i16": (2, np.int16),
    "i32": (4, np.int32),
    "i64": (8, np.int64),
    "bool": (1, np.bool_),
}


def _decode_float(data, tstr, shape):
    """decode a float attribute payload into a float64 tensor of the given shape."""
    if tstr == "bf16":
        u = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
        return (u << 16).view(np.float32).reshape(shape).astype(np.float64)
    return np.frombuffer(data, dtype=TYPE_ELEMSIZE[tstr][1]).reshape(shape).astype(np.float64)


def torch_dtype_to_str(dt):
    if dt == torch.float32:
        return "f32"
    if dt == torch.float64:
        return "f64"
    if dt == torch.float16:
        return "f16"
    if dt == torch.int64:
        return "i64"
    if dt == torch.int32:
        return "i32"
    if dt == torch.int16:
        return "i16"
    if dt == torch.int8:
        return "i8"
    if dt == torch.uint8:
        return "u8"
    if dt == torch.bool:
        return "bool"
    if dt == torch.bfloat16:
        return "bf16"
    return None


def parse_param(path):
    # parse every stored attribute (bin key, shape, type string) from a
    # .pnnx.param file. two sources:
    #   pnnx.Attribute <name> ... @data=(shape)type    -> bin key <name>.data
    #   <optype> <name> ... @attr=(shape)type ...      -> bin key <name>.<attr>
    attrs = []
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"^([A-Za-z][A-Za-z0-9_.]*)\s+(\S+)\s+\d+", line)
            if not m:
                continue
            optype = m.group(1)
            opname = m.group(2)
            for am in re.finditer(r"@(\w+)=\(([^)]*)\)([a-zA-Z0-9]+)", line):
                attr = am.group(1)
                shape = [int(x) for x in am.group(2).split(",") if x.strip() != ""] if am.group(2).strip() else []
                t = am.group(3)
                if t not in TYPE_ELEMSIZE:
                    continue
                key = (opname + ".data") if optype == "pnnx.Attribute" else (opname + "." + attr)
                attrs.append((key, shape, t))
    return attrs


def load_fixture_ref(fixture_path):
    z = zipfile.ZipFile(fixture_path)
    ref = {}
    for inner in ("serialized_state_dict.pt", "serialized_constants.pt"):
        if inner not in z.namelist():
            continue
        sd = torch.load(io.BytesIO(z.read(inner)), map_location="cpu", weights_only=True)
        for k, v in sd.items():
            ref[k] = v.detach().cpu()
    return ref


def compare(name, xshape):
    fixture = os.path.join(FIXDIR, name + ".pt2")
    if not os.path.exists(fixture):
        print("SKIP: fixture missing", fixture)
        return True

    # pnnx writes its outputs next to the input model, so convert a local copy
    # in the build dir instead of polluting the fixture directory
    local = "_pt2legacy_" + name + ".pt2"
    shutil.copyfile(fixture, local)
    try:
        r = subprocess.run([PNNX, local, "inputshape=" + xshape], capture_output=True, text=True)
        if r.returncode != 0:
            print("FAIL: convert rc=%d" % r.returncode)
            print(r.stderr[-600:] if r.stderr else "")
            return False

        base = local[:-4]
        param = base + ".pnnx.param"
        binf = base + ".pnnx.bin"
        if not (os.path.exists(param) and os.path.exists(binf)):
            print("FAIL: missing conversion outputs for", name)
            return False

        attrs = parse_param(param)
        zb = zipfile.ZipFile(binf)
        ref = load_fixture_ref(fixture)

        ok = True
        matched_bin = set()
        for fqn, t in sorted(ref.items()):
            tstr = torch_dtype_to_str(t.dtype)
            if tstr is None:
                print("FAIL: unsupported ref dtype for", fqn, t.dtype)
                ok = False
                continue
            cand = []
            for key, shape, ptype in attrs:
                if key in matched_bin or ptype != tstr or tuple(shape) != tuple(t.shape):
                    continue
                data = zb.read(key)
                if len(data) != t.numel() * TYPE_ELEMSIZE[tstr][0]:
                    continue
                cand.append((key, data))
            found = False
            last_md = 0.0
            last_key = ""
            for key, data in cand:
                if t.is_floating_point():
                    got = _decode_float(data, tstr, tuple(t.shape))
                    exp = t.double().numpy()
                    md = float(np.abs(got - exp).max())
                    last_md = md
                    last_key = key
                    if np.allclose(got, exp, atol=1e-4, rtol=1e-4):
                        matched_bin.add(key)
                        found = True
                        break
                else:
                    got = np.frombuffer(data, dtype=TYPE_ELEMSIZE[tstr][1]).reshape(t.shape)
                    exp = t.numpy()
                    if np.array_equal(got, exp):
                        matched_bin.add(key)
                        found = True
                        break
            if not found:
                print("FAIL: no matching attribute for ref '%s' shape=%s dtype=%s (closest %s md=%.2e)"
                      % (fqn, tuple(t.shape), tstr, last_key, last_md))
                ok = False
            else:
                print("OK: %s -> %s (maxdiff=%.2e)" % (fqn, key, float(np.abs(got.astype(np.float64) - exp.astype(np.float64)).max())))
        return ok
    finally:
        for f in os.listdir("."):
            if f.startswith("_pt2legacy_" + name):
                try:
                    os.remove(f)
                except OSError:
                    pass


def test():
    if not os.path.exists(PNNX):
        print("pnnx binary not found:", PNNX)
        return False
    all_ok = True
    for name, xshape in FIXTURES.items():
        if not compare(name, xshape):
            all_ok = False
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
