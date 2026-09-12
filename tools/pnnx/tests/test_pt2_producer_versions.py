# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# cross-producer coverage: fixtures/pt2_producer/*.pt2 are real
# ExportedProgram archives written by several torch releases at and after the
# 2.8 pt2-archive switch (see that directory's generate.py). The
# loader is producer-agnostic: the archive layout and the raw-payload schema
# field set differ between releases, so each fixture must convert and produce
# numerically identical output. The reference is the eager model built by
# generate.py, whose parameters are deterministic by construction, so it can be
# recomputed with whatever torch the CI has installed.

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile


def _find_pnnx():
    # runner cwd is tools/pnnx/build/tests (or tests/ncnn); the binary lives in
    # the build tree's src dir next to it
    for rel in (os.path.join("..", "src", "pnnx.exe"),
                os.path.join("..", "..", "src", "pnnx.exe"),
                os.path.join("..", "src", "pnnx"),
                os.path.join("..", "..", "src", "pnnx")):
        p = os.path.abspath(rel)
        if os.path.isfile(p):
            return p
    return None


def _case(name, ok, detail):
    print("[producer] %-14s %s" % (name, "PASS" if ok else "FAIL"))
    if not ok:
        print(detail)
    return ok


def _load_generator(fixtures):
    spec = importlib.util.spec_from_file_location("pt2_producer_generate", os.path.join(fixtures, "generate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test():
    pnnx = _find_pnnx()
    if pnnx is None:
        print("[producer] pnnx binary not found")
        return False

    try:
        import numpy as np
        import torch
    except Exception:
        print("[producer] torch not available, skip")
        return True

    fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "pt2_producer")
    names = sorted(n for n in os.listdir(fixtures) if n.startswith("producer_") and n.endswith(".pt2"))
    if not names:
        print("[producer] no fixtures, skip")
        return True

    gen = _load_generator(fixtures)
    model = gen.build_model()
    x = gen.example_input()
    with torch.no_grad():
        ref = model(x).detach().numpy()
    ishape = "[%d,%d,%d,%d]f32" % gen.INPUT_SHAPE

    results = []
    with tempfile.TemporaryDirectory() as workdir:
        sys.path.insert(0, workdir)
        for name in names:
            tag = name[len("producer_"):-len(".pt2")]
            pt2 = os.path.join(workdir, name)
            shutil.copy2(os.path.join(fixtures, name), pt2)
            base = os.path.join(workdir, "producer_%s" % tag)
            # forward-slash form for every path pnnx embeds in the generated py
            pt2_arg = pt2.replace("\\", "/")
            base_arg = base.replace("\\", "/")
            try:
                r = subprocess.run(
                    [pnnx, pt2_arg, "inputshape=%s" % ishape, "pnnxparam=%s.pnnx.param" % base_arg, "pnnxbin=%s.pnnx.bin" % base_arg],
                    cwd=workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                results.append(_case(tag, False, "(timeout)"))
                continue
            if r.returncode != 0:
                results.append(_case(tag, False, r.stdout.decode("utf-8", "replace")[-600:]))
                continue

            # a conversion that succeeds but emits a model that cannot be
            # imported or run is still a failure of this test
            try:
                pymod = "%s_pnnx.py" % base
                spec = importlib.util.spec_from_file_location("producer_%s" % tag, pymod)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                with torch.no_grad():
                    out = mod.Model()(x).detach().numpy()
            except Exception as e:
                results.append(_case(tag, False, "generated model failed: %r" % (e,)))
                continue

            results.append(_case(tag, np.allclose(out, ref, atol=1e-5), "maxdiff=%f" % np.abs(out - ref).max()))

    return all(results)


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
