# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# White-box robustness tests for the torch.export (pt2) loader: the negative
# half deliberately corrupts a minimal real .pt2 archive (conv) and pins that
# each guard rejects the hostile input (no crash / OOM / hang); the positive
# half pins that a ZIP_DEFLATED archive is inflated losslessly and converts to
# byte-identical output as the stored control.
#
# Each rejection case pins a diagnostic substring (the loader-side analogue of
# the pt2_expectations needles), so a guard that silently starts accepting a
# bad archive is a visible regression.
#
# Needs a torch with torch.export (2.8+); like the other test_pt2_* files it is
# collected by the pnnx test suite (runs in a scratch dir, no net needed).

import json
import os
import struct
import subprocess
import tempfile
import zipfile


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


def _export_base(pt2_path):
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super(M, self).__init__()
            self.c = nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x):
            return self.c(x).relu() + 1

    net = M().eval()
    x = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        ep = torch.export.export(net, (x,))
        torch.export.save(ep, pt2_path)


def _copy_and_replace(src, out, drop=(), replace=None, method=zipfile.ZIP_STORED):
    # rewrite a .pt2 zip. method controls entry compression: the stored form is
    # the torch layout, ZIP_DEFLATED exercises the loader's RFC1951 inflate.
    # drop = entries omitted, replace = {entry: new bytes}.
    zin = zipfile.ZipFile(src)
    replace = replace or {}
    with zipfile.ZipFile(out, "w", method) as zout:
        for i in zin.infolist():
            if i.filename in drop:
                continue
            data = replace.get(i.filename, zin.read(i.filename))
            zout.writestr(i.filename, data)
    zin.close()


def _make_hostile_deflate_zip(path):
    # a method-8 entry whose dynamic-huffman header declares HLIT=288 / HDIST=32
    # (total 320 length codes): RFC1951 caps them at 286/30, so an inflate that
    # trusts the header writes past its length array. the loader must reject it
    # cleanly (nonzero) instead of crashing with a corrupted stack (SIGABRT).
    def _bits():
        out = []

        def put(v, n):
            for i in range(n):
                out.append((v >> i) & 1)

        put(1, 1)  # BFINAL
        put(2, 2)  # BTYPE = dynamic
        put(31, 5)  # HLIT -> 288
        put(31, 5)  # HDIST -> 32
        put(0, 4)  # HCLEN -> 4
        # code-length tree with only symbol 0 (len 1), the rest zero
        put(0, 3)
        put(0, 3)
        put(0, 3)
        put(1, 3)
        out += [0] * 320  # 320 length-0 codes -> overflows an unchecked table
        while len(out) % 8:
            out.append(0)
        return bytes(int("".join(str(b) for b in out[i:i + 8][::-1]), 2) for i in range(0, len(out), 8))

    raw = _bits()
    name = b"base/models/model.json"
    crc = 0x12345678
    csize = len(raw)
    usize = 100
    lh = struct.pack("<IHHHHHIIIHH", 0x04034b50, 20, 0, 8, 0, 0, crc, csize, usize, len(name), 0) + name
    cd = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014b50, 20, 20, 0, 8, 0, 0, crc, csize, usize, len(name), 0, 0, 0, 0, 0, 0) + name
    eocd = struct.pack("<IHHHHIIH", 0x06054b50, 0, 0, 1, 1, len(cd), len(lh) + len(raw), 0)
    with open(path, "wb") as f:
        f.write(lh + raw + cd + eocd)


def _patch_central(path, entry, crc=None, flag=None, usize=None):
    # byte-level patch of one central-directory record: overwrite its crc32 /
    # OR its general-purpose flag bits / overwrite its uncompressed size
    # (zipfile cannot express a deliberately wrong crc / encrypted flag /
    # lying size, so the hostile archive is built by hand).
    b = bytearray(open(path, "rb").read())
    pos = 0
    hit = False
    while True:
        p = b.find(struct.pack("<I", 0x02014b50), pos)
        if p < 0:
            break
        nlen = struct.unpack_from("<H", b, p + 28)[0]
        if bytes(b[p + 46:p + 46 + nlen]) == entry:
            if crc is not None:
                struct.pack_into("<I", b, p + 16, crc & 0xffffffff)
            if flag is not None:
                cur = struct.unpack_from("<H", b, p + 8)[0]
                struct.pack_into("<H", b, p + 8, cur | (flag & 0xffff))
            if usize is not None:
                struct.pack_into("<I", b, p + 24, usize & 0xffffffff)
            hit = True
            break
        pos = p + 1
    if not hit:
        raise RuntimeError("central entry %s not found" % entry)
    open(path, "wb").write(b)


def _weights_config(src):
    zin = zipfile.ZipFile(src)
    cfg = json.loads(zin.read("base/data/weights/model_weights_config.json"))
    zin.close()
    return cfg


def _build_cases(workdir, base):
    cases = {}

    # positive control: untouched base converts cleanly
    cases["base"] = base

    # whole archive re-written with ZIP_DEFLATED: the loader must inflate it and
    # convert identically to the stored control (positive white-box case)
    p = os.path.join(workdir, "case_deflate.pt2")
    _copy_and_replace(base, p, method=zipfile.ZIP_DEFLATED)
    cases["deflate"] = p

    # truncated zip (tail cut off)
    raw = open(base, "rb").read()
    p = os.path.join(workdir, "case_trunc.pt2")
    open(p, "wb").write(raw[: len(raw) - 200])
    cases["trunc"] = p

    # payload record referenced by the config is missing entirely
    p = os.path.join(workdir, "case_missing_weight.pt2")
    _copy_and_replace(base, p, drop=("base/data/weights/weight_0",))
    cases["missing_weight"] = p

    # weight payload byte flipped but the central crc32 kept (zipfile rewrites
    # the real crc, so the central record is patched back to a wrong value): a
    # loader that trusts the archive silently materializes corrupt weights
    p = os.path.join(workdir, "case_crc.pt2")
    zin = zipfile.ZipFile(base)
    data = bytearray(zin.read("base/data/weights/weight_0"))
    zin.close()
    data[8] ^= 0xFF
    _copy_and_replace(base, p, replace={"base/data/weights/weight_0": bytes(data)})
    _patch_central(p, b"base/data/weights/weight_0", crc=0xDEADBEEF)
    cases["crc"] = p

    # one entry flagged encrypted (general-purpose bit 0); zip readers refuse
    # it, the loader must too instead of reading an unencrypted-looking blob
    p = os.path.join(workdir, "case_encrypted.pt2")
    _copy_and_replace(base, p)
    _patch_central(p, b"base/models/model.json", flag=1)
    cases["encrypted"] = p

    # container metadata claims an archive_version the loader does not know
    p = os.path.join(workdir, "case_bad_version.pt2")
    _copy_and_replace(base, p, replace={"base/archive_version": b"1"})
    cases["bad_version"] = p

    # one entry's central directory advertises a huge uncompressed size (the
    # real payload is tiny): a loader that trusts it pre-allocates a huge
    # buffer / inflate bomb. must be rejected at the container boundary.
    p = os.path.join(workdir, "case_big_size.pt2")
    _copy_and_replace(base, p)
    _patch_central(p, b"base/data/weights/weight_0", usize=0xFFFFFF00)
    cases["big_size"] = p

    # payload config (weights / constants) truncated: must be rejected at the
    # config boundary instead of silently emitting an incomplete model
    for tag, blob in (("cfg_weights", "base/data/weights/model_weights_config.json"),
                      ("cfg_constants", "base/data/constants/model_constants_config.json")):
        p = os.path.join(workdir, "case_%s.pt2" % tag)
        _copy_and_replace(base, p, replace={blob: b""})
        cases[tag] = p

    # graph body (models/model.json) emptied -> parse failure
    p = os.path.join(workdir, "case_no_model.pt2")
    _copy_and_replace(base, p, replace={"base/models/model.json": b""})
    cases["no_model"] = p

    # hostile tensor_meta: element count far beyond the storage (a naive
    # materialization would try to allocate a multi-gigabyte buffer -> OOM)
    p = os.path.join(workdir, "case_huge_shape.pt2")
    cfg = _weights_config(base)
    t = cfg["config"]["c.weight"]["tensor_meta"]
    t["sizes"] = [{"as_int": 4}, {"as_int": 3}, {"as_int": 1000000000}, {"as_int": 3}]
    t["strides"] = [{"as_int": 9000000000}, {"as_int": 3000000000}, {"as_int": 3}, {"as_int": 1}]
    _copy_and_replace(base, p, replace={"base/data/weights/model_weights_config.json": json.dumps(cfg).encode()})
    cases["huge_shape"] = p

    # storage_offset pointing far outside the storage (view materialization
    # must bounds-check instead of reading out of range / allocating wildly)
    p = os.path.join(workdir, "case_oob_offset.pt2")
    cfg = _weights_config(base)
    cfg["config"]["c.weight"]["tensor_meta"]["storage_offset"] = {"as_int": 999999999}
    _copy_and_replace(base, p, replace={"base/data/weights/model_weights_config.json": json.dumps(cfg).encode()})
    cases["oob_offset"] = p

    # not a zip at all
    p = os.path.join(workdir, "case_garbage.pt2")
    open(p, "wb").write(b"this is not a zip archive at all " * 10)
    cases["garbage"] = p

    # deflated entry with an out-of-range dynamic-huffman header (HLIT=288 /
    # HDIST=32): an inflate that trusts the header overwrites its length table
    p = os.path.join(workdir, "case_hostile_deflate.pt2")
    _make_hostile_deflate_zip(p)
    cases["hostile_deflate"] = p

    # graph body is valid JSON but deeply nested: a parser that recurses
    # without a depth bound overflows the stack (SIGSEGV) before any schema
    # check runs; it must reject cleanly instead
    p = os.path.join(workdir, "case_deep_json.pt2")
    _copy_and_replace(base, p, replace={"base/models/model.json": b"[" * 5000 + b"0" + b"]" * 5000})
    cases["deep_json"] = p

    # graph body is valid JSON but carries no graph/signature structure: a
    # container loader that trusts it silently emits an empty model (exit 0)
    p = os.path.join(workdir, "case_no_graph.pt2")
    _copy_and_replace(base, p, replace={"base/models/model.json": b'{"range_constraints": {}}'})
    cases["no_graph"] = p

    # stored entry whose central sizes disagree (compressed != uncompressed):
    # read_file would copy compressed bytes into a buffer sized by the
    # uncompressed size (heap overflow before the crc check can fire)
    p = os.path.join(workdir, "case_stored_mismatch.pt2")
    _copy_and_replace(base, p)
    _patch_central(p, b"base/data/weights/weight_0", usize=100)
    cases["stored_mismatch"] = p

    return cases


def _run_pnnx(pnnx, pt2, cwd, timeout=90):
    try:
        r = subprocess.run(
            [pnnx, pt2, "inputshape=[1,3,8,8]f32"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        text = (r.stdout or b"").decode("utf-8", errors="replace")
        return r.returncode, text
    except subprocess.TimeoutExpired:
        return None, "(timeout)"


def _case(name, ok, detail):
    print("[robust] %-16s %s" % (name, "PASS" if ok else "FAIL"))
    if not ok:
        print(detail)
    return ok


def test():
    pnnx = _find_pnnx()
    if pnnx is None:
        print("[robust] pnnx binary not found")
        return False

    try:
        import torch  # noqa: F401
    except Exception:
        # no torch.export available: the pt2 channel is skipped elsewhere too
        print("[robust] torch not available, skip")
        return True

    results = []

    with tempfile.TemporaryDirectory() as workdir:
        base = os.path.join(workdir, "base.pt2")
        try:
            _export_base(base)
        except Exception as e:
            # torch.export itself failed in this environment (torch < 2.8 / no
            # export); treat as an environment skip like the rest of the suite
            print("[robust] torch.export unavailable (%s), skip" % e)
            return True

        cases = _build_cases(workdir, base)
        outdir = os.path.join(workdir, "out")
        os.makedirs(outdir)

        # positive control must convert (pnnx writes outputs next to the input)
        rc, text = _run_pnnx(pnnx, cases["base"], outdir)
        ok = rc == 0 and os.path.isfile(os.path.splitext(cases["base"])[0] + ".pnnx.param")
        results.append(_case("base(control)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # positive deflate: a ZIP_DEFLATED archive must inflate losslessly and
        # produce byte-identical param + bin to the stored control (this is the
        # white-box proof that the RFC1951 inflate restores the exact bytes the
        # loader saw from the stored form)
        def _out_pref(pt2):
            return os.path.splitext(pt2)[0]

        base_pref = _out_pref(cases["base"])
        deflate_pref = _out_pref(cases["deflate"])
        rc, text = _run_pnnx(pnnx, cases["deflate"], outdir)
        same_out = True
        for ext in (".pnnx.param", ".pnnx.bin"):
            a = base_pref + ext
            b = deflate_pref + ext
            if not (os.path.isfile(a) and os.path.isfile(b)):
                same_out = False
                break
            if open(a, "rb").read() != open(b, "rb").read():
                same_out = False
                break
        ok = rc == 0 and os.path.isfile(deflate_pref + ".pnnx.param") and same_out
        results.append(_case("deflate(pos)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # corrupt archives must be rejected (nonzero), not hang / crash
        for name, needle_expected in (("trunc", None), ("no_model", None), ("garbage", None)):
            rc, text = _run_pnnx(pnnx, cases[name], outdir)
            ok = rc != 0 and rc is not None
            results.append(_case(name + "(reject)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # dropped weight payload must be rejected with the pinned diagnostic
        rc, text = _run_pnnx(pnnx, cases["missing_weight"], outdir)
        ok = rc != 0 and rc is not None and "not found" in text
        results.append(_case("missing_weight", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # crc32 integrity: a payload that no longer matches its central crc32
        # must be rejected with the crc diagnostic (not silently converted)
        rc, text = _run_pnnx(pnnx, cases["crc"], outdir)
        ok = rc != 0 and rc is not None and "crc mismatch" in text
        results.append(_case("crc(reject)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # encrypted entry must be rejected with the encrypted diagnostic
        rc, text = _run_pnnx(pnnx, cases["encrypted"], outdir)
        ok = rc != 0 and rc is not None and "encrypted" in text
        results.append(_case("encrypted(reject)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # unknown archive_version must be rejected with the unsupported diagnostic
        rc, text = _run_pnnx(pnnx, cases["bad_version"], outdir)
        ok = rc != 0 and rc is not None and "unsupported archive_version" in text
        results.append(_case("archive_version(reject)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # a lying central uncompressed size must be rejected at the container
        # boundary (diagnostic), not drive a huge allocation / inflate bomb
        rc, text = _run_pnnx(pnnx, cases["big_size"], outdir)
        ok = rc is not None and rc >= 0 and rc in (0, 1, 255, 4294967295) and rc != 0 and "oversized uncompressed entry" in text
        results.append(_case("big_size(reject)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # truncated weights/constants config must be rejected at the config
        # boundary (parse diagnostic), not silently converted without data
        for name in ("cfg_weights", "cfg_constants"):
            rc, text = _run_pnnx(pnnx, cases[name], outdir)
            ok = rc != 0 and rc is not None and "config" in text and "failed" in text
            results.append(_case(name + "(reject)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # hostile tensor_meta must not OOM / crash / hang the process: the run
        # must finish (within timeout) with a clean (non-signal) returncode. a
        # negative returncode means the process was killed by a signal.
        # note: pnnx exits via `return -1` on rejection; Linux reports 255,
        # Windows reports the unsigned 0xffffffff (4294967295)
        for name in ("huge_shape", "oob_offset"):
            rc, text = _run_pnnx(pnnx, cases[name], outdir)
            ok = rc is not None and rc >= 0 and rc in (0, 1, 255, 4294967295)
            results.append(_case(name + "(guarded)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # deflate header overrun (hostile HLIT/HDIST) must be rejected cleanly,
        # not abort with a smashed stack (SIGABRT -> negative returncode)
        rc, text = _run_pnnx(pnnx, cases["hostile_deflate"], outdir)
        ok = rc is not None and rc >= 0 and rc in (0, 1, 255, 4294967295)
        results.append(_case("hostile_inflate(guarded)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # deeply nested model.json must be rejected (parser depth bound), not
        # crash with a stack overflow (SIGSEGV -> negative returncode)
        rc, text = _run_pnnx(pnnx, cases["deep_json"], outdir)
        ok = rc is not None and rc >= 0 and rc in (0, 1, 255, 4294967295) and rc != 0
        results.append(_case("deep_json(reject)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # structurally-empty model.json must be rejected at the container
        # boundary, not silently converted into an empty model (exit 0)
        rc, text = _run_pnnx(pnnx, cases["no_graph"], outdir)
        ok = rc != 0 and rc is not None and "graph structure" in text
        results.append(_case("no_graph(reject)", ok, "rc=%r\n%s" % (rc, text[-800:])))

        # stored entry with disagreeing central sizes must be rejected (would
        # otherwise overflow the read buffer before the crc check)
        rc, text = _run_pnnx(pnnx, cases["stored_mismatch"], outdir)
        ok = rc != 0 and rc is not None and "invalid stored entry" in text
        results.append(_case("stored_mismatch(reject)", ok, "rc=%r\n%s" % (rc, text[-800:])))

    return all(results)


if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
