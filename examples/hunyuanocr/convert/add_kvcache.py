"""Rewrite text_decoder.ncnn.param to use incremental KV-cache SDPA.

pnnx exports SDPA with 4 inputs (q,k,v,mask) -> 1 output.
ncnn's SDPA supports incremental KV cache: 6 inputs
(q,k,v,mask,cache_k,cache_v) -> 3 outputs (out,out_k,out_v) when param 7=1.

Usage:
    python add_kvcache.py text_decoder.ncnn.param
A backup <param>.nokv is written the first time.
"""
import sys
import os
import math

HEAD_DIM = 128
SCALE = "%g" % (1.0 / math.sqrt(HEAD_DIM))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "text_decoder.ncnn.param"
    raw = open(path, encoding="utf-8").read()
    lines = raw.split("\n")
    if lines[0].strip() != "7767517":
        raise SystemExit("not an ncnn param file: " + path)

    bak = path + ".nokv"
    if not os.path.exists(bak):
        open(bak, "w", encoding="utf-8").write(raw)
        print("[backup]", bak)

    header = lines[1].split()
    orig_layers, orig_blobs = int(header[0]), int(header[1])
    body = lines[2:]

    # locate the Input in3 line to insert kv_cache Input after it
    in3_idx = None
    for i, l in enumerate(body):
        f = l.split()
        if len(f) >= 5 and f[0] == "Input" and f[-1] == "in3":
            in3_idx = i
            break
    if in3_idx is None:
        raise SystemExit("could not find `Input in3` line")

    sdpa_count = sum(1 for l in body if l.startswith("SDPA ") or l.startswith("MultiHeadAttention "))

    # insert one Input layer for kv caches (produces cache_k0..cache_kN-1, cache_v0..cache_vN-1)
    kvcache_input = (
        f"Input kv_cache 0 {2*sdpa_count} "
        + " ".join(f"cache_k{i}" for i in range(sdpa_count))
        + " "
        + " ".join(f"cache_v{i}" for i in range(sdpa_count))
    )

    sdpa_i = 0
    out_body = []
    for l in body:
        f = l.split()
        if len(f) >= 2 and f[0] in ("SDPA", "MultiHeadAttention"):
            # original: SDPA name 4 1 q k v mask out
            # rewrite:  SDPA name 6 3 q k v mask cache_ki cache_vi out_i out_ki out_vi 5=1 6=SCALE 7=1
            name = f[1]
            # find q,k,v,mask,out from original split
            # format: Layer Type  name  nin nout  in0..inN  out0..outN  [params]
            nin = int(f[2])
            nout = int(f[3])
            inputs = f[4: 4 + nin]
            outputs = f[4 + nin: 4 + nin + nout]
            q, k, v, mask = inputs[:4]
            out = outputs[0]
            ck = f"cache_k{sdpa_i}"
            cv = f"cache_v{sdpa_i}"
            ok = f"out_k{sdpa_i}"
            ov = f"out_v{sdpa_i}"
            l = (
                f"SDPA {name} 6 3 {q} {k} {v} {mask} {ck} {cv} "
                f"{out} {ok} {ov} 5=1 6={SCALE} 7=1"
            )
            sdpa_i += 1
        out_body.append(l)

    out_body.insert(in3_idx + 1, kvcache_input)

    # recount layers and blobs
    layer_lines = [l for l in out_body if l.strip()]
    new_layers = len(layer_lines)

    def noutputs(l):
        f = l.split()
        return int(f[3]) if len(f) >= 4 else 0

    new_blobs = sum(noutputs(l) for l in layer_lines)

    out_lines = ["7767517", f"{new_layers} {new_blobs}"] + out_body
    open(path, "w", encoding="utf-8").write("\n".join(out_lines))
    print(f"[done] {path}: {orig_layers}->{new_layers} layers, "
          f"{orig_blobs}->{new_blobs} blobs, {sdpa_i} SDPA ops patched")


if __name__ == "__main__":
    main()
