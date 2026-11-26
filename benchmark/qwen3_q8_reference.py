# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""Compare benchncnn_llm --qwen3-q8-0 logits with an independent FP32 reference."""

import argparse

import gguf
import numpy as np
import torch
from torch.nn import functional as F


# same token ids and prefill length as benchncnn_llm
TOKENS = [151644, 872, 198, 2028, 374, 264, 1296, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198, 40, 2776, 311, 1401, 279]
PREFILL = 16


def rms_norm(x, weight):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * weight


def rotary(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


def qwen3_logits(weights):
    keys, values = [None] * 28, [None] * 28
    embedding = weights["token_embd.weight"]
    head = weights.get("output.weight", embedding)
    for start in [0, *range(PREFILL, len(TOKENS))]:
        length = 1 if start else PREFILL
        x = embedding[TOKENS[start : start + length]]
        positions = torch.arange(start, start + length)
        angles = positions.float()[:, None] * 1000000 ** (-torch.arange(64).float() / 64)
        cos, sin = angles.cos(), angles.sin()
        mask = torch.arange(start + length)[None, :] <= positions[:, None]

        for index in range(28):
            prefix = f"blk.{index}."
            y = rms_norm(x, weights[prefix + "attn_norm.weight"])
            q = F.linear(y, weights[prefix + "attn_q.weight"]).reshape(length, 16, 128).transpose(0, 1)
            k = F.linear(y, weights[prefix + "attn_k.weight"]).reshape(length, 8, 128).transpose(0, 1)
            v = F.linear(y, weights[prefix + "attn_v.weight"]).reshape(length, 8, 128).transpose(0, 1)
            q = rotary(rms_norm(q, weights[prefix + "attn_q_norm.weight"]), cos, sin)
            k = rotary(rms_norm(k, weights[prefix + "attn_k_norm.weight"]), cos, sin)
            if start:
                k = torch.cat([keys[index], k], dim=1)
                v = torch.cat([values[index], v], dim=1)
            keys[index], values[index] = k, v

            attn = F.scaled_dot_product_attention(q, k.repeat_interleave(2, dim=0), v.repeat_interleave(2, dim=0), attn_mask=mask)
            x = x + F.linear(attn.transpose(0, 1).reshape(length, 2048), weights[prefix + "attn_output.weight"])
            y = rms_norm(x, weights[prefix + "ffn_norm.weight"])
            gate = F.silu(F.linear(y, weights[prefix + "ffn_gate.weight"]))
            up = F.linear(y, weights[prefix + "ffn_up.weight"])
            x = x + F.linear(gate * up, weights[prefix + "ffn_down.weight"])

        yield F.linear(rms_norm(x[-1], weights["output_norm.weight"]), head).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("logits")
    parser.add_argument("--threads", default=8, type=int)
    parser.add_argument("--max-relative-l2", required=True, type=float)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    reader = gguf.GGUFReader(args.gguf)
    weights = {tensor.name: torch.from_numpy(gguf.dequantize(tensor.data, tensor.tensor_type).copy()) for tensor in reader.tensors}
    actual = np.fromfile(args.logits, dtype=np.float32).reshape(len(TOKENS) - PREFILL + 1, 151936)
    passed = True
    with torch.inference_mode():
        for step, expected in enumerate(qwen3_logits(weights)):
            error = np.linalg.norm(actual[step] - expected) / np.linalg.norm(expected)
            actual_top = actual[step].argmax()
            expected_top = expected.argmax()
            print(f"step {step}: relative L2 {error:.8f}, top token {actual_top}/{expected_top}")
            passed = passed and error <= args.max_relative_l2 and actual_top == expected_top

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
