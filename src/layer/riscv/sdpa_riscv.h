// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SDPA_RISCV_H
#define LAYER_SDPA_RISCV_H

#include "sdpa.h"

namespace ncnn {

class SDPA_riscv : public SDPA
{
public:
    SDPA_riscv();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    // ---- Fast paths added by this repo (entry branches of forward; fall through to the upstream GEMM implementation when not taken) ----
    // Decode (M=1) RVV fast path: per-position QK^T dot product + streaming P·V.
    // Takes over only for fp32/elempack=1/kv_cache/attn_mask/src_seqlen==1; returning -1 means falling back to the upstream implementation.
    int forward_rvv_decode(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    // SpacemiT K3 A100 IME2 (smt.vfwmadot) prefill acceleration
    // Takes over only for: fp32 input, elempack=1, kv_cache=1, attn_mask=1, src_seqlen>1
    int forward_ime2_prefill(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    int ime2_available(int num_threads) const;
    int use_ime2_sdpa; // runtime probe result (A100=1)
#endif

public:
    // Upstream GEMM-accelerated sub-layers (created in create_pipeline, driven by forward)
    Layer* qk_gemm;
    Layer* qkv_gemm;

    Layer* qk_softmax;
};

} // namespace ncnn

#endif // LAYER_SDPA_RISCV_H
