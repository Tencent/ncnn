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

    // ---- 本仓库新增的快速路径（作为 forward 的入口分支；不命中时继续走上游 GEMM 实现）----
    // 解码（M=1）RVV 快速路径：QK^T 逐位置点积 + 流式 P·V。
    // 仅在 fp32/elempack=1/kv_cache/attn_mask/src_seqlen==1 时接管；返回 -1 表示回退上游实现。
    int forward_rvv_decode(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    // SpacemiT K3 A100 IME2 (smt.vfwmadot) 预填充加速
    // 仅在: fp32 输入、elempack=1、kv_cache=1、attn_mask=1、src_seqlen>1 时接管
    int forward_ime2_prefill(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    int ime2_available() const;
    int use_ime2_sdpa; // 运行期探测结果（A100=1）
#endif

public:
    // 上游的 GEMM 加速子层（create_pipeline 创建、forward 驱动）
    Layer* qk_gemm;
    Layer* qkv_gemm;

    Layer* qk_softmax;
};

} // namespace ncnn

#endif // LAYER_SDPA_RISCV_H
