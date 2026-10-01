// SpacemiT K3 IME2 SDPA acceleration
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

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    // SpacemiT K3 A100 IME2 (smt.vfwmadot) 预填充加速
    // 仅在: fp32 输入、elempack=1、kv_cache=1、attn_mask=1、src_seqlen>1 时接管
    int forward_ime2_prefill(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    int ime2_available() const;
    int use_ime2_sdpa; // 运行期探测结果（A100=1）
#endif
};

} // namespace ncnn

#endif // LAYER_SDPA_RISCV_H
