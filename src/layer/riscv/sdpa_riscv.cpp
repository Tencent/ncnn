// SpacemiT K3 IME2 SDPA acceleration
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_riscv.h"

#include <stdlib.h>

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
#include <math.h>
#endif

namespace ncnn {

SDPA_riscv::SDPA_riscv()
{
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    use_ime2_sdpa = 0;
#endif
}

int SDPA_riscv::create_pipeline(const Option& opt)
{
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    // 探测在 zfh 变体 TU 里做（那里有 smt.vfwmadot 的汇编）
    use_ime2_sdpa = ime2_available();
    // 调试开关：NCNN_SDPA_IME2=0 可强制关闭（定位问题用）
    const char* env = getenv("NCNN_SDPA_IME2");
    if (env && env[0] == '0' && env[1] == '\0')
        use_ime2_sdpa = 0;
#endif

    return SDPA::create_pipeline(opt);
}

int SDPA_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    if (use_ime2_sdpa && kv_cache && attn_mask && !int8_scale_term)
    {
        const Mat& query = bottom_blobs[0];
        // 只接管 fp32/elempack=1 的预填充（q 序列长度 > 1）；解码(L=1)走原路径
        if (query.elembits() == 32 && query.elempack == 1 && query.h > 1)
            return forward_ime2_prefill(bottom_blobs, top_blobs, opt);
    }
#endif

    return SDPA::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
