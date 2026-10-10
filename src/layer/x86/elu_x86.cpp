// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "elu_x86.h"

#include "x86_activation.h"

#include "cpu.h"

namespace ncnn {

#include "elu_fp32.h"

#if NCNN_BF16
#include "elu_bf16s.h"
#endif

ELU_x86::ELU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int ELU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    elu_fp32(bottom_top_blob, alpha, opt);

    return 0;
}

#if NCNN_BF16
int ELU_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    elu_bf16s(bottom_top_blob, alpha, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
