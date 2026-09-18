// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "hardsigmoid_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#include "hardsigmoid_fp32.h"

#if NCNN_BF16
#include "hardsigmoid_bf16s.h"
#endif

HardSigmoid_x86::HardSigmoid_x86()
{
#if __SSE2__
    support_packing = true;
    support_any_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int HardSigmoid_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    hardsigmoid_fp32(bottom_top_blob, alpha, beta, lower, upper, opt);

    return 0;
}

#if NCNN_BF16
int HardSigmoid_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    hardsigmoid_bf16s(bottom_top_blob, alpha, beta, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
