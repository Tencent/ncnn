// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "selu_x86.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#include "selu_fp32.h"

#if NCNN_BF16
#include "selu_bf16s.h"
#endif

SELU_x86::SELU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int SELU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    selu_fp32(bottom_top_blob, alpha, lambda, opt);

    return 0;
}

#if NCNN_BF16
int SELU_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    float alphaxlambda = alpha * lambda;
    selu_bf16s(bottom_top_blob, alphaxlambda, lambda, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
