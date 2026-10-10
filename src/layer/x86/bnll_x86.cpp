// Copyright 2022 JasonZhang892 <zqhy_0929@163.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "bnll_x86.h"

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

#include "bnll_fp32.h"

#if NCNN_BF16
#include "bnll_bf16s.h"
#endif

BNLL_x86::BNLL_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int BNLL_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    bnll_fp32(bottom_top_blob, opt);

    return 0;
}

#if NCNN_BF16
int BNLL_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    bnll_bf16s(bottom_top_blob, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
