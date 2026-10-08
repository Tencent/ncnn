// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lrn_x86.h"

#include "cpu.h"

#if __AVX__
#include "avx_mathfun.h"
#endif // __AVX__

namespace ncnn {

#include "lrn_fp32.h"

int LRN_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    return lrn_fp32(bottom_top_blob, region_type, local_size, alpha, beta, bias, opt);
}

} // namespace ncnn
