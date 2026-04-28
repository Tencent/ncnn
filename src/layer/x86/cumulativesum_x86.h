// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CUMULATIVESUM_X86_H
#define LAYER_CUMULATIVESUM_X86_H

#include "cumulativesum.h"

namespace ncnn {

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__
void prefix_sum_row_avx2(float* ptr, int w);
void cumulative_sum_add_avx2(const float* ptr, float* outptr, int size);
#endif

class CumulativeSum_x86 : public CumulativeSum
{
public:
    CumulativeSum_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CUMULATIVESUM_X86_H
