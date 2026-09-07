// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "eltwise_bf16s.h"

void eltwise_bf16s_avxneconvert(const std::vector<Mat>& bottom_blobs, Mat& top_blob, int op_type, const Mat& coeffs, const Option& opt)
{
    eltwise_bf16s(bottom_blobs, top_blob, op_type, coeffs, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
