// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWIGLU_MIPS_H
#define LAYER_SWIGLU_MIPS_H

#include "swiglu.h"

namespace ncnn {

class SwiGLU_mips : public SwiGLU
{
public:
    SwiGLU_mips();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int forward_fp32(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const;
#if NCNN_BF16
    int forward_bf16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SWIGLU_MIPS_H
