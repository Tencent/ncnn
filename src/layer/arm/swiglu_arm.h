// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWIGLU_ARM_H
#define LAYER_SWIGLU_ARM_H

#include "swiglu.h"

namespace ncnn {

class SwiGLU_arm : public SwiGLU
{
public:
    SwiGLU_arm();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int forward_fp32(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const;
#if NCNN_ARM82
    int forward_fp16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_bf16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SWIGLU_ARM_H
