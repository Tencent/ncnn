// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SELU_ARM_H
#define LAYER_SELU_ARM_H

#include "selu.h"

namespace ncnn {

class SELU_arm : public SELU
{
public:
    SELU_arm();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ARM82
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SELU_ARM_H
