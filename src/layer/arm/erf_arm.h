// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ERF_ARM_H
#define LAYER_ERF_ARM_H

#include "erf.h"

namespace ncnn {

class Erf_arm : public Erf
{
public:
    Erf_arm();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ARM82
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ERF_ARM_H
