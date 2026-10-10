// Copyright 2019 Leo <leo@nullptr.com.cn>
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BIAS_MIPS_H
#define LAYER_BIAS_MIPS_H

#include "bias.h"

namespace ncnn {

class Bias_mips : public Bias
{
public:
    Bias_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_BIAS_MIPS_H
