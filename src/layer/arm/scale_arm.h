// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SCALE_ARM_H
#define LAYER_SCALE_ARM_H

#include "scale.h"

namespace ncnn {

class Scale_arm : public Scale
{
public:
    Scale_arm();

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;

protected:
#if NCNN_ARM82
    int forward_inplace_fp16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SCALE_ARM_H
