// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SCALE_MIPS_H
#define LAYER_SCALE_MIPS_H

#include "scale.h"

namespace ncnn {

class Scale_mips : public Scale
{
public:
    Scale_mips();

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SCALE_MIPS_H
