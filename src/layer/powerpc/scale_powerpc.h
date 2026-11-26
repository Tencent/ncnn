// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SCALE_POWERPC_H
#define LAYER_SCALE_POWERPC_H

#include "scale.h"

namespace ncnn {

class Scale_powerpc : public Scale
{
public:
    Scale_powerpc();

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_SCALE_POWERPC_H
