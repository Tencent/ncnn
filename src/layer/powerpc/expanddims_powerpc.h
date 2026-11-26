// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_EXPANDDIMS_POWERPC_H
#define LAYER_EXPANDDIMS_POWERPC_H

#include "expanddims.h"

namespace ncnn {

class ExpandDims_powerpc : public ExpandDims
{
public:
    ExpandDims_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_EXPANDDIMS_POWERPC_H
