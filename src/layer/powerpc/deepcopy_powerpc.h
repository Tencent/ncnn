// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEEPCOPY_POWERPC_H
#define LAYER_DEEPCOPY_POWERPC_H

#include "deepcopy.h"

namespace ncnn {

class DeepCopy_powerpc : public DeepCopy
{
public:
    DeepCopy_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DEEPCOPY_POWERPC_H
