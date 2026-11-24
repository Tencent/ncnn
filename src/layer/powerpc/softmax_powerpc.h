// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SOFTMAX_POWERPC_H
#define LAYER_SOFTMAX_POWERPC_H

#include "softmax.h"

namespace ncnn {

class Softmax_powerpc : public Softmax
{
public:
    Softmax_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SOFTMAX_POWERPC_H
