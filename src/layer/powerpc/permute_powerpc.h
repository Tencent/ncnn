// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PERMUTE_POWERPC_H
#define LAYER_PERMUTE_POWERPC_H

#include "permute.h"

namespace ncnn {

class Permute_powerpc : public Permute
{
public:
    Permute_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PERMUTE_POWERPC_H
