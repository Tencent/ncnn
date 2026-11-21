// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PACKING_POWERPC_H
#define LAYER_PACKING_POWERPC_H

#include "packing.h"

namespace ncnn {

class Packing_powerpc : public Packing
{
public:
    Packing_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PACKING_POWERPC_H
