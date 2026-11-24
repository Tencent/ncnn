// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SPP_POWERPC_H
#define LAYER_SPP_POWERPC_H

#include "spp.h"

namespace ncnn {

class SPP_powerpc : public SPP
{
public:
    SPP_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SPP_POWERPC_H
