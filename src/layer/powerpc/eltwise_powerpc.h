// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELTWISE_POWERPC_H
#define LAYER_ELTWISE_POWERPC_H

#include "eltwise.h"

namespace ncnn {

class Eltwise_powerpc : public Eltwise
{
public:
    Eltwise_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ELTWISE_POWERPC_H
