// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RESHAPE_POWERPC_H
#define LAYER_RESHAPE_POWERPC_H

#include "reshape.h"

namespace ncnn {

class Reshape_powerpc : public Reshape
{
public:
    Reshape_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_RESHAPE_POWERPC_H
