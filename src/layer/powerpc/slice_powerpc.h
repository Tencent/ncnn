// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SLICE_POWERPC_H
#define LAYER_SLICE_POWERPC_H

#include "slice.h"

namespace ncnn {

class Slice_powerpc : public Slice
{
public:
    Slice_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SLICE_POWERPC_H
