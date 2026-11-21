// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SPLIT_POWERPC_H
#define LAYER_SPLIT_POWERPC_H

#include "split.h"

namespace ncnn {

class Split_powerpc : public Split
{
public:
    Split_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SPLIT_POWERPC_H
