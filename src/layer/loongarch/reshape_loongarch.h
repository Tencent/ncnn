// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RESHAPE_LOONGARCH_H
#define LAYER_RESHAPE_LOONGARCH_H

#include "reshape.h"

namespace ncnn {

class Reshape_loongarch : public Reshape
{
public:
    Reshape_loongarch();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_RESHAPE_LOONGARCH_H
