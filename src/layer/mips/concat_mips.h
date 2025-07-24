// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONCAT_MIPS_H
#define LAYER_CONCAT_MIPS_H

#include "concat.h"

namespace ncnn {

class Concat_mips : public Concat
{
public:
    Concat_mips();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CONCAT_MIPS_H
