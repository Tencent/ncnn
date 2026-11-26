// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONCAT_POWERPC_H
#define LAYER_CONCAT_POWERPC_H

#include "concat.h"

namespace ncnn {

class Concat_powerpc : public Concat
{
public:
    Concat_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CONCAT_POWERPC_H
