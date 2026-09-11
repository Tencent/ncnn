// Copyright 2026 darkavatar23 <matteo.forzan95@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SCALE_RISCV_H
#define LAYER_SCALE_RISCV_H

#include "scale.h"

namespace ncnn {

class Scale_riscv : public Scale
{
public:
    Scale_riscv();

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SCALE_RISCV_H
