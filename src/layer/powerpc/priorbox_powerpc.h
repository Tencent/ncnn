// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PRIORBOX_POWERPC_H
#define LAYER_PRIORBOX_POWERPC_H

#include "priorbox.h"

namespace ncnn {

class PriorBox_powerpc : public PriorBox
{
public:
    PriorBox_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PRIORBOX_POWERPC_H
