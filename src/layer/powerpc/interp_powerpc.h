// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INTERP_POWERPC_H
#define LAYER_INTERP_POWERPC_H

#include "interp.h"

namespace ncnn {

class Interp_powerpc : public Interp
{
public:
    Interp_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_INTERP_POWERPC_H
