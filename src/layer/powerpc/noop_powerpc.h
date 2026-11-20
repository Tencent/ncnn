// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_NOOP_POWERPC_H
#define LAYER_NOOP_POWERPC_H

#include "noop.h"

namespace ncnn {

class Noop_powerpc : public Noop
{
public:
    Noop_powerpc();

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_NOOP_POWERPC_H
