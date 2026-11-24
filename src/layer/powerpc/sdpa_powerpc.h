// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SDPA_POWERPC_H
#define LAYER_SDPA_POWERPC_H

#include "sdpa.h"

namespace ncnn {

class SDPA_powerpc : public SDPA
{
public:
    SDPA_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SDPA_POWERPC_H
