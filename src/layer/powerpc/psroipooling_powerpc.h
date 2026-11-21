// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PSROIPOOLING_POWERPC_H
#define LAYER_PSROIPOOLING_POWERPC_H

#include "psroipooling.h"

namespace ncnn {

class PSROIPooling_powerpc : public PSROIPooling
{
public:
    PSROIPooling_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PSROIPOOLING_POWERPC_H
