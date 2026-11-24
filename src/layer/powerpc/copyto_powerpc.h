// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_COPYTO_POWERPC_H
#define LAYER_COPYTO_POWERPC_H

#include "copyto.h"

namespace ncnn {

class CopyTo_powerpc : public CopyTo
{
public:
    CopyTo_powerpc();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_COPYTO_POWERPC_H
