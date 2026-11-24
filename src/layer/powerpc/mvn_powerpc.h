// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MVN_POWERPC_H
#define LAYER_MVN_POWERPC_H

#include "mvn.h"

namespace ncnn {

class MVN_powerpc : public MVN
{
public:
    MVN_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_MVN_POWERPC_H
