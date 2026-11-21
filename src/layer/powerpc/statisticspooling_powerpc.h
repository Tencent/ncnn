// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_STATISTICSPOOLING_POWERPC_H
#define LAYER_STATISTICSPOOLING_POWERPC_H

#include "statisticspooling.h"

namespace ncnn {

class StatisticsPooling_powerpc : public StatisticsPooling
{
public:
    StatisticsPooling_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_STATISTICSPOOLING_POWERPC_H
