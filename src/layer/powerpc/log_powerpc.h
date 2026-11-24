// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LOG_POWERPC_H
#define LAYER_LOG_POWERPC_H

#include "log.h"

namespace ncnn {

class Log_powerpc : public Log
{
public:
    Log_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_LOG_POWERPC_H
