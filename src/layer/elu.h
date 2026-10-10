// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELU_H
#define LAYER_ELU_H

#include "layer.h"

namespace ncnn {

class ELU : public Layer
{
public:
    ELU();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float alpha;
};

} // namespace ncnn

#endif // LAYER_ELU_H
