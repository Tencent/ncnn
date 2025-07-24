// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_NOOP_H
#define LAYER_NOOP_H

#include "layer.h"

namespace ncnn {

class Noop : public Layer
{
public:
    Noop();

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_NOOP_H
