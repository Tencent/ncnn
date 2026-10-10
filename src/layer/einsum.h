// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_EINSUM_H
#define LAYER_EINSUM_H

#include "layer.h"

namespace ncnn {

class Einsum : public Layer
{
public:
    Einsum();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    // equation tokens
    std::vector<std::string> lhs_tokens;
    std::string rhs_token;
};

} // namespace ncnn

#endif // LAYER_EINSUM_H
