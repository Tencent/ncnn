// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INTERP_X86_H
#define LAYER_INTERP_X86_H

#include "interp.h"

namespace ncnn {

class Interp_x86 : public Interp
{
public:
    Interp_x86();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    bool is_identity_resize(int w, int h, int outw, int outh, int dims) const;
    int eval_size_expr(const std::vector<Mat>& bottom_blobs, int& outw, int& outh) const;

    double source_width_scale;
    double source_height_scale;
    int source_scale_dims;
    float height_scale_inv;
    float width_scale_inv;

#if NCNN_BF16
    int forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_INTERP_X86_H
