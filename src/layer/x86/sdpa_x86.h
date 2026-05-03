// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SDPA_X86_H
#define LAYER_SDPA_X86_H

#include "sdpa.h"

namespace ncnn {

class SDPA_x86 : public SDPA
{
public:
    SDPA_x86();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

private:
#if NCNN_INT8
    mutable Mat cached_key_int8;
    mutable Mat cached_key_scales;
    mutable Mat cached_value_int8;
    mutable Mat cached_value_scales;
    mutable int cached_kv_seqlen;
    mutable int cached_num_group;
    mutable int cached_embed_dim;
    mutable int cached_out_embed_dim;
#endif
};

} // namespace ncnn

#endif // LAYER_SDPA_X86_H
