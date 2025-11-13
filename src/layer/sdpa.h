// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SDPA_H
#define LAYER_SDPA_H

#include "layer.h"

namespace ncnn {

class SDPA : public Layer
{
public:
    SDPA();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void resolve_bottom_blob_index(int bottom_blob_count, int& query_i, int& key_i, int& value_i, int& attn_mask_i, int& past_key_i, int& past_value_i) const;

#if NCNN_INT8
    int forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    int attn_mask;
    float scale;
    int kv_cache;

    int int8_scale_term;
};

} // namespace ncnn

#endif // LAYER_SDPA_H
