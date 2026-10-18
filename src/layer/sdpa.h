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
    static int kvcache_capacity(int current_capacity, int new_seqlen, int max_seqlen_hint);

    int create_or_grow_kvcache(const Mat& cache, Mat& new_cache, int new_seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack, const Option& opt) const;

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
