// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MULTIHEADATTENTION_H
#define LAYER_MULTIHEADATTENTION_H

#include "layer.h"

namespace ncnn {

class MultiHeadAttention : public Layer
{
public:
    MultiHeadAttention();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void resolve_bottom_blob_index(int bottom_blob_count, int& q_blob_i, int& k_blob_i, int& v_blob_i, int& attn_mask_i, int& cached_xk_i, int& cached_xv_i) const;

#if NCNN_INT8
    int forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    int embed_dim;
    int num_heads;
    int weight_data_size;
    int kdim;
    int vdim;
    int attn_mask;
    float scale;
    int kv_cache;

    int int8_scale_term;

    Mat q_weight_data;
    Mat q_bias_data;
    Mat k_weight_data;
    Mat k_bias_data;
    Mat v_weight_data;
    Mat v_bias_data;
    Mat out_weight_data;
    Mat out_bias_data;

#if NCNN_INT8
    Mat q_weight_data_int8_scales;
    Mat k_weight_data_int8_scales;
    Mat v_weight_data_int8_scales;
    float out_weight_data_int8_scale;
#endif
};

} // namespace ncnn

#endif // LAYER_MULTIHEADATTENTION_H
