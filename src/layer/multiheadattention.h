// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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

#ifdef NCNN_INT8
    int forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    int embed_dim;
    int num_head;
    int weight_data_size;
    int int8_scale_term;

    int embed_dim_per_head;
    float inv_sqrt_embed_dim_per_head;

    Mat q_weight_data;
    Mat q_bias_data;
    Mat k_weight_data;
    Mat k_bias_data;
    Mat v_weight_data;
    Mat v_bias_data;
    Mat out_weight_data;
    Mat out_bias_data;

#ifdef NCNN_INT8
    Mat q_input_scale;
    Mat k_input_scale;
    Mat v_input_scale;

    Mat q_weight_scales;
    Mat k_weight_scales;
    Mat v_weight_scales;
    Mat o_weight_scales;

    /**
     * @brief mha consists of multiple GEMM, they also have input scale
     *
     * internal_scales = [
     *  q_affine_scale,
     *  k_affine_scale,
     *  v_affine_scale,
     *  energy_scale,
     *  feat_scael
     * ]
     *
     */
    Mat internal_scales;
#endif
};

} // namespace ncnn

#endif // LAYER_MULTIHEADATTENTION_H
