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

#include "multiheadattention_x86.h"

#include <float.h>

#include "layer_type.h"

namespace ncnn {

MultiHeadAttention_x86::MultiHeadAttention_x86()
{
    q_gemm = 0;
    permute_wch = 0;

    k_gemm = 0;
    permute_cwh = 0;

    v_gemm = 0;

    qk_matmul = 0;
    qk_softmax = 0;
    qkv_matmul = 0;

    o_gemm = 0;
}

int MultiHeadAttention_x86::create_pipeline(const Option& opt)
{
    {
        const int embed_dim_per_head = embed_dim / num_head;
        const float inv_sqrt_embed_dim_per_head = 1.f / sqrt(embed_dim_per_head);

        q_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(0, inv_sqrt_embed_dim_per_head);
        pd.set(1, 1.f);
        pd.set(2, 0);      // transA
        pd.set(3, 1);      // transB
        pd.set(4, 0);      // constantA
        pd.set(5, 0);      // constantB
        pd.set(6, 0);      // constantC
        pd.set(7, 0);      // M = outch
        pd.set(8, 0);      // N = size
        pd.set(9, 0);      // K = maxk*inch
        pd.set(10, -1);    // constant_broadcast_type_C = null
        pd.set(11, 0);     // output_N1M
        pd.set(12, 1);     // output_elempack
        q_gemm->load_param(pd);
        q_gemm->load_model(ModelBinFromMatArray(0));
        q_gemm->create_pipeline(opt);
    }
    {
        permute_wch = ncnn::create_layer(ncnn::LayerType::Permute);
        ncnn::ParamDict pd;
        pd.set(0, 2); // wch
        permute_wch->load_param(pd);
        permute_wch->load_model(ModelBinFromMatArray(0));
        permute_wch->create_pipeline(opt);
    }

    {
        k_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(2, 0);      // transA
        pd.set(3, 1);      // transB
        pd.set(4, 0);      // constantA
        pd.set(5, 0);      // constantB
        pd.set(6, 0);      // constantC
        pd.set(7, 0);      // M = outch
        pd.set(8, 0);      // N = size
        pd.set(9, 0);      // K = maxk*inch
        pd.set(10, -1);    // constant_broadcast_type_C = null
        pd.set(11, 0);     // output_N1M
        pd.set(12, 1);     // output_elempack
        k_gemm->load_param(pd);
        k_gemm->load_model(ModelBinFromMatArray(0));
        k_gemm->create_pipeline(opt);
    }
    {
        permute_cwh = ncnn::create_layer(ncnn::LayerType::Permute);
        ncnn::ParamDict pd;
        pd.set(0, 3); // cwh
        permute_cwh->load_param(pd);
        permute_cwh->load_model(ModelBinFromMatArray(0));
        permute_cwh->create_pipeline(opt);
    }

    {
        v_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(2, 0);      // transA
        pd.set(3, 1);      // transB
        pd.set(4, 0);      // constantA
        pd.set(5, 0);      // constantB
        pd.set(6, 0);      // constantC
        pd.set(7, 0);      // M = outch
        pd.set(8, 0);      // N = size
        pd.set(9, 0);      // K = maxk*inch
        pd.set(10, -1);    // constant_broadcast_type_C = null
        pd.set(11, 0);     // output_N1M
        pd.set(12, 1);     // output_elempack
        v_gemm->load_param(pd);
        v_gemm->load_model(ModelBinFromMatArray(0));
        v_gemm->create_pipeline(opt);
    }

    {
        qk_matmul = ncnn::create_layer(ncnn::LayerType::MatMul);
        ncnn::ParamDict pd;
        qk_matmul->load_param(pd);
        qk_matmul->load_model(ModelBinFromMatArray(0));
        qk_matmul->create_pipeline(opt);
    }
    {
        qk_softmax = ncnn::create_layer(ncnn::LayerType::Softmax);
        ncnn::ParamDict pd;
        pd.set(0, -1);
        pd.set(1, 1);
        qk_softmax->load_param(pd);
        qk_softmax->load_model(ModelBinFromMatArray(0));
        qk_softmax->create_pipeline(opt);
    }
    {
        qkv_matmul = ncnn::create_layer(ncnn::LayerType::MatMul);
        ncnn::ParamDict pd;
        qkv_matmul->load_param(pd);
        qkv_matmul->load_model(ModelBinFromMatArray(0));
        qkv_matmul->create_pipeline(opt);
    }

    {
        o_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(2, 0);      // transA
        pd.set(3, 1);      // transB
        pd.set(4, 0);      // constantA
        pd.set(5, 0);      // constantB
        pd.set(6, 0);      // constantC
        pd.set(7, 0);      // M = outch
        pd.set(8, 0);      // N = size
        pd.set(9, 0);      // K = maxk*inch
        pd.set(10, -1);    // constant_broadcast_type_C = null
        pd.set(11, 0);     // output_N1M
        pd.set(12, 1);     // output_elempack
        o_gemm->load_param(pd);
        o_gemm->load_model(ModelBinFromMatArray(0));
        o_gemm->create_pipeline(opt);
    }

    return 0;
}

int MultiHeadAttention_x86::destroy_pipeline(const Option& opt)
{
    if (q_gemm)
    {
        q_gemm->destroy_pipeline(opt);
        delete q_gemm;
        q_gemm = 0;
    }
    if (permute_wch)
    {
        permute_wch->destroy_pipeline(opt);
        delete permute_wch;
        permute_wch = 0;
    }

    if (k_gemm)
    {
        k_gemm->destroy_pipeline(opt);
        delete k_gemm;
        k_gemm = 0;
    }
    if (permute_cwh)
    {
        permute_cwh->destroy_pipeline(opt);
        delete permute_cwh;
        permute_cwh = 0;
    }

    if (v_gemm)
    {
        v_gemm->destroy_pipeline(opt);
        delete v_gemm;
        v_gemm = 0;
    }

    if (qk_matmul)
    {
        qk_matmul->destroy_pipeline(opt);
        delete qk_matmul;
        qk_matmul = 0;
    }
    if (qk_softmax)
    {
        qk_softmax->destroy_pipeline(opt);
        delete qk_softmax;
        qk_softmax = 0;
    }
    if (qkv_matmul)
    {
        qkv_matmul->destroy_pipeline(opt);
        delete qkv_matmul;
        qkv_matmul = 0;
    }

    if (o_gemm)
    {
        o_gemm->destroy_pipeline(opt);
        delete o_gemm;
        o_gemm = 0;
    }

    return 0;
}

int MultiHeadAttention_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& q_blob = bottom_blobs[0];
    const Mat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const Mat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs.size() == 2 ? k_blob : bottom_blobs[2];

    const int embed_dim_per_head = embed_dim / num_head;
    const int src_seqlen = q_blob.h;
    const int dst_seqlen = k_blob.h;

    Option opt_no = opt;
    opt_no.use_packing_layout = false;

    ncnn::Mat q_affine_reshape_wch;
    {
        std::vector<Mat> q_bottom_blobs(3);
        q_bottom_blobs[0] = q_blob;
        q_bottom_blobs[1] = q_weight_data.reshape(embed_dim, embed_dim);
        q_bottom_blobs[2] = q_bias_data;
        std::vector<Mat> q_affine(1);
        q_gemm->forward(q_bottom_blobs, q_affine, opt_no);

        ncnn::Mat q_affine_reshape = q_affine[0].reshape(embed_dim_per_head,num_head,src_seqlen);

        permute_wch->forward(q_affine_reshape, q_affine_reshape_wch, opt_no);
    }

    ncnn::Mat v_affine_reshape_wch;
    {
        std::vector<Mat> v_bottom_blobs(3);
        v_bottom_blobs[0] = v_blob;
        v_bottom_blobs[1] = v_weight_data.reshape(vdim, embed_dim);
        v_bottom_blobs[2] = v_bias_data;
        std::vector<Mat> v_affine(1);
        v_gemm->forward(v_bottom_blobs, v_affine, opt_no);

        ncnn::Mat v_affine_reshape = v_affine[0].reshape(embed_dim_per_head,num_head,dst_seqlen);

        permute_wch->forward(v_affine_reshape, v_affine_reshape_wch, opt_no);
    }

    ncnn::Mat k_affine_reshape_cwh;
    {
        std::vector<Mat> k_bottom_blobs(3);
        k_bottom_blobs[0] = k_blob;
        k_bottom_blobs[1] = k_weight_data.reshape(kdim, embed_dim);
        k_bottom_blobs[2] = k_bias_data;
        std::vector<Mat> k_affine(1);
        k_gemm->forward(k_bottom_blobs, k_affine, opt_no);

        ncnn::Mat k_affine_reshape = k_affine[0].reshape(embed_dim_per_head,num_head,dst_seqlen);

        permute_cwh->forward(k_affine_reshape, k_affine_reshape_cwh, opt_no);
    }

    std::vector<Mat> qkv_cross(1);
    {
        std::vector<Mat> qk_bottom_blobs(2);
        qk_bottom_blobs[0] = q_affine_reshape_wch;
        qk_bottom_blobs[1] = k_affine_reshape_cwh;
        std::vector<Mat> qk_cross(1);
        qk_matmul->forward(qk_bottom_blobs, qk_cross, opt_no); 

        qk_softmax->forward_inplace(qk_cross[0], opt_no);

        std::vector<Mat> qkv_bottom_blobs(2);
        qkv_bottom_blobs[0] = qk_cross[0];
        qkv_bottom_blobs[1] = v_affine_reshape_wch;
        qkv_matmul->forward(qkv_bottom_blobs, qkv_cross, opt_no);
    }

    {
        ncnn::Mat qkv_wch;
        permute_wch->forward(qkv_cross[0], qkv_wch, opt_no);

        ncnn::Mat qkv_better = qkv_wch.reshape(embed_dim,src_seqlen);

        std::vector<Mat> o_bottom_blobs(3);
        o_bottom_blobs[0] = qkv_better;
        o_bottom_blobs[1] = out_weight_data.reshape(embed_dim, embed_dim);
        o_bottom_blobs[2] = out_bias_data;
        o_gemm->forward(o_bottom_blobs, top_blobs, opt_no); 
    }

    return 0;
}

} // namespace ncnn
