// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "multiheadattention_vulkan.h"

#include "layer_type.h"

namespace ncnn {

MultiHeadAttention_vulkan::MultiHeadAttention_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    q_gemm = 0;
    k_gemm = 0;
    v_gemm = 0;

    qk_gemm = 0;
    qkv_gemm = 0;

    qk_softmax = 0;

    o_gemm = 0;

    slice = 0;
    concat = 0;
}

int MultiHeadAttention_vulkan::create_pipeline(const Option& opt)
{
    {
        const int embed_dim_per_head = embed_dim / num_head;
        const float inv_sqrt_embed_dim_per_head = 1.f / sqrt(embed_dim_per_head);

        q_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        q_gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(0, inv_sqrt_embed_dim_per_head);
        pd.set(1, 1.f);
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 1);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, embed_dim); // M
        pd.set(8, 0);         // N
        pd.set(9, embed_dim); // K
        pd.set(10, 1);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        // pd.set(12, 1);        // output_elempack
        pd.set(14, 0); // output_transpose
        q_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = q_weight_data;
        weights[1] = q_bias_data;
        q_gemm->load_model(ModelBinFromMatArray(weights));
        q_gemm->create_pipeline(opt);

        if (opt.lightmode)
        {
            q_weight_data.release();
            q_bias_data.release();
        }
    }

    {
        k_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        k_gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 1);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, embed_dim); // M
        pd.set(8, 0);         // N
        pd.set(9, kdim);      // K
        pd.set(10, 1);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        // pd.set(12, 1);        // output_elempack
        pd.set(14, 0); // output_transpose
        k_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = k_weight_data;
        weights[1] = k_bias_data;
        k_gemm->load_model(ModelBinFromMatArray(weights));
        k_gemm->create_pipeline(opt);

        if (opt.lightmode)
        {
            k_weight_data.release();
            k_bias_data.release();
        }
    }

    {
        v_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        v_gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 1);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, embed_dim); // M
        pd.set(8, 0);         // N
        pd.set(9, vdim);      // K
        pd.set(10, 1);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        // pd.set(12, 1);        // output_elempack
        pd.set(14, 0); // output_transpose
        v_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = v_weight_data;
        weights[1] = v_bias_data;
        v_gemm->load_model(ModelBinFromMatArray(weights));
        v_gemm->create_pipeline(opt);

        if (opt.lightmode)
        {
            v_weight_data.release();
            v_bias_data.release();
        }
    }

    {
        qk_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        qk_gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(2, 1);   // transA
        pd.set(3, 0);   // transB
        pd.set(4, 0);   // constantA
        pd.set(5, 0);   // constantB
        pd.set(6, 1);   // constantC
        pd.set(7, 0);   // M
        pd.set(8, 0);   // N
        pd.set(9, 0);   // K
        pd.set(10, -1); // constant_broadcast_type_C
        pd.set(11, 0);  // output_N1M
        // pd.set(12, 1);  // output_elempack
        qk_gemm->load_param(pd);
        qk_gemm->load_model(ModelBinFromMatArray(0));
        Option opt1 = opt;
        opt1.num_threads = 1;
        qk_gemm->create_pipeline(opt1);
    }
    {
        qkv_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        qkv_gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(2, 0);   // transA
        pd.set(3, 1);   // transB
        pd.set(4, 0);   // constantA
        pd.set(5, 0);   // constantB
        pd.set(6, 1);   // constantC
        pd.set(7, 0);   // M
        pd.set(8, 0);   // N
        pd.set(9, 0);   // K
        pd.set(10, -1); // constant_broadcast_type_C
        pd.set(11, 0);  // output_N1M
        // pd.set(12, 1);  // output_elempack
        pd.set(14, 1); // output_transpose
        qkv_gemm->load_param(pd);
        qkv_gemm->load_model(ModelBinFromMatArray(0));
        Option opt1 = opt;
        opt1.num_threads = 1;
        qkv_gemm->create_pipeline(opt1);
    }

    {
        qk_softmax = ncnn::create_layer(ncnn::LayerType::Softmax);
        qk_softmax->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(0, -1);
        pd.set(1, 1);
        qk_softmax->load_param(pd);
        qk_softmax->load_model(ModelBinFromMatArray(0));
        qk_softmax->create_pipeline(opt);
    }

    {
        o_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        o_gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(2, 1);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 1);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M = outch
        pd.set(8, embed_dim); // N = size
        pd.set(9, embed_dim); // K = maxk*inch
        pd.set(10, 4);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        o_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = out_weight_data;
        weights[1] = out_bias_data;
        o_gemm->load_model(ModelBinFromMatArray(weights));
        o_gemm->create_pipeline(opt);

        if (opt.lightmode)
        {
            out_weight_data.release();
            out_bias_data.release();
        }
    }

    {
        slice = ncnn::create_layer(ncnn::LayerType::Slice);
        slice->vkdev = vkdev;
        ncnn::ParamDict pd;
        Mat slices(num_head);
        slices.fill<int>(-233);
        pd.set(0, slices);
        pd.set(1, 0);
        slice->load_param(pd);
        slice->load_model(ModelBinFromMatArray(0));
        slice->create_pipeline(opt);
    }

    {
        concat = ncnn::create_layer(ncnn::LayerType::Concat);
        concat->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(0, 0);
        concat->load_param(pd);
        concat->load_model(ModelBinFromMatArray(0));
        concat->create_pipeline(opt);
    }

    return 0;
}

int MultiHeadAttention_vulkan::destroy_pipeline(const Option& opt)
{
    if (q_gemm)
    {
        q_gemm->destroy_pipeline(opt);
        delete q_gemm;
        q_gemm = 0;
    }

    if (k_gemm)
    {
        k_gemm->destroy_pipeline(opt);
        delete k_gemm;
        k_gemm = 0;
    }

    if (v_gemm)
    {
        v_gemm->destroy_pipeline(opt);
        delete v_gemm;
        v_gemm = 0;
    }

    if (qk_gemm)
    {
        qk_gemm->destroy_pipeline(opt);
        delete qk_gemm;
        qk_gemm = 0;
    }
    if (qkv_gemm)
    {
        qkv_gemm->destroy_pipeline(opt);
        delete qkv_gemm;
        qkv_gemm = 0;
    }

    if (qk_softmax)
    {
        qk_softmax->destroy_pipeline(opt);
        delete qk_softmax;
        qk_softmax = 0;
    }

    if (o_gemm)
    {
        o_gemm->destroy_pipeline(opt);
        delete o_gemm;
        o_gemm = 0;
    }

    return 0;
}

int MultiHeadAttention_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (q_gemm)
    {
        q_gemm->upload_model(cmd, opt);
    }

    if (k_gemm)
    {
        k_gemm->upload_model(cmd, opt);
    }

    if (v_gemm)
    {
        v_gemm->upload_model(cmd, opt);
    }

    if (o_gemm)
    {
        o_gemm->upload_model(cmd, opt);
    }

    return 0;
}

int MultiHeadAttention_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& q_blob = bottom_blobs[0];
    const VkMat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const VkMat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs.size() == 2 ? k_blob : bottom_blobs[2];

    const int embed_dim_per_head = embed_dim / num_head;
    const int src_seqlen = q_blob.h * q_blob.elempack;
    const int dst_seqlen = k_blob.h * k_blob.elempack;

    std::vector<VkMat> q_gemm_inputs(1);
    std::vector<VkMat> q_gemm_outputs(1);
    q_gemm_inputs[0] = q_blob;
    q_gemm->forward(q_gemm_inputs, q_gemm_outputs, cmd, opt);
    VkMat q_affine = q_gemm_outputs[0];

    std::vector<VkMat> k_gemm_inputs(1);
    std::vector<VkMat> k_gemm_outputs(1);
    k_gemm_inputs[0] = k_blob;
    k_gemm->forward(k_gemm_inputs, k_gemm_outputs, cmd, opt);
    VkMat k_affine = k_gemm_outputs[0];

    std::vector<VkMat> q_affine_slice_inputs(1);
    q_affine_slice_inputs[0] = q_affine;
    std::vector<VkMat> q_affine_slice_outputs(num_head);
    slice->forward(q_affine_slice_inputs, q_affine_slice_outputs, cmd, opt);

    std::vector<VkMat> k_affine_slice_inputs(1);
    k_affine_slice_inputs[0] = k_affine;
    std::vector<VkMat> k_affine_slice_outputs(num_head);
    slice->forward(k_affine_slice_inputs, k_affine_slice_outputs, cmd, opt);

    std::vector<VkMat> qk_cross_concat_inputs(num_head);
    for (int i = 0; i < num_head; i++)
    {
        std::vector<VkMat> qk_gemm_inputs(2);
        qk_gemm_inputs[0] = q_affine_slice_outputs[i];
        qk_gemm_inputs[1] = k_affine_slice_outputs[i];
        std::vector<VkMat> qk_gemm_outputs(1);
        qk_gemm->forward(qk_gemm_inputs, qk_gemm_outputs, cmd, opt);
        qk_cross_concat_inputs[i] = qk_gemm_outputs[0];
    }

    std::vector<VkMat> qk_cross_concat_outputs(1);
    concat->forward(qk_cross_concat_inputs, qk_cross_concat_outputs, cmd, opt);
    VkMat qk_cross = qk_cross_concat_outputs[0];

    q_affine.release();
    k_affine.release();

    qk_softmax->forward_inplace(qk_cross, cmd, opt);

    std::vector<VkMat> v_gemm_inputs(1);
    std::vector<VkMat> v_gemm_outputs(1);
    v_gemm_inputs[0] = v_blob;
    v_gemm->forward(v_gemm_inputs, v_gemm_outputs, cmd, opt);
    VkMat v_affine = v_gemm_outputs[0];

    std::vector<VkMat> qk_cross_slice_inputs(1);
    qk_cross_slice_inputs[0] = qk_cross;
    std::vector<VkMat> qk_cross_slice_outputs(num_head);
    slice->forward(qk_cross_slice_inputs, qk_cross_slice_outputs, cmd, opt);

    std::vector<VkMat> v_affine_slice_inputs(1);
    v_affine_slice_inputs[0] = v_affine;
    std::vector<VkMat> v_affine_slice_outputs(num_head);
    slice->forward(v_affine_slice_inputs, v_affine_slice_outputs, cmd, opt);

    std::vector<VkMat> qkv_cross_concat_inputs(num_head);
    for (int i = 0; i < num_head; i++)
    {
        std::vector<VkMat> qkv_gemm_inputs(2);
        qkv_gemm_inputs[0] = qk_cross_slice_outputs[i];
        qkv_gemm_inputs[1] = v_affine_slice_outputs[i];
        std::vector<VkMat> qkv_gemm_outputs(1);
        qkv_gemm->forward(qkv_gemm_inputs, qkv_gemm_outputs, cmd, opt);
        qkv_cross_concat_inputs[i] = qkv_gemm_outputs[0];
    }

    std::vector<VkMat> qkv_cross_concat_outputs(1);
    concat->forward(qkv_cross_concat_inputs, qkv_cross_concat_outputs, cmd, opt);
    VkMat qkv_cross = qkv_cross_concat_outputs[0];

    v_affine.release();

    std::vector<VkMat> o_gemm_inputs(1);
    std::vector<VkMat> o_gemm_outputs(1);
    o_gemm_inputs[0] = qkv_cross;
    o_gemm->forward(o_gemm_inputs, o_gemm_outputs, cmd, opt);
    top_blobs[0] = o_gemm_outputs[0];

    return 0;
}

int MultiHeadAttention_vulkan::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkImageMat& q_blob = bottom_blobs[0];
    const VkImageMat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const VkImageMat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs.size() == 2 ? k_blob : bottom_blobs[2];

    const int embed_dim_per_head = embed_dim / num_head;
    const int src_seqlen = q_blob.h * q_blob.elempack;
    const int dst_seqlen = k_blob.h * k_blob.elempack;

    std::vector<VkImageMat> q_gemm_inputs(1);
    std::vector<VkImageMat> q_gemm_outputs(1);
    q_gemm_inputs[0] = q_blob;
    q_gemm->forward(q_gemm_inputs, q_gemm_outputs, cmd, opt);
    VkImageMat q_affine = q_gemm_outputs[0];

    std::vector<VkImageMat> k_gemm_inputs(1);
    std::vector<VkImageMat> k_gemm_outputs(1);
    k_gemm_inputs[0] = k_blob;
    k_gemm->forward(k_gemm_inputs, k_gemm_outputs, cmd, opt);
    VkImageMat k_affine = k_gemm_outputs[0];

    std::vector<VkImageMat> q_affine_slice_inputs(1);
    q_affine_slice_inputs[0] = q_affine;
    std::vector<VkImageMat> q_affine_slice_outputs(num_head);
    slice->forward(q_affine_slice_inputs, q_affine_slice_outputs, cmd, opt);

    std::vector<VkImageMat> k_affine_slice_inputs(1);
    k_affine_slice_inputs[0] = k_affine;
    std::vector<VkImageMat> k_affine_slice_outputs(num_head);
    slice->forward(k_affine_slice_inputs, k_affine_slice_outputs, cmd, opt);

    std::vector<VkImageMat> qk_cross_concat_inputs(num_head);
    for (int i = 0; i < num_head; i++)
    {
        std::vector<VkImageMat> qk_gemm_inputs(2);
        qk_gemm_inputs[0] = q_affine_slice_outputs[i];
        qk_gemm_inputs[1] = k_affine_slice_outputs[i];
        std::vector<VkImageMat> qk_gemm_outputs(1);
        qk_gemm->forward(qk_gemm_inputs, qk_gemm_outputs, cmd, opt);
        qk_cross_concat_inputs[i] = qk_gemm_outputs[0];
    }

    std::vector<VkImageMat> qk_cross_concat_outputs(1);
    concat->forward(qk_cross_concat_inputs, qk_cross_concat_outputs, cmd, opt);
    VkImageMat qk_cross = qk_cross_concat_outputs[0];

    q_affine.release();
    k_affine.release();

    qk_softmax->forward_inplace(qk_cross, cmd, opt);

    std::vector<VkImageMat> v_gemm_inputs(1);
    std::vector<VkImageMat> v_gemm_outputs(1);
    v_gemm_inputs[0] = v_blob;
    v_gemm->forward(v_gemm_inputs, v_gemm_outputs, cmd, opt);
    VkImageMat v_affine = v_gemm_outputs[0];

    std::vector<VkImageMat> qk_cross_slice_inputs(1);
    qk_cross_slice_inputs[0] = qk_cross;
    std::vector<VkImageMat> qk_cross_slice_outputs(num_head);
    slice->forward(qk_cross_slice_inputs, qk_cross_slice_outputs, cmd, opt);

    std::vector<VkImageMat> v_affine_slice_inputs(1);
    v_affine_slice_inputs[0] = v_affine;
    std::vector<VkImageMat> v_affine_slice_outputs(num_head);
    slice->forward(v_affine_slice_inputs, v_affine_slice_outputs, cmd, opt);

    std::vector<VkImageMat> qkv_cross_concat_inputs(num_head);
    for (int i = 0; i < num_head; i++)
    {
        std::vector<VkImageMat> qkv_gemm_inputs(2);
        qkv_gemm_inputs[0] = qk_cross_slice_outputs[i];
        qkv_gemm_inputs[1] = v_affine_slice_outputs[i];
        std::vector<VkImageMat> qkv_gemm_outputs(1);
        qkv_gemm->forward(qkv_gemm_inputs, qkv_gemm_outputs, cmd, opt);
        qkv_cross_concat_inputs[i] = qkv_gemm_outputs[0];
    }

    std::vector<VkImageMat> qkv_cross_concat_outputs(1);
    concat->forward(qkv_cross_concat_inputs, qkv_cross_concat_outputs, cmd, opt);
    VkImageMat qkv_cross = qkv_cross_concat_outputs[0];

    v_affine.release();

    std::vector<VkImageMat> o_gemm_inputs(1);
    std::vector<VkImageMat> o_gemm_outputs(1);
    o_gemm_inputs[0] = qkv_cross;
    o_gemm->forward(o_gemm_inputs, o_gemm_outputs, cmd, opt);
    top_blobs[0] = o_gemm_outputs[0];

    return 0;
}

} // namespace ncnn
