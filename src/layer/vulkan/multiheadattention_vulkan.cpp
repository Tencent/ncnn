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

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

MultiHeadAttention_vulkan::MultiHeadAttention_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    q_gemm = 0;
    k_gemm = 0;
    v_gemm = 0;

    qk_softmax = 0;

    o_gemm = 0;

    pipeline_multiheadattention_qk_cross = 0;
    pipeline_multiheadattention_qk_cross_pack4 = 0;
    pipeline_multiheadattention_qk_cross_pack1to4 = 0;
    pipeline_multiheadattention_qk_cross_pack4to1 = 0;

    pipeline_multiheadattention_qkv_cross = 0;
    pipeline_multiheadattention_qkv_cross_pack4 = 0;
    pipeline_multiheadattention_qkv_cross_pack1to4 = 0;
    pipeline_multiheadattention_qkv_cross_pack4to1 = 0;
}

int MultiHeadAttention_vulkan::create_pipeline(const Option& opt)
{
    const int embed_dim_per_head = embed_dim / num_heads;
    {
        const float inv_sqrt_embed_dim_per_head = 1.f / sqrtf(embed_dim_per_head);

        q_gemm = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
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

        q_weight_data.release();
        q_bias_data.release();
    }

    {
        k_gemm = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
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

        k_weight_data.release();
        k_bias_data.release();
    }

    {
        v_gemm = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
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

        v_weight_data.release();
        v_bias_data.release();
    }

    {
        std::vector<vk_specialization_type> specializations(6);
        specializations[0].i = attn_mask;
        specializations[1].i = 0; //constantM;
        specializations[2].i = 0; //constantN;
        specializations[3].i = 0; //embed_dim_per_head;//constantK;
        specializations[4].i = num_heads;
        specializations[5].i = 0; //attn_mask.dims;

        {
            pipeline_multiheadattention_qk_cross = new Pipeline(vkdev);
            pipeline_multiheadattention_qk_cross->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qk_cross->create(LayerShaderType::multiheadattention_qk_cross, opt, specializations);
        }
        {
            pipeline_multiheadattention_qk_cross_pack4 = new Pipeline(vkdev);
            pipeline_multiheadattention_qk_cross_pack4->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qk_cross_pack4->create(LayerShaderType::multiheadattention_qk_cross_pack4, opt, specializations);
        }
        {
            pipeline_multiheadattention_qk_cross_pack1to4 = new Pipeline(vkdev);
            pipeline_multiheadattention_qk_cross_pack1to4->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qk_cross_pack1to4->create(LayerShaderType::multiheadattention_qk_cross_pack1to4, opt, specializations);
        }
        {
            pipeline_multiheadattention_qk_cross_pack4to1 = new Pipeline(vkdev);
            pipeline_multiheadattention_qk_cross_pack4to1->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qk_cross_pack4to1->create(LayerShaderType::multiheadattention_qk_cross_pack4to1, opt, specializations);
        }
    }
    {
        std::vector<vk_specialization_type> specializations(4);
        specializations[0].i = 0; //constantM;
        specializations[1].i = 0; //embed_dim_per_head;//constantN;
        specializations[2].i = 0; //constantK;
        specializations[3].i = num_heads;

        {
            pipeline_multiheadattention_qkv_cross = new Pipeline(vkdev);
            pipeline_multiheadattention_qkv_cross->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qkv_cross->create(LayerShaderType::multiheadattention_qkv_cross, opt, specializations);
        }
        {
            pipeline_multiheadattention_qkv_cross_pack4 = new Pipeline(vkdev);
            pipeline_multiheadattention_qkv_cross_pack4->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qkv_cross_pack4->create(LayerShaderType::multiheadattention_qkv_cross_pack4, opt, specializations);
        }
        {
            pipeline_multiheadattention_qkv_cross_pack1to4 = new Pipeline(vkdev);
            pipeline_multiheadattention_qkv_cross_pack1to4->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qkv_cross_pack1to4->create(LayerShaderType::multiheadattention_qkv_cross_pack1to4, opt, specializations);
        }
        {
            pipeline_multiheadattention_qkv_cross_pack4to1 = new Pipeline(vkdev);
            pipeline_multiheadattention_qkv_cross_pack4to1->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qkv_cross_pack4to1->create(LayerShaderType::multiheadattention_qkv_cross_pack4to1, opt, specializations);
        }
    }

    {
        qk_softmax = ncnn::create_layer_vulkan(ncnn::LayerType::Softmax);
        qk_softmax->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(0, -1);
        pd.set(1, 1);
        qk_softmax->load_param(pd);
        qk_softmax->load_model(ModelBinFromMatArray(0));
        qk_softmax->create_pipeline(opt);
    }

    {
        o_gemm = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
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

        out_weight_data.release();
        out_bias_data.release();
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

    delete pipeline_multiheadattention_qk_cross;
    pipeline_multiheadattention_qk_cross = 0;

    delete pipeline_multiheadattention_qk_cross_pack4;
    pipeline_multiheadattention_qk_cross_pack4 = 0;

    delete pipeline_multiheadattention_qk_cross_pack1to4;
    pipeline_multiheadattention_qk_cross_pack1to4 = 0;

    delete pipeline_multiheadattention_qk_cross_pack4to1;
    pipeline_multiheadattention_qk_cross_pack4to1 = 0;

    delete pipeline_multiheadattention_qkv_cross;
    pipeline_multiheadattention_qkv_cross = 0;

    delete pipeline_multiheadattention_qkv_cross_pack4;
    pipeline_multiheadattention_qkv_cross_pack4 = 0;

    delete pipeline_multiheadattention_qkv_cross_pack1to4;
    pipeline_multiheadattention_qkv_cross_pack1to4 = 0;

    delete pipeline_multiheadattention_qkv_cross_pack4to1;
    pipeline_multiheadattention_qkv_cross_pack4to1 = 0;

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
    const VkMat& k_blob = (bottom_blobs.size() == 1 || (bottom_blobs.size() == 2 && attn_mask)) ? q_blob : bottom_blobs[1];
    const VkMat& v_blob = (bottom_blobs.size() == 1 || (bottom_blobs.size() == 2 && attn_mask)) ? q_blob : (bottom_blobs.size() == 2 || (bottom_blobs.size() == 3 && attn_mask)) ? k_blob : bottom_blobs[2];
    VkMat attn_mask_blob = attn_mask ? bottom_blobs[bottom_blobs.size() - 1] : VkMat();

    const int embed_dim_per_head = embed_dim / num_heads;
    const int src_seqlen = q_blob.h * q_blob.elempack;
    const int dst_seqlen = k_blob.h * k_blob.elempack;

    VkMat q_affine;
    q_gemm->forward(q_blob, q_affine, cmd, opt);

    VkMat k_affine;
    k_gemm->forward(k_blob, k_affine, cmd, opt);

    VkMat qk_cross;
    {
        int M = q_affine.w;
        int N = k_affine.w;
        int K = q_affine.h * q_affine.elempack / num_heads;
        int B = num_heads;

        // int K_elempack = opt.use_shader_pack8 && K % 8 == 0 ? 8 : K % 4 == 0 ? 4 : 1;
        // int M_elempack = opt.use_shader_pack8 && M % 8 == 0 ? 8 : M % 4 == 0 ? 4 : 1;
        // int MB_elempack = opt.use_shader_pack8 && (M * B) % 8 == 0 ? 8 : (M * B) % 4 == 0 ? 4 : 1;
        int K_elempack = K % 4 == 0 ? 4 : 1;
        int M_elempack = M % 4 == 0 ? 4 : 1;
        int MB_elempack = (M * B) % 4 == 0 ? 4 : 1;
        size_t M_elemsize = q_affine.elemsize / q_affine.elempack * M_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (M_elempack == 8) M_elemsize = 8 * 2u;
            if (M_elempack == 4) M_elemsize = 4 * 2u;
            if (M_elempack == 1) M_elemsize = 4u;
        }

        if (K_elempack < q_affine.elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(q_affine, tmp, K_elempack, cmd, opt);
            q_affine = tmp;
        }
        if (K_elempack < k_affine.elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(k_affine, tmp, K_elempack, cmd, opt);
            k_affine = tmp;
        }
        if (M_elempack < attn_mask_blob.elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(attn_mask_blob, tmp, M_elempack, cmd, opt);
            attn_mask_blob = tmp;
        }

        qk_cross.create(N, M / M_elempack * B, M_elemsize, M_elempack, opt.blob_vkallocator);
        if (qk_cross.empty())
            return -100;

        std::vector<VkMat> bindings(4);
        bindings[0] = q_affine;
        bindings[1] = k_affine;
        bindings[2] = qk_cross;
        bindings[3] = attn_mask_blob;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = M / M_elempack;
        constants[1].i = N;
        constants[2].i = K / K_elempack;
        constants[3].i = B;
        constants[4].i = attn_mask_blob.dims;

        VkMat dispatcher;
        dispatcher.w = N;
        dispatcher.h = M / M_elempack;
        dispatcher.c = B;

        const Pipeline* pipeline = 0;
        if (K_elempack == 1 && M_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qk_cross;
        }
        if (K_elempack == 1 && M_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qk_cross_pack1to4;
        }
        if (K_elempack == 4 && M_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qk_cross_pack4to1;
        }
        if (K_elempack == 4 && M_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qk_cross_pack4;
        }

        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

        if (MB_elempack > M_elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(qk_cross, tmp, MB_elempack, cmd, opt);
            qk_cross = tmp;
        }
    }

    q_affine.release();
    k_affine.release();

    qk_softmax->forward_inplace(qk_cross, cmd, opt);

    if (vkdev->info.vendor_id() == 0x10de)
    {
        // FIXME softmax produces nan result on nvidia (about 20% chance)
        // memory barrier seems to be not enough here
        // device copy-to and copy-back is better than queue submit anyway  --- nihui

        // cmd.submit_and_wait();
        // cmd.reset();

        VkImageMat qk_cross2;
        cmd.record_buffer_to_image(qk_cross, qk_cross2, opt);
        cmd.record_image_to_buffer(qk_cross2, qk_cross, opt);
    }

    VkMat v_affine;
    v_gemm->forward(v_blob, v_affine, cmd, opt);

    VkMat qkv_cross;
    {
        int M = qk_cross.h * qk_cross.elempack / num_heads;
        int N = v_affine.h * v_affine.elempack / num_heads;
        int K = v_affine.w;
        int B = num_heads;

        // int M_elempack = opt.use_shader_pack8 && M % 8 == 0 ? 8 : M % 4 == 0 ? 4 : 1;
        // int N_elempack = opt.use_shader_pack8 && N % 8 == 0 ? 8 : N % 4 == 0 ? 4 : 1;
        // int NB_elempack = opt.use_shader_pack8 && (N * B) % 8 == 0 ? 8 : (N * B) % 4 == 0 ? 4 : 1;
        int M_elempack = M % 4 == 0 ? 4 : 1;
        int N_elempack = N % 4 == 0 ? 4 : 1;
        int NB_elempack = (N * B) % 4 == 0 ? 4 : 1;
        size_t N_elemsize = v_affine.elemsize / v_affine.elempack * N_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (N_elempack == 8) N_elemsize = 8 * 2u;
            if (N_elempack == 4) N_elemsize = 4 * 2u;
            if (N_elempack == 1) N_elemsize = 4u;
        }

        if (M_elempack < qk_cross.elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(qk_cross, tmp, M_elempack, cmd, opt);
            qk_cross = tmp;
        }

        if (N_elempack < v_affine.elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(v_affine, tmp, N_elempack, cmd, opt);
            v_affine = tmp;
        }

        qkv_cross.create(M, N / N_elempack * B, N_elemsize, N_elempack, opt.blob_vkallocator);
        if (qkv_cross.empty())
            return -100;

        std::vector<VkMat> bindings(3);
        bindings[0] = qk_cross;
        bindings[1] = v_affine;
        bindings[2] = qkv_cross;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = M / M_elempack;
        constants[1].i = N / N_elempack;
        constants[2].i = K;
        constants[3].i = B;

        VkMat dispatcher;
        dispatcher.w = N / N_elempack;
        dispatcher.h = M / M_elempack;
        dispatcher.c = B;

        const Pipeline* pipeline = 0;
        if (M_elempack == 1 && N_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qkv_cross;
        }
        if (M_elempack == 1 && N_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qkv_cross_pack1to4;
        }
        if (M_elempack == 4 && N_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qkv_cross_pack4to1;
        }
        if (M_elempack == 4 && N_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qkv_cross_pack4;
        }

        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

        if (NB_elempack > N_elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(qkv_cross, tmp, NB_elempack, cmd, opt);
            qkv_cross = tmp;
        }
    }

    qk_cross.release();
    v_affine.release();

    o_gemm->forward(qkv_cross, top_blobs[0], cmd, opt);

    return 0;
}

int MultiHeadAttention_vulkan::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkImageMat& q_blob = bottom_blobs[0];
    const VkImageMat& k_blob = (bottom_blobs.size() == 1 || (bottom_blobs.size() == 2 && attn_mask)) ? q_blob : bottom_blobs[1];
    const VkImageMat& v_blob = (bottom_blobs.size() == 1 || (bottom_blobs.size() == 2 && attn_mask)) ? q_blob : (bottom_blobs.size() == 2 || (bottom_blobs.size() == 3 && attn_mask)) ? k_blob : bottom_blobs[2];
    VkImageMat attn_mask_blob = attn_mask ? bottom_blobs[bottom_blobs.size() - 1] : VkImageMat();

    const int embed_dim_per_head = embed_dim / num_heads;
    const int src_seqlen = q_blob.h * q_blob.elempack;
    const int dst_seqlen = k_blob.h * k_blob.elempack;

    VkImageMat q_affine;
    q_gemm->forward(q_blob, q_affine, cmd, opt);

    VkImageMat k_affine;
    k_gemm->forward(k_blob, k_affine, cmd, opt);

    VkImageMat qk_cross;
    {
        int M = q_affine.w;
        int N = k_affine.w;
        int K = q_affine.h * q_affine.elempack / num_heads;
        int B = num_heads;

        // int K_elempack = opt.use_shader_pack8 && K % 8 == 0 ? 8 : K % 4 == 0 ? 4 : 1;
        // int M_elempack = opt.use_shader_pack8 && M % 8 == 0 ? 8 : M % 4 == 0 ? 4 : 1;
        // int MB_elempack = opt.use_shader_pack8 && (M * B) % 8 == 0 ? 8 : (M * B) % 4 == 0 ? 4 : 1;
        int K_elempack = K % 4 == 0 ? 4 : 1;
        int M_elempack = M % 4 == 0 ? 4 : 1;
        int MB_elempack = (M * B) % 4 == 0 ? 4 : 1;
        size_t M_elemsize = q_affine.elemsize / q_affine.elempack * M_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (M_elempack == 8) M_elemsize = 8 * 2u;
            if (M_elempack == 4) M_elemsize = 4 * 2u;
            if (M_elempack == 1) M_elemsize = 4u;
        }

        if (K_elempack < q_affine.elempack)
        {
            VkImageMat tmp;
            vkdev->convert_packing(q_affine, tmp, K_elempack, cmd, opt);
            q_affine = tmp;
        }
        if (K_elempack < k_affine.elempack)
        {
            VkImageMat tmp;
            vkdev->convert_packing(k_affine, tmp, K_elempack, cmd, opt);
            k_affine = tmp;
        }
        if (M_elempack < attn_mask_blob.elempack)
        {
            VkImageMat tmp;
            vkdev->convert_packing(attn_mask_blob, tmp, M_elempack, cmd, opt);
            attn_mask_blob = tmp;
        }

        qk_cross.create(N, M / M_elempack * B, M_elemsize, M_elempack, opt.blob_vkallocator);
        if (qk_cross.empty())
            return -100;

        std::vector<VkImageMat> bindings(4);
        bindings[0] = q_affine;
        bindings[1] = k_affine;
        bindings[2] = qk_cross;
        bindings[3] = attn_mask_blob;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = M / M_elempack;
        constants[1].i = N;
        constants[2].i = K / K_elempack;
        constants[3].i = B;
        constants[4].i = attn_mask_blob.dims;

        VkImageMat dispatcher;
        dispatcher.w = N;
        dispatcher.h = M / M_elempack;
        dispatcher.c = B;

        const Pipeline* pipeline = 0;
        if (K_elempack == 1 && M_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qk_cross;
        }
        if (K_elempack == 1 && M_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qk_cross_pack1to4;
        }
        if (K_elempack == 4 && M_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qk_cross_pack4to1;
        }
        if (K_elempack == 4 && M_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qk_cross_pack4;
        }

        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

        if (MB_elempack > M_elempack)
        {
            VkImageMat tmp;
            vkdev->convert_packing(qk_cross, tmp, MB_elempack, cmd, opt);
            qk_cross = tmp;
        }
    }

    q_affine.release();
    k_affine.release();

    qk_softmax->forward_inplace(qk_cross, cmd, opt);

    VkImageMat v_affine;
    v_gemm->forward(v_blob, v_affine, cmd, opt);

    VkImageMat qkv_cross;
    {
        int M = qk_cross.h * qk_cross.elempack / num_heads;
        int N = v_affine.h * v_affine.elempack / num_heads;
        int K = v_affine.w;
        int B = num_heads;

        // int M_elempack = opt.use_shader_pack8 && M % 8 == 0 ? 8 : M % 4 == 0 ? 4 : 1;
        // int N_elempack = opt.use_shader_pack8 && N % 8 == 0 ? 8 : N % 4 == 0 ? 4 : 1;
        // int NB_elempack = opt.use_shader_pack8 && (N * B) % 8 == 0 ? 8 : (N * B) % 4 == 0 ? 4 : 1;
        int M_elempack = M % 4 == 0 ? 4 : 1;
        int N_elempack = N % 4 == 0 ? 4 : 1;
        int NB_elempack = (N * B) % 4 == 0 ? 4 : 1;
        size_t N_elemsize = v_affine.elemsize / v_affine.elempack * N_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (N_elempack == 8) N_elemsize = 8 * 2u;
            if (N_elempack == 4) N_elemsize = 4 * 2u;
            if (N_elempack == 1) N_elemsize = 4u;
        }

        if (M_elempack < qk_cross.elempack)
        {
            VkImageMat tmp;
            vkdev->convert_packing(qk_cross, tmp, M_elempack, cmd, opt);
            qk_cross = tmp;
        }

        if (N_elempack < v_affine.elempack)
        {
            VkImageMat tmp;
            vkdev->convert_packing(v_affine, tmp, N_elempack, cmd, opt);
            v_affine = tmp;
        }

        qkv_cross.create(M, N / N_elempack * B, N_elemsize, N_elempack, opt.blob_vkallocator);
        if (qkv_cross.empty())
            return -100;

        std::vector<VkImageMat> bindings(3);
        bindings[0] = qk_cross;
        bindings[1] = v_affine;
        bindings[2] = qkv_cross;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = M / M_elempack;
        constants[1].i = N / N_elempack;
        constants[2].i = K;
        constants[3].i = B;

        VkImageMat dispatcher;
        dispatcher.w = N / N_elempack;
        dispatcher.h = M / M_elempack;
        dispatcher.c = B;

        const Pipeline* pipeline = 0;
        if (M_elempack == 1 && N_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qkv_cross;
        }
        if (M_elempack == 1 && N_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qkv_cross_pack1to4;
        }
        if (M_elempack == 4 && N_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qkv_cross_pack4to1;
        }
        if (M_elempack == 4 && N_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qkv_cross_pack4;
        }

        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

        if (NB_elempack > N_elempack)
        {
            VkImageMat tmp;
            vkdev->convert_packing(qkv_cross, tmp, NB_elempack, cmd, opt);
            qkv_cross = tmp;
        }
    }

    qk_cross.release();
    v_affine.release();

    o_gemm->forward(qkv_cross, top_blobs[0], cmd, opt);

    return 0;
}

} // namespace ncnn
