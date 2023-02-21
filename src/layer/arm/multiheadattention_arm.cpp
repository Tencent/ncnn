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

#include "multiheadattention_arm.h"

#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

MultiHeadAttention_arm::MultiHeadAttention_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

    support_bf16_storage = false;

    cvtfp16_to_fp32 = 0;
    cvtfp32_to_fp16 = 0;

    q_gemm = 0;
    k_gemm = 0;
    v_gemm = 0;
    o_gemm = 0;

    qk_gemm = 0;
    qkv_gemm = 0;

    qk_softmax = 0;
    permute_wch = 0;
}

int MultiHeadAttention_arm::create_pipeline(const Option& opt)
{
    Option optn = opt;
    optn.use_bf16_storage = false;

    Option opt32 = opt;
    opt32.use_bf16_storage = false;
    opt32.use_fp16_arithmetic = false;
    opt32.use_fp16_packed = false;
    opt32.use_fp16_storage = false;

    {
        cvtfp16_to_fp32 = ncnn::create_layer(ncnn::LayerType::Cast);
        ncnn::ParamDict pd;
        pd.set(0, 2); // from fp16
        pd.set(1, 1); // from fp32
        cvtfp16_to_fp32->load_param(pd);
        cvtfp16_to_fp32->load_model(ModelBinFromMatArray(0));
        cvtfp16_to_fp32->create_pipeline(optn);
    }
    {
        cvtfp32_to_fp16 = ncnn::create_layer(ncnn::LayerType::Cast);
        ncnn::ParamDict pd;
        pd.set(0, 1); // from fp32
        pd.set(1, 2); // from fp16
        cvtfp32_to_fp16->load_param(pd);
        cvtfp32_to_fp16->load_model(ModelBinFromMatArray(0));
        cvtfp32_to_fp16->create_pipeline(optn);
    }

    {
        qk_softmax = ncnn::create_layer(ncnn::LayerType::Softmax);
        ncnn::ParamDict pd;
        pd.set(0, -1);
        pd.set(1, 1);
        qk_softmax->load_param(pd);
        qk_softmax->load_model(ModelBinFromMatArray(0));
        qk_softmax->create_pipeline(opt32);
    }
    {
        permute_wch = ncnn::create_layer(ncnn::LayerType::Permute);
        ncnn::ParamDict pd;
        pd.set(0, 2); // wch
        permute_wch->load_param(pd);
        permute_wch->load_model(ModelBinFromMatArray(0));
        permute_wch->create_pipeline(opt32);
    }

#if NCNN_ARM82
    if (support_fp16_storage && optn.use_fp16_packed)
    {
        Option optopt = optn;

        {
            const int embed_dim_per_head = embed_dim / num_head;
            const float inv_sqrt_embed_dim_per_head = 1.f / sqrt(embed_dim_per_head);

            q_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
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
            pd.set(12, 1);        // output_elempack
            q_gemm->load_param(pd);
            Mat weights[2];
            weights[0] = q_weight_data;
            weights[1] = q_bias_data;
            q_gemm->load_model(ModelBinFromMatArray(weights));
            q_gemm->create_pipeline(optopt);

            if (optopt.lightmode)
            {
                q_weight_data.release();
                q_bias_data.release();
            }
        }

        {
            k_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
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
            pd.set(12, 1);        // output_elempack
            k_gemm->load_param(pd);
            Mat weights[2];
            weights[0] = k_weight_data;
            weights[1] = k_bias_data;
            k_gemm->load_model(ModelBinFromMatArray(weights));
            k_gemm->create_pipeline(optopt);

            if (optopt.lightmode)
            {
                k_weight_data.release();
                k_bias_data.release();
            }
        }

        {
            v_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
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
            pd.set(12, 1);        // output_elempack
            v_gemm->load_param(pd);
            Mat weights[2];
            weights[0] = v_weight_data;
            weights[1] = v_bias_data;
            v_gemm->load_model(ModelBinFromMatArray(weights));
            v_gemm->create_pipeline(optopt);

            if (optopt.lightmode)
            {
                v_weight_data.release();
                v_bias_data.release();
            }
        }

        {
            o_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
            ncnn::ParamDict pd;
            pd.set(2, 0);         // transA
            pd.set(3, 1);         // transB
            pd.set(4, 0);         // constantA
            pd.set(5, 1);         // constantB
            pd.set(6, 1);         // constantC
            pd.set(7, 0);         // M = outch
            pd.set(8, embed_dim); // N = size
            pd.set(9, embed_dim); // K = maxk*inch
            pd.set(10, 4);        // constant_broadcast_type_C = null
            pd.set(11, 0);        // output_N1M
            o_gemm->load_param(pd);
            Mat weights[2];
            weights[0] = out_weight_data;
            weights[1] = out_bias_data;
            o_gemm->load_model(ModelBinFromMatArray(weights));
            o_gemm->create_pipeline(optopt);

            if (optopt.lightmode)
            {
                out_weight_data.release();
                out_bias_data.release();
            }
        }

        {
            qk_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
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
            pd.set(12, 1);  // output_elempack
            qk_gemm->load_param(pd);
            qk_gemm->load_model(ModelBinFromMatArray(0));
            Option opt1 = optopt;
            opt1.num_threads = 1;
            qk_gemm->create_pipeline(opt1);
        }

        {
            qkv_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
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
            pd.set(12, 1);  // output_elempack
            qkv_gemm->load_param(pd);
            qkv_gemm->load_model(ModelBinFromMatArray(0));
            Option opt1 = optopt;
            opt1.num_threads = 1;
            qkv_gemm->create_pipeline(opt1);
        }

        return 0;
    }
#endif

    Option optopt = optn;
    optopt.use_bf16_storage = false;
    optopt.use_fp16_arithmetic = false;
    optopt.use_fp16_packed = false;
    optopt.use_fp16_storage = false;

    {
        const int embed_dim_per_head = embed_dim / num_head;
        const float inv_sqrt_embed_dim_per_head = 1.f / sqrt(embed_dim_per_head);

        q_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
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
        pd.set(12, 1);        // output_elempack
        q_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = q_weight_data;
        weights[1] = q_bias_data;
        q_gemm->load_model(ModelBinFromMatArray(weights));
        q_gemm->create_pipeline(optopt);

        if (optopt.lightmode)
        {
            q_weight_data.release();
            q_bias_data.release();
        }
    }

    {
        k_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
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
        pd.set(12, 1);        // output_elempack
        k_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = k_weight_data;
        weights[1] = k_bias_data;
        k_gemm->load_model(ModelBinFromMatArray(weights));
        k_gemm->create_pipeline(optopt);

        if (optopt.lightmode)
        {
            k_weight_data.release();
            k_bias_data.release();
        }
    }

    {
        v_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
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
        pd.set(12, 1);        // output_elempack
        v_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = v_weight_data;
        weights[1] = v_bias_data;
        v_gemm->load_model(ModelBinFromMatArray(weights));
        v_gemm->create_pipeline(optopt);

        if (optopt.lightmode)
        {
            v_weight_data.release();
            v_bias_data.release();
        }
    }

    {
        o_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 1);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M = outch
        pd.set(8, embed_dim); // N = size
        pd.set(9, embed_dim); // K = maxk*inch
        pd.set(10, 4);        // constant_broadcast_type_C = null
        pd.set(11, 0);        // output_N1M
        o_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = out_weight_data;
        weights[1] = out_bias_data;
        o_gemm->load_model(ModelBinFromMatArray(weights));
        o_gemm->create_pipeline(optopt);

        if (optopt.lightmode)
        {
            out_weight_data.release();
            out_bias_data.release();
        }
    }

    {
        qk_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
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
        pd.set(12, 1);  // output_elempack
        qk_gemm->load_param(pd);
        qk_gemm->load_model(ModelBinFromMatArray(0));
        Option opt1 = optopt;
        opt1.num_threads = 1;
        qk_gemm->create_pipeline(opt1);
    }

    {
        qkv_gemm = ncnn::create_layer(ncnn::LayerType::Gemm);
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
        pd.set(12, 1);  // output_elempack
        qkv_gemm->load_param(pd);
        qkv_gemm->load_model(ModelBinFromMatArray(0));
        Option opt1 = optopt;
        opt1.num_threads = 1;
        qkv_gemm->create_pipeline(opt1);
    }

    return 0;
}

int MultiHeadAttention_arm::destroy_pipeline(const Option& opt)
{
    Option optn = opt;
    optn.use_bf16_storage = false;

    Option opt32 = optn;
    opt32.use_bf16_storage = false;
    opt32.use_fp16_arithmetic = false;
    opt32.use_fp16_packed = false;
    opt32.use_fp16_storage = false;

    if (cvtfp16_to_fp32)
    {
        cvtfp16_to_fp32->destroy_pipeline(optn);
        delete cvtfp16_to_fp32;
        cvtfp16_to_fp32 = 0;
    }
    if (cvtfp32_to_fp16)
    {
        cvtfp32_to_fp16->destroy_pipeline(optn);
        delete cvtfp32_to_fp16;
        cvtfp32_to_fp16 = 0;
    }

    if (qk_softmax)
    {
        qk_softmax->destroy_pipeline(opt32);
        delete qk_softmax;
        qk_softmax = 0;
    }

    if (permute_wch)
    {
        permute_wch->destroy_pipeline(opt32);
        delete permute_wch;
        permute_wch = 0;
    }

#if NCNN_ARM82
    if (support_fp16_storage && optn.use_fp16_packed)
    {
        Option optopt = optn;

        if (q_gemm)
        {
            q_gemm->destroy_pipeline(optopt);
            delete q_gemm;
            q_gemm = 0;
        }

        if (k_gemm)
        {
            k_gemm->destroy_pipeline(optopt);
            delete k_gemm;
            k_gemm = 0;
        }

        if (v_gemm)
        {
            v_gemm->destroy_pipeline(optopt);
            delete v_gemm;
            v_gemm = 0;
        }

        if (o_gemm)
        {
            o_gemm->destroy_pipeline(optopt);
            delete o_gemm;
            o_gemm = 0;
        }

        if (qk_gemm)
        {
            qk_gemm->destroy_pipeline(optopt);
            delete qk_gemm;
            qk_gemm = 0;
        }

        if (qkv_gemm)
        {
            qkv_gemm->destroy_pipeline(optopt);
            delete qkv_gemm;
            qkv_gemm = 0;
        }

        return 0;
    }
#endif

    Option optopt = optn;
    optopt.use_bf16_storage = false;
    optopt.use_fp16_arithmetic = false;
    optopt.use_fp16_packed = false;
    optopt.use_fp16_storage = false;

    if (q_gemm)
    {
        q_gemm->destroy_pipeline(optopt);
        delete q_gemm;
        q_gemm = 0;
    }

    if (k_gemm)
    {
        k_gemm->destroy_pipeline(optopt);
        delete k_gemm;
        k_gemm = 0;
    }

    if (v_gemm)
    {
        v_gemm->destroy_pipeline(optopt);
        delete v_gemm;
        v_gemm = 0;
    }

    if (o_gemm)
    {
        o_gemm->destroy_pipeline(optopt);
        delete o_gemm;
        o_gemm = 0;
    }

    if (qk_gemm)
    {
        qk_gemm->destroy_pipeline(optopt);
        delete qk_gemm;
        qk_gemm = 0;
    }

    if (qkv_gemm)
    {
        qkv_gemm->destroy_pipeline(optopt);
        delete qkv_gemm;
        qkv_gemm = 0;
    }

    return 0;
}

int MultiHeadAttention_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& q_blob = bottom_blobs[0];
    const Mat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const Mat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs.size() == 2 ? k_blob : bottom_blobs[2];

    const int embed_dim_per_head = embed_dim / num_head;
    const int src_seqlen = q_blob.h * q_blob.elempack;
    const int dst_seqlen = k_blob.h * k_blob.elempack;

    const int elembits = q_blob.elembits();

    Option optn = opt;
    optn.use_bf16_storage = false;

    Option opt32 = optn;
    opt32.use_bf16_storage = false;
    opt32.use_fp16_arithmetic = false;
    opt32.use_fp16_packed = false;
    opt32.use_fp16_storage = false;

#if NCNN_ARM82
    if (support_fp16_storage && optn.use_fp16_packed && elembits == 16)
    {
        Mat q_affine, k_affine, v_affine;
        Mat qk_cross(dst_seqlen, src_seqlen * num_head, 2u, optn.blob_allocator);
        Mat qkv_cross(embed_dim_per_head, src_seqlen, num_head, 2u, optn.blob_allocator);
        Mat qkv_wch_fp16(embed_dim, src_seqlen, 2u, opt.blob_allocator);

        q_gemm->forward(q_blob, q_affine, optn);
        k_gemm->forward(k_blob, k_affine, optn);

        #pragma omp parallel for num_threads(optn.num_threads)
        for (int i = 0; i < num_head; i++)
        {
            std::vector<Mat> qk_bottom_blobs(2);
            qk_bottom_blobs[0] = q_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
            qk_bottom_blobs[1] = k_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
            std::vector<Mat> qk_top_blobs(1);
            qk_top_blobs[0] = qk_cross.row_range(i * src_seqlen, src_seqlen);
            Option opt1 = optn;
            opt1.num_threads = 1;
            qk_gemm->forward(qk_bottom_blobs, qk_top_blobs, opt1);
        }

        q_affine.release();
        k_affine.release();

        Mat qk_cross_fp32, qk_cross_fp32_fp16;
        cvtfp16_to_fp32->forward(qk_cross, qk_cross_fp32, optn);
        qk_softmax->forward_inplace(qk_cross_fp32, opt32);
        cvtfp32_to_fp16->forward(qk_cross_fp32, qk_cross_fp32_fp16, optn);

        qk_cross.release();
        qk_cross_fp32.release();

        v_gemm->forward(v_blob, v_affine, optn);

        #pragma omp parallel for num_threads(optn.num_threads)
        for (int i = 0; i < num_head; i++)
        {
            std::vector<Mat> qkv_bottom_blobs(2);
            qkv_bottom_blobs[0] = qk_cross_fp32_fp16.row_range(i * src_seqlen, src_seqlen);
            qkv_bottom_blobs[1] = v_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
            std::vector<Mat> qkv_top_blobs(1);
            qkv_top_blobs[0] = qkv_cross.channel(i);
            Option opt1 = optn;
            opt1.num_threads = 1;
            qkv_gemm->forward(qkv_bottom_blobs, qkv_top_blobs, opt1);
        }

        qk_cross_fp32_fp16.release();
        v_affine.release();

        // permute + reshape
        #pragma omp parallel for num_threads(optn.num_threads)
        for (int q = 0; q < src_seqlen; q++)
        {
            __fp16* outptr = qkv_wch_fp16.row<__fp16>(q);
            for (int i = 0; i < num_head; i++)
            {
                __fp16* ptr = qkv_cross.channel(i).row<__fp16>(q);
                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    *outptr++ = ptr[j];
                }
            }
        }

        qkv_cross.release();

        o_gemm->forward(qkv_wch_fp16, top_blobs[0], optn);

        return 0;
    }
#endif

    Mat q_affine;
    q_gemm->forward(q_blob, q_affine, opt32);

    Mat k_affine;
    k_gemm->forward(k_blob, k_affine, opt32);

    Mat qk_cross(dst_seqlen, src_seqlen * num_head, 4u, opt32.blob_allocator);
    #pragma omp parallel for num_threads(opt32.num_threads)
    for (int i = 0; i < num_head; i++)
    {
        std::vector<Mat> qk_bottom_blobs(2);
        qk_bottom_blobs[0] = q_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
        qk_bottom_blobs[1] = k_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
        std::vector<Mat> qk_top_blobs(1);
        qk_top_blobs[0] = qk_cross.row_range(i * src_seqlen, src_seqlen);
        Option opt1 = opt32;
        opt1.num_threads = 1;
        qk_gemm->forward(qk_bottom_blobs, qk_top_blobs, opt1);
    }

    q_affine.release();
    k_affine.release();

    qk_softmax->forward_inplace(qk_cross, opt32);

    Mat v_affine;
    v_gemm->forward(v_blob, v_affine, opt32);

    Mat qkv_cross(embed_dim_per_head, src_seqlen, num_head, 4u, opt32.blob_allocator);
    #pragma omp parallel for num_threads(opt32.num_threads)
    for (int i = 0; i < num_head; i++)
    {
        std::vector<Mat> qkv_bottom_blobs(2);
        qkv_bottom_blobs[0] = qk_cross.row_range(i * src_seqlen, src_seqlen);
        qkv_bottom_blobs[1] = v_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
        std::vector<Mat> qkv_top_blobs(1);
        qkv_top_blobs[0] = qkv_cross.channel(i);
        Option opt1 = opt32;
        opt1.num_threads = 1;
        qkv_gemm->forward(qkv_bottom_blobs, qkv_top_blobs, opt1);
    }

    qk_cross.release();
    v_affine.release();

    {
        Mat qkv_wch;
        permute_wch->forward(qkv_cross, qkv_wch, opt32);

        qkv_cross.release();

        qkv_wch = qkv_wch.reshape(embed_dim, src_seqlen);

        o_gemm->forward(qkv_wch, top_blobs[0], opt32);
    }

    return 0;
}

} // namespace ncnn
