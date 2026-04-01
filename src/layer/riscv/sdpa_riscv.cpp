// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_riscv.h"

#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif
#include "riscv_usability.h"

namespace ncnn {

SDPA_riscv::SDPA_riscv()
{
    support_packing = true;

    qk_gemm = 0;
    qkv_gemm = 0;
    qk_softmax = 0;
}

int SDPA_riscv::create_pipeline(const Option& opt)
{
    {
        qk_softmax = ncnn::create_layer_cpu(ncnn::LayerType::Softmax);
        ncnn::ParamDict pd;
        pd.set(0, -1); // axis
        pd.set(1, 1);
        qk_softmax->load_param(pd);
        qk_softmax->load_model(ModelBinFromMatArray(0));
        qk_softmax->create_pipeline(opt);
    }

    // Q * K^T
    {
        qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;

        pd.set(0, 1.f);                 // alpha (will be set in forward)
        pd.set(1, 0.f);                 // beta
        pd.set(2, 0);                   // transA (Q: Seq x Embed)
        pd.set(3, 1);                   // transB (K: Seq x Embed -> K^T: Embed x Seq) => Q * K^T
        pd.set(4, 0);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, 1);                   // constantC (None)
        pd.set(7, 0);                   // M
        pd.set(8, 0);                   // N
        pd.set(9, 0);                   // K
        pd.set(10, -1);                 // constant_broadcast_type_C
        pd.set(11, 0);                  // output_N1M
        pd.set(12, 1);                  // output_elempack
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        qk_gemm->load_param(pd);
        qk_gemm->load_model(ModelBinFromMatArray(0));
        Option opt1 = opt;
        opt1.num_threads = 1;
        qk_gemm->create_pipeline(opt1);
    }

    // Attn * V
    {
        qkv_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(0, 1.f); // alpha
        pd.set(1, 1.f); // beta
        pd.set(2, 0);   // transA (Attn: Seq x Seq)
        pd.set(3, 0);   // transB (V: Seq x Embed) => Attn * V
        pd.set(4, 0);   // constantA
        pd.set(5, 0);   // constantB
        pd.set(6, 1);   // constantC (None)
        pd.set(7, 0);   // M
        pd.set(8, 0);   // N
        pd.set(9, 0);   // K
        pd.set(10, -1); // constant_broadcast_type_C
        pd.set(11, 0);  // output_N1M
        pd.set(12, 1);  // output_elempack
        pd.set(14, 0);  // output_transpose
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        qkv_gemm->load_param(pd);
        qkv_gemm->load_model(ModelBinFromMatArray(0));
        Option opt1 = opt;
        opt1.num_threads = 1;
        qkv_gemm->create_pipeline(opt1);
    }

    return 0;
}

int SDPA_riscv::destroy_pipeline(const Option& opt)
{
    if (qk_softmax)
    {
        qk_softmax->destroy_pipeline(opt);
        delete qk_softmax;
        qk_softmax = 0;
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

    return 0;
}

int SDPA_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& _opt) const
{
    Option opt = _opt;
    const Mat& query = bottom_blobs[0];
    const Mat& cur_key = bottom_blobs[1];
    const Mat& cur_value = bottom_blobs[2];
    const Mat& attn_mask_blob = attn_mask ? bottom_blobs[3] : Mat();
    const Mat& past_key = kv_cache ? bottom_blobs[attn_mask ? 4 : 3] : Mat();
    const Mat& past_value = kv_cache ? bottom_blobs[attn_mask ? 5 : 4] : Mat();

    const int embed_dim = query.w;
    const int src_seqlen = query.h;
    const int num_heads = query.c;
    const int cur_seqlen = cur_key.h;
    const int num_group = cur_key.c;
    const int out_embed_dim = cur_value.w;
    const int past_seqlen = kv_cache ? past_key.h : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;
    const int elempack = query.elempack;

    if (elempack > 1)
    {
        // Fallback for packed data
        // TODO: Implement optimized RVV paths for group=2 with elempack=2,4,8, and group=4 with elempack=4
        
        // Unpack input blobs
        std::vector<Mat> bottom_blobs_unpacked = bottom_blobs;
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;

        Mat query_unpacked;
        convert_packing(query, query_unpacked, 1, opt_unpack);
        bottom_blobs_unpacked[0] = query_unpacked;

        Mat cur_key_unpacked;
        convert_packing(cur_key, cur_key_unpacked, 1, opt_unpack);
        bottom_blobs_unpacked[1] = cur_key_unpacked;

        Mat cur_value_unpacked;
        convert_packing(cur_value, cur_value_unpacked, 1, opt_unpack);
        bottom_blobs_unpacked[2] = cur_value_unpacked;

        if (attn_mask)
        {
            Mat attn_mask_unpacked;
            convert_packing(attn_mask_blob, attn_mask_unpacked, 1, opt_unpack);
            bottom_blobs_unpacked[3] = attn_mask_unpacked;
        }

        if (kv_cache)
        {
            Mat past_key_unpacked;
            convert_packing(past_key, past_key_unpacked, 1, opt_unpack);
            bottom_blobs_unpacked[attn_mask ? 4 : 3] = past_key_unpacked;

            Mat past_value_unpacked;
            convert_packing(past_value, past_value_unpacked, 1, opt_unpack);
            bottom_blobs_unpacked[attn_mask ? 5 : 4] = past_value_unpacked;
        }

        std::vector<Mat> top_blobs_unpacked(top_blobs.size());
        int ret = SDPA::forward(bottom_blobs_unpacked, top_blobs_unpacked, opt);
        if (ret != 0)
            return ret;

        // Repack output blobs
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            if (top_blobs_unpacked[i].empty())
                continue;
            convert_packing(top_blobs_unpacked[i], top_blobs[i], elempack, opt);
        }

        return 0;
    }

    Mat key;
    if (past_seqlen > 0)
    {
        key.create(embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
        if (key.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_group; q++)
        {
            const Mat past_key_head = past_key.channel(q);
            const Mat cur_key_head = cur_key.channel(q);
            Mat key_head = key.channel(q);

            memcpy(key_head.row(0), past_key_head, embed_dim * past_seqlen * sizeof(float));
            memcpy(key_head.row(past_seqlen), cur_key_head, embed_dim * cur_seqlen * sizeof(float));
        }
    }
    else
    {
        key = cur_key;
    }

    Mat value;
    if (past_seqlen > 0)
    {
        value.create(out_embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
        if (value.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_group; q++)
        {
            const Mat past_value_head = past_value.channel(q);
            const Mat cur_value_head = cur_value.channel(q);
            Mat value_head = value.channel(q);

            memcpy(value_head.row(0), past_value_head, out_embed_dim * past_seqlen * sizeof(float));
            memcpy(value_head.row(past_seqlen), cur_value_head, out_embed_dim * cur_seqlen * sizeof(float));
        }
    }
    else
    {
        value = cur_value;
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_heads_per_group = num_heads / num_group;

    Mat qk_cross(dst_seqlen, src_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (qk_cross.empty())
        return -100;

    std::vector<int> retqks(num_heads);

    float _scale = scale;
    if (_scale == 0.f)
    {
        _scale = 1.f / sqrt(embed_dim);
    }

    // Create local Gemm if scale is dynamic or different from 1.f
    Layer* _qk_gemm = qk_gemm;
    bool local_gemm = false;
    if (_scale != 1.f)
    {
        _qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(0, _scale);              // alpha
        pd.set(1, 0.f);                 // beta
        pd.set(2, 0);                   // transA
        pd.set(3, 1);                   // transB
        pd.set(4, 0);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, 1);                   // constantC (None)
        pd.set(7, 0);                   // M
        pd.set(8, 0);                   // N
        pd.set(9, 0);                   // K
        pd.set(10, -1);                 // constant_broadcast_type_C
        pd.set(11, 0);                  // output_N1M
        pd.set(12, 1);                  // output_elempack
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        _qk_gemm->load_param(pd);
        _qk_gemm->load_model(ModelBinFromMatArray(0));
        Option opt1 = opt;
        opt1.num_threads = 1;
        _qk_gemm->create_pipeline(opt1);
        local_gemm = true;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        // 1. Q * K^T
        const Mat q_head = query.channel(i);
        const Mat k_head = key.channel(i / num_heads_per_group);
        Mat score_head = qk_cross.channel(i);

        for (int j = 0; j < src_seqlen; j++)
        {
            const float* qptr = q_head.row(j);
            float* outptr = score_head.row(j);
            const float* mptr_row = 0;
            if (attn_mask)
            {
                const Mat& maskm = attn_mask_blob.c > 1 ? attn_mask_blob.channel(i) : attn_mask_blob;
                mptr_row = maskm.row(j);
            }

            for (int k = 0; k < dst_seqlen; k++)
            {
                const float* kptr = k_head.row(k);
                float sum = 0.f;

#if __riscv_vector
                size_t vlmax = __riscv_vsetvlmax_e32m8();
                vfloat32m8_t _sum_v = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
                int l = 0;
                for (; l < embed_dim; )
                {
                    size_t vl = __riscv_vsetvl_e32m8(embed_dim - l);
                    vfloat32m8_t _q = __riscv_vle32_v_f32m8(qptr + l, vl);
                    vfloat32m8_t _k = __riscv_vle32_v_f32m8(kptr + l, vl);
                    _sum_v = __riscv_vfmacc_vv_f32m8(_sum_v, _q, _k, vl);
                    l += vl;
                }
                vfloat32m1_t _sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
                _sum_scalar = __riscv_vfredusum_vs_f32m8_f32m1(_sum_v, _sum_scalar, vlmax);
                sum = __riscv_vfmv_f_s_f32m1_f32(_sum_scalar);
#else
                for (int l = 0; l < embed_dim; l++)
                {
                    sum += qptr[l] * kptr[l];
                }
#endif
                outptr[k] = sum * _scale;
                if (attn_mask)
                    outptr[k] += mptr_row[k];
            }
        }
    }

    for (int i = 0; i < num_heads; i++)
    {
        if (retqks[i] != 0)
            return retqks[i];
    }

    if (local_gemm)
    {
        Option opt1 = opt;
        opt1.num_threads = 1;
        _qk_gemm->destroy_pipeline(opt1);
        delete _qk_gemm;
    }

    // 2. Softmax
    int retqk = qk_softmax->forward_inplace(qk_cross, opt);
    if (retqk != 0)
        return retqk;

    // 3. Attn * V
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        const Mat score_head = qk_cross.channel(i);
        const Mat v_head = value.channel(i / num_heads_per_group);
        Mat out_head = top_blob.channel(i);

        for (int j = 0; j < src_seqlen; j++)
        {
            const float* qkptr = score_head.row(j);
            float* outptr = out_head.row(j);

            for (int k = 0; k < out_embed_dim; k++)
            {
                float sum = 0.f;
#if __riscv_vector
                size_t vlmax = __riscv_vsetvlmax_e32m8();
                vfloat32m8_t _sum_v = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
                int l = 0;
                for (; l < dst_seqlen; )
                {
                    size_t vl = __riscv_vsetvl_e32m8(dst_seqlen - l);
                    vfloat32m8_t _qk = __riscv_vle32_v_f32m8(qkptr + l, vl);
                    vfloat32m8_t _v = __riscv_vlse32_v_f32m8(v_head.row(l) + k, out_embed_dim * sizeof(float), vl);
                    _sum_v = __riscv_vfmacc_vv_f32m8(_sum_v, _qk, _v, vl);
                    l += vl;
                }
                vfloat32m1_t _sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
                _sum_scalar = __riscv_vfredusum_vs_f32m8_f32m1(_sum_v, _sum_scalar, vlmax);
                sum = __riscv_vfmv_f_s_f32m1_f32(_sum_scalar);
#else
                for (int l = 0; l < dst_seqlen; l++)
                {
                    sum += qkptr[l] * v_head.row(l)[k];
                }
#endif
                outptr[k] = sum;
            }
        }
    }

    // for (int i = 0; i < num_heads; i++)
    // {
    //     if (retqkvs[i] != 0)
    //         return retqkvs[i];
    // }

    if (kv_cache)
    {
        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}

} // namespace ncnn

