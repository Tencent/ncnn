// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif

#include "layer_type.h"

namespace ncnn {

SDPA_riscv::SDPA_riscv()
{
    qk_gemm = 0;
    qkv_gemm = 0;
    qk_softmax = 0;
}

int SDPA_riscv::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }

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
    if (scale != 0.f)
    {
        qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;

        pd.set(0, scale);               // alpha
        pd.set(1, 1.f / scale);         // beta
        pd.set(2, 0);                   // transA (Q: Seq x Embed)
        pd.set(3, 1);                   // transB (K: Seq x Embed -> K^T: Embed x Seq) => Q * K^T
        pd.set(4, 0);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, attn_mask ? 0 : 1);   // constantC (if mask exists, use it)
        pd.set(7, 0);                   // M
        pd.set(8, 0);                   // N
        pd.set(9, 0);                   // K
        pd.set(10, attn_mask ? 3 : -1); // constant_broadcast_type_C (MxN)
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

int SDPA_riscv::destroy_pipeline(const Option& _opt)
{
    Option opt = _opt;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }

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
    const Mat& query = bottom_blobs[0];
    int elempack = query.elempack;

    if (elempack > 1)
    {
        // fallback: unpack -> forward -> repack
        Option opt = _opt;
        opt.blob_allocator = _opt.workspace_allocator;

        std::vector<Mat> unpacked_bottom_blobs(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            if (bottom_blobs[i].empty()) continue;

            if (bottom_blobs[i].elempack == 1)
            {
                unpacked_bottom_blobs[i] = bottom_blobs[i];
            }
            else
            {
                ncnn::Layer* packing = ncnn::create_layer_cpu(ncnn::LayerType::Packing);
                ncnn::ParamDict pd;
                pd.set(0, 1); // out_elempack
                packing->load_param(pd);
                packing->forward(bottom_blobs[i], unpacked_bottom_blobs[i], opt);
                delete packing;
            }
        }

        std::vector<Mat> unpacked_top_blobs(top_blobs.size());

        // call forward with elempack=1
        int ret = forward(unpacked_bottom_blobs, unpacked_top_blobs, _opt);
        if (ret != 0) return ret;

        // repack outputs
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            if (unpacked_top_blobs[i].empty()) continue;

            ncnn::Layer* packing = ncnn::create_layer_cpu(ncnn::LayerType::Packing);
            ncnn::ParamDict pd;
            pd.set(0, elempack); // out_elempack
            packing->load_param(pd);
            packing->forward(unpacked_top_blobs[i], top_blobs[i], _opt); // use original allocator for output
            delete packing;
        }

        return 0;
    }

    Option opt = _opt;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }

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

    // Dynamic Scale Calculation and Beta Correction
    Layer* _qk_gemm = qk_gemm;
    if (scale == 0.f)
    {
        float _scale = 1.f / sqrt(embed_dim);

        _qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;

        pd.set(0, _scale);              // alpha
        pd.set(1, 1.f / _scale);        // beta
        pd.set(2, 0);                   // transA (Q: Seq x Embed)
        pd.set(3, 1);                   // transB (K: Seq x Embed -> K^T: Embed x Seq) => Q * K^T
        pd.set(4, 0);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, attn_mask ? 0 : 1);   // constantC (if mask exists, use it)
        pd.set(7, 0);                   // M
        pd.set(8, 0);                   // N
        pd.set(9, 0);                   // K
        pd.set(10, attn_mask ? 3 : -1); // constant_broadcast_type_C (MxN)
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
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        // 1. Q * K^T
        std::vector<Mat> qk_bottom_blobs;
        qk_bottom_blobs.push_back(query.channel(i));                     // Q: [Seq, Embed]
        qk_bottom_blobs.push_back(key.channel(i / num_heads_per_group)); // K: [DstSeq, Embed]

        if (attn_mask)
        {
            // Ensure mask is 2D for Gemm auto-broadcast detection
            Mat maskm = attn_mask_blob;
            if (maskm.dims == 3)
            {
                // If c > 1, pick i-th head mask. If c == 1, pick 0-th (broadcast)
                maskm = maskm.channel(maskm.c > 1 ? i : 0);
            }
            qk_bottom_blobs.push_back(maskm);
        }

        std::vector<Mat> qk_top_blobs(1);
        qk_top_blobs[0] = qk_cross.channel(i);

        Option opt1 = opt;
        opt1.num_threads = 1;
        opt1.blob_allocator = qk_cross.allocator;
        retqks[i] = _qk_gemm->forward(qk_bottom_blobs, qk_top_blobs, opt1);
    }

    if (scale == 0.f)
    {
        Option opt1 = opt;
        opt1.num_threads = 1;
        _qk_gemm->destroy_pipeline(opt1);

        delete _qk_gemm;
        _qk_gemm = 0;
    }

    for (int i = 0; i < num_heads; i++)
    {
        if (retqks[i] != 0)
            return retqks[i];
    }

    // 2. Softmax
    int retqk = qk_softmax->forward_inplace(qk_cross, opt);
    if (retqk != 0)
        return retqk;

    // 3. Attn * V
    std::vector<int> retqkvs(num_heads);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        std::vector<Mat> qkv_bottom_blobs(2);
        qkv_bottom_blobs[0] = qk_cross.channel(i);                    // Attn: [DstSeq, Seq]
        qkv_bottom_blobs[1] = value.channel(i / num_heads_per_group); // V: [DstSeq, OutEmbed]

        std::vector<Mat> qkv_top_blobs(1);
        qkv_top_blobs[0] = top_blob.channel(i); // Output

        Option opt1 = opt;
        opt1.num_threads = 1;
        retqkvs[i] = qkv_gemm->forward(qkv_bottom_blobs, qkv_top_blobs, opt1);
    }

    for (int i = 0; i < num_heads; i++)
    {
        if (retqkvs[i] != 0)
            return retqkvs[i];
    }

    if (kv_cache)
    {
        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}

} // namespace ncnn
