// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_mips.h"

#include "layer_type.h"

namespace ncnn {

SDPA_mips::SDPA_mips()
{
#if NCNN_BF16
    support_bf16_storage = true;
#endif

    qk_gemm = 0;
    qkv_gemm = 0;
    qk_softmax = 0;
}

int SDPA_mips::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
        support_bf16_storage = false;
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
        pd.set(13, 1);                  // output_elemtype = fp32
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
        pd.set(13, 1);  // output_elemtype = fp32
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

int SDPA_mips::destroy_pipeline(const Option& _opt)
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

int SDPA_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& _opt) const
{
    Option opt = _opt;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }

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

    const size_t elemsize = query.elemsize;

    Mat key;
    if (past_seqlen > 0)
    {
        key.create(embed_dim, dst_seqlen, num_group, elemsize, opt.blob_allocator);
        if (key.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_group; q++)
        {
            const Mat past_key_head = past_key.channel(q);
            const Mat cur_key_head = cur_key.channel(q);
            Mat key_head = key.channel(q);

            memcpy(key_head.row(0), past_key_head, embed_dim * past_seqlen * elemsize);
            memcpy(key_head.row(past_seqlen), cur_key_head, embed_dim * cur_seqlen * elemsize);
        }
    }
    else
    {
        key = cur_key;
    }

    Mat value;
    if (past_seqlen > 0)
    {
        value.create(out_embed_dim, dst_seqlen, num_group, elemsize, opt.blob_allocator);
        if (value.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_group; q++)
        {
            const Mat past_value_head = past_value.channel(q);
            const Mat cur_value_head = cur_value.channel(q);
            Mat value_head = value.channel(q);

            memcpy(value_head.row(0), past_value_head, out_embed_dim * past_seqlen * elemsize);
            memcpy(value_head.row(past_seqlen), cur_value_head, out_embed_dim * cur_seqlen * elemsize);
        }
    }
    else
    {
        value = cur_value;
    }

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
        pd.set(13, 1);                  // output_elemtype = fp32
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

    Mat value_fp32 = value;
#if NCNN_BF16
    if (opt.use_bf16_storage && value.elembits() == 16)
    {
        // qkv_gemm need fp32 inputs
        cast_bfloat16_to_float32(value, value_fp32, opt);
        if (value_fp32.empty())
            return -100;
    }
#endif

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // 3. Attn * V
    std::vector<int> retqkvs(num_heads);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        std::vector<Mat> qkv_bottom_blobs(2);
        qkv_bottom_blobs[0] = qk_cross.channel(i);                         // Attn: [DstSeq, Seq]
        qkv_bottom_blobs[1] = value_fp32.channel(i / num_heads_per_group); // V: [DstSeq, OutEmbed]

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

    value_fp32.release();

    if (kv_cache)
    {
        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}

} // namespace ncnn
