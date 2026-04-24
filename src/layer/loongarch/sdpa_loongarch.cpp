// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_loongarch.h"

#include "layer_type.h"

namespace ncnn {

SDPA_loongarch::SDPA_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif

    qk_gemm = 0;
    qkv_gemm = 0;
    qk_softmax = 0;
}

static int unpack_or_cast_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    if (src.empty())
    {
        dst = src;
        return 0;
    }

    Mat unpacked = src;
    if (src.elempack != 1)
    {
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;

        convert_packing(src, unpacked, 1, opt_unpack);
        if (unpacked.empty())
            return -100;
    }

#if NCNN_BF16
    if (unpacked.elembits() == 16)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = opt.workspace_allocator;

        cast_bfloat16_to_float32(unpacked, dst, opt_cast);
        if (dst.empty())
            return -100;
        return 0;
    }
#endif

    dst = unpacked;
    return 0;
}

static bool is_scalar_fp32(const Mat& m)
{
    return m.empty() || (m.elempack == 1 && m.elembits() == 32);
}

static int restore_layout_from_reference(const Mat& src, Mat& dst, const Mat& reference, const Option& opt)
{
    if (src.empty())
    {
        dst = src;
        return 0;
    }

    Mat tmp = src;
    if (opt.use_packing_layout && reference.elempack != 1)
    {
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        Mat packed;
        convert_packing(tmp, packed, reference.elempack, opt_pack);
        if (packed.empty())
            return -100;

        tmp = packed;
    }

#if NCNN_BF16
    if (opt.use_bf16_storage && reference.elembits() == 16)
    {
        cast_float32_to_bfloat16(tmp, dst, opt);
        if (dst.empty())
            return -100;

        return 0;
    }
#endif

    dst = tmp;
    return 0;
}

int SDPA_loongarch::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
        support_bf16_storage = false;
    }
    else
    {
        return 0;
    }

    Option opt_fp32 = opt;
    opt_fp32.use_fp16_packed = false;
    opt_fp32.use_fp16_storage = false;
    opt_fp32.use_fp16_arithmetic = false;
    opt_fp32.use_bf16_packed = false;
    opt_fp32.use_bf16_storage = false;

    {
        qk_softmax = ncnn::create_layer_cpu(ncnn::LayerType::Softmax);
        ncnn::ParamDict pd;
        pd.set(0, -1); // axis
        pd.set(1, 1);
        qk_softmax->load_param(pd);
        qk_softmax->load_model(ModelBinFromMatArray(0));
        qk_softmax->create_pipeline(opt_fp32);
    }

    if (scale != 0.f)
    {
        qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;

        pd.set(0, scale);               // alpha
        pd.set(1, 1.f / scale);         // beta
        pd.set(2, 0);                   // transA
        pd.set(3, 1);                   // transB
        pd.set(4, 0);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, attn_mask ? 0 : 1);   // constantC
        pd.set(7, 0);                   // M
        pd.set(8, 0);                   // N
        pd.set(9, 0);                   // K
        pd.set(10, attn_mask ? 3 : -1); // constant_broadcast_type_C
        pd.set(11, 0);                  // output_N1M
        pd.set(12, 1);                  // output_elempack
        pd.set(13, 1);                  // output_elemtype = fp32
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        qk_gemm->load_param(pd);
        qk_gemm->load_model(ModelBinFromMatArray(0));
        Option opt1 = opt_fp32;
        opt1.num_threads = 1;
        qk_gemm->create_pipeline(opt1);
    }

    {
        qkv_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(0, 1.f); // alpha
        pd.set(1, 1.f); // beta
        pd.set(2, 0);   // transA
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
        pd.set(13, 1);  // output_elemtype = fp32
        pd.set(14, 0);  // output_transpose
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        qkv_gemm->load_param(pd);
        qkv_gemm->load_model(ModelBinFromMatArray(0));
        Option opt1 = opt_fp32;
        opt1.num_threads = 1;
        qkv_gemm->create_pipeline(opt1);
    }

    return 0;
}

int SDPA_loongarch::destroy_pipeline(const Option& _opt)
{
    Option opt = _opt;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }
    else
    {
        return 0;
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

int SDPA_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& _opt) const
{
    if (!int8_scale_term)
    {
        std::vector<Mat> bottom_blobs_fp32(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            if (unpack_or_cast_to_float32(bottom_blobs[i], bottom_blobs_fp32[i], _opt) != 0)
                return -100;
        }

        Option opt_fp32 = _opt;
        opt_fp32.use_packing_layout = false;
        opt_fp32.use_fp16_packed = false;
        opt_fp32.use_fp16_storage = false;
        opt_fp32.use_fp16_arithmetic = false;
        opt_fp32.use_bf16_packed = false;
        opt_fp32.use_bf16_storage = false;

        std::vector<Mat> top_blobs_fp32(top_blobs.size());
        int ret = SDPA::forward(bottom_blobs_fp32, top_blobs_fp32, opt_fp32);
        if (ret != 0)
            return ret;

        if (restore_layout_from_reference(top_blobs_fp32[0], top_blobs[0], bottom_blobs[0], _opt) != 0)
            return -100;

        if (kv_cache)
        {
            if (restore_layout_from_reference(top_blobs_fp32[1], top_blobs[1], bottom_blobs[1], _opt) != 0)
                return -100;
            if (restore_layout_from_reference(top_blobs_fp32[2], top_blobs[2], bottom_blobs[2], _opt) != 0)
                return -100;
        }

        return 0;
    }

    Option opt = _opt;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }

    Option opt_fp32 = opt;
    opt_fp32.use_fp16_packed = false;
    opt_fp32.use_fp16_storage = false;
    opt_fp32.use_fp16_arithmetic = false;
    opt_fp32.use_bf16_packed = false;
    opt_fp32.use_bf16_storage = false;

    const Mat& query = bottom_blobs[0];
    const Mat& cur_key = bottom_blobs[1];
    const Mat& cur_value = bottom_blobs[2];
    const Mat& attn_mask_blob = attn_mask ? bottom_blobs[3] : Mat();
    const Mat& past_key = kv_cache ? bottom_blobs[attn_mask ? 4 : 3] : Mat();
    const Mat& past_value = kv_cache ? bottom_blobs[attn_mask ? 5 : 4] : Mat();

    Mat query_fp32;
    Mat cur_key_fp32;
    Mat cur_value_fp32;
    Mat attn_mask_blob_fp32;
    Mat past_key_fp32;
    Mat past_value_fp32;
    if (unpack_or_cast_to_float32(query, query_fp32, opt) != 0
            || unpack_or_cast_to_float32(cur_key, cur_key_fp32, opt) != 0
            || unpack_or_cast_to_float32(cur_value, cur_value_fp32, opt) != 0
            || unpack_or_cast_to_float32(attn_mask_blob, attn_mask_blob_fp32, opt) != 0
            || unpack_or_cast_to_float32(past_key, past_key_fp32, opt) != 0
            || unpack_or_cast_to_float32(past_value, past_value_fp32, opt) != 0)
        return -100;

    const int embed_dim = query_fp32.w;
    const int src_seqlen = query_fp32.h;
    const int num_heads = query_fp32.c;
    const int cur_seqlen = cur_key_fp32.h;
    const int num_group = cur_key_fp32.c;
    const int out_embed_dim = cur_value_fp32.w;
    const int past_seqlen = kv_cache ? past_key_fp32.h : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    const size_t elemsize = query_fp32.elemsize;

    Mat key;
    if (past_seqlen > 0)
    {
        key.create(embed_dim, dst_seqlen, num_group, elemsize, opt_fp32.blob_allocator);
        if (key.empty())
            return -100;

        #pragma omp parallel for num_threads(opt_fp32.num_threads)
        for (int q = 0; q < num_group; q++)
        {
            const Mat past_key_head = past_key_fp32.channel(q);
            const Mat cur_key_head = cur_key_fp32.channel(q);
            Mat key_head = key.channel(q);

            memcpy(key_head.row(0), past_key_head, embed_dim * past_seqlen * elemsize);
            memcpy(key_head.row(past_seqlen), cur_key_head, embed_dim * cur_seqlen * elemsize);
        }
    }
    else
    {
        key = cur_key_fp32;
    }

    Mat value;
    if (past_seqlen > 0)
    {
        value.create(out_embed_dim, dst_seqlen, num_group, elemsize, opt_fp32.blob_allocator);
        if (value.empty())
            return -100;

        #pragma omp parallel for num_threads(opt_fp32.num_threads)
        for (int q = 0; q < num_group; q++)
        {
            const Mat past_value_head = past_value_fp32.channel(q);
            const Mat cur_value_head = cur_value_fp32.channel(q);
            Mat value_head = value.channel(q);

            memcpy(value_head.row(0), past_value_head, out_embed_dim * past_seqlen * elemsize);
            memcpy(value_head.row(past_seqlen), cur_value_head, out_embed_dim * cur_seqlen * elemsize);
        }
    }
    else
    {
        value = cur_value_fp32;
    }

    const int num_heads_per_group = num_heads / num_group;

    Mat qk_cross(dst_seqlen, src_seqlen, num_heads, 4u, opt_fp32.workspace_allocator);
    if (qk_cross.empty())
        return -100;

    std::vector<int> retqks(num_heads);

    Layer* _qk_gemm = qk_gemm;
    if (scale == 0.f)
    {
        float _scale = 1.f / sqrt(embed_dim);

        _qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;

        pd.set(0, _scale);              // alpha
        pd.set(1, 1.f / _scale);        // beta
        pd.set(2, 0);                   // transA
        pd.set(3, 1);                   // transB
        pd.set(4, 0);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, attn_mask ? 0 : 1);   // constantC
        pd.set(7, 0);                   // M
        pd.set(8, 0);                   // N
        pd.set(9, 0);                   // K
        pd.set(10, attn_mask ? 3 : -1); // constant_broadcast_type_C
        pd.set(11, 0);                  // output_N1M
        pd.set(12, 1);                  // output_elempack
        pd.set(13, 1);                  // output_elemtype = fp32
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        _qk_gemm->load_param(pd);
        _qk_gemm->load_model(ModelBinFromMatArray(0));

        Option opt1 = opt_fp32;
        opt1.num_threads = 1;
        _qk_gemm->create_pipeline(opt1);
    }

    #pragma omp parallel for num_threads(opt_fp32.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        std::vector<Mat> qk_bottom_blobs;
        qk_bottom_blobs.push_back(query_fp32.channel(i));
        qk_bottom_blobs.push_back(key.channel(i / num_heads_per_group));

        if (attn_mask)
        {
            Mat maskm = attn_mask_blob_fp32;
            if (maskm.dims == 3)
            {
                maskm = maskm.channel(maskm.c > 1 ? i : 0);
            }
            qk_bottom_blobs.push_back(maskm);
        }

        std::vector<Mat> qk_top_blobs(1);
        qk_top_blobs[0] = qk_cross.channel(i);

        Option opt1 = opt_fp32;
        opt1.num_threads = 1;
        opt1.blob_allocator = qk_cross.allocator;
        retqks[i] = _qk_gemm->forward(qk_bottom_blobs, qk_top_blobs, opt1);
    }

    if (scale == 0.f)
    {
        Option opt1 = opt_fp32;
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

    int retqk = qk_softmax->forward_inplace(qk_cross, opt_fp32);
    if (retqk != 0)
        return retqk;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt_fp32.blob_allocator);
    if (top_blob.empty())
        return -100;

    std::vector<int> retqkvs(num_heads);

    #pragma omp parallel for num_threads(opt_fp32.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        std::vector<Mat> qkv_bottom_blobs(2);
        qkv_bottom_blobs[0] = qk_cross.channel(i);
        qkv_bottom_blobs[1] = value.channel(i / num_heads_per_group);

        std::vector<Mat> qkv_top_blobs(1);
        qkv_top_blobs[0] = top_blob.channel(i);

        Option opt1 = opt_fp32;
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
