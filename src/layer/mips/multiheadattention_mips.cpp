// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "multiheadattention_mips.h"

#include "layer_type.h"

namespace ncnn {

MultiHeadAttention_mips::MultiHeadAttention_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif

    q_gemm = 0;
    k_gemm = 0;
    v_gemm = 0;

    qk_gemm = 0;
    qkv_gemm = 0;

    qk_softmax = 0;

    o_gemm = 0;
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

int MultiHeadAttention_mips::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    if (int8_scale_term)
    {
        support_packing = false;
        support_bf16_storage = false;

        opt.use_packing_layout = false; // TODO enable packing
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
        pd.set(0, -1);
        pd.set(1, 1);
        qk_softmax->load_param(pd);
        qk_softmax->load_model(ModelBinFromMatArray(0));
        qk_softmax->create_pipeline(opt_fp32);
    }

    const int qdim = weight_data_size / embed_dim;

    {
        q_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(0, scale);
        pd.set(1, 1.f);
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 1);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, embed_dim); // M
        pd.set(8, 0);         // N
        pd.set(9, qdim);      // K
        pd.set(10, 1);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        pd.set(12, 1);        // output_elempack
        pd.set(14, 0);        // output_transpose
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        q_gemm->load_param(pd);
        Mat weights[3];
        weights[0] = q_weight_data;
        weights[1] = q_bias_data;
#if NCNN_INT8
        weights[2] = q_weight_data_int8_scales;
#endif
        q_gemm->load_model(ModelBinFromMatArray(weights));
        q_gemm->create_pipeline(opt_fp32);

        if (opt.lightmode)
        {
            q_weight_data.release();
            q_bias_data.release();
        }
    }

    {
        k_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
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
        pd.set(14, 0);        // output_transpose
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        k_gemm->load_param(pd);
        Mat weights[3];
        weights[0] = k_weight_data;
        weights[1] = k_bias_data;
#if NCNN_INT8
        weights[2] = k_weight_data_int8_scales;
#endif
        k_gemm->load_model(ModelBinFromMatArray(weights));
        k_gemm->create_pipeline(opt_fp32);

        if (opt.lightmode)
        {
            k_weight_data.release();
            k_bias_data.release();
        }
    }

    {
        v_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
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
        pd.set(14, 0);        // output_transpose
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        v_gemm->load_param(pd);
        Mat weights[3];
        weights[0] = v_weight_data;
        weights[1] = v_bias_data;
#if NCNN_INT8
        weights[2] = v_weight_data_int8_scales;
#endif
        v_gemm->load_model(ModelBinFromMatArray(weights));
        v_gemm->create_pipeline(opt_fp32);

        if (opt.lightmode)
        {
            v_weight_data.release();
            v_bias_data.release();
        }
    }

    {
        o_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(2, 1);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 1);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M = outch
        pd.set(8, qdim);      // N = size
        pd.set(9, embed_dim); // K = maxk*inch
        pd.set(10, 4);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
#if NCNN_INT8
        pd.set(18, int8_scale_term);
#endif
        o_gemm->load_param(pd);
        Mat weights[3];
        weights[0] = out_weight_data;
        weights[1] = out_bias_data;
#if NCNN_INT8
        Mat out_weight_data_int8_scales(1);
        out_weight_data_int8_scales[0] = out_weight_data_int8_scale;
        weights[2] = out_weight_data_int8_scales;
#endif
        o_gemm->load_model(ModelBinFromMatArray(weights));
        o_gemm->create_pipeline(opt_fp32);

        if (opt.lightmode)
        {
            out_weight_data.release();
            out_bias_data.release();
        }
    }

    {
        qk_gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        ncnn::ParamDict pd;
        pd.set(2, 1);                   // transA
        pd.set(3, 0);                   // transB
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
        pd.set(13, 1);  // output_elemtype = fp32
        pd.set(14, 1);  // output_transpose
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

int MultiHeadAttention_mips::destroy_pipeline(const Option& _opt)
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

    if (o_gemm)
    {
        o_gemm->destroy_pipeline(opt);
        delete o_gemm;
        o_gemm = 0;
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

int MultiHeadAttention_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& _opt) const
{
    int q_blob_i = 0;
    int k_blob_i = 0;
    int v_blob_i = 0;
    int attn_mask_i = 0;
    int cached_xk_i = 0;
    int cached_xv_i = 0;
    resolve_bottom_blob_index((int)bottom_blobs.size(), q_blob_i, k_blob_i, v_blob_i, attn_mask_i, cached_xk_i, cached_xv_i);

    const Mat& q_blob = bottom_blobs[q_blob_i];
    const Mat& k_blob = bottom_blobs[k_blob_i];
    const Mat& v_blob = bottom_blobs[v_blob_i];
    const Mat& attn_mask_blob = attn_mask ? bottom_blobs[attn_mask_i] : Mat();
    const Mat& cached_xk_blob = kv_cache ? bottom_blobs[cached_xk_i] : Mat();
    const Mat& cached_xv_blob = kv_cache ? bottom_blobs[cached_xv_i] : Mat();

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
        int ret = MultiHeadAttention::forward(bottom_blobs_fp32, top_blobs_fp32, opt_fp32);
        if (ret != 0)
            return ret;

        if (restore_layout_from_reference(top_blobs_fp32[0], top_blobs[0], q_blob, _opt) != 0)
            return -100;

        if (kv_cache)
        {
            if (restore_layout_from_reference(top_blobs_fp32[1], top_blobs[1], k_blob, _opt) != 0)
                return -100;
            if (restore_layout_from_reference(top_blobs_fp32[2], top_blobs[2], v_blob, _opt) != 0)
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

    Mat q_blob_unpacked;
    Mat k_blob_unpacked;
    Mat v_blob_unpacked;
    Mat attn_mask_blob_unpacked;
    Mat cached_xk_blob_unpacked;
    Mat cached_xv_blob_unpacked;
    if (unpack_or_cast_to_float32(q_blob, q_blob_unpacked, opt) != 0
            || unpack_or_cast_to_float32(k_blob, k_blob_unpacked, opt) != 0
            || unpack_or_cast_to_float32(v_blob, v_blob_unpacked, opt) != 0
            || unpack_or_cast_to_float32(attn_mask_blob, attn_mask_blob_unpacked, opt) != 0
            || unpack_or_cast_to_float32(cached_xk_blob, cached_xk_blob_unpacked, opt) != 0
            || unpack_or_cast_to_float32(cached_xv_blob, cached_xv_blob_unpacked, opt) != 0)
        return -100;

    const int embed_dim_per_head = embed_dim / num_heads;
    const int src_seqlen = q_blob_unpacked.h;
    const int cur_seqlen = k_blob_unpacked.h;
    const int past_seqlen = kv_cache && !cached_xk_blob_unpacked.empty() ? cached_xk_blob_unpacked.w : 0;
    const int dst_seqlen = past_seqlen > 0 ? (q_blob_i == k_blob_i ? (past_seqlen + cur_seqlen) : past_seqlen) : cur_seqlen;

    Mat q_affine;
    int retq = q_gemm->forward(q_blob_unpacked, q_affine, opt_fp32);
    if (retq != 0)
        return retq;

    Mat k_affine;
    if (past_seqlen > 0)
    {
        if (q_blob_i == k_blob_i)
        {
            Mat k_affine_q;
            int retk = k_gemm->forward(q_blob_unpacked, k_affine_q, opt_fp32);
            if (retk != 0)
                return retk;

            k_affine.create(dst_seqlen, embed_dim, k_affine_q.elemsize, opt_fp32.workspace_allocator);
            if (k_affine.empty())
                return -100;

            for (int i = 0; i < embed_dim; i++)
            {
                const unsigned char* ptr = cached_xk_blob_unpacked.row<const unsigned char>(i);
                const unsigned char* ptrq = k_affine_q.row<const unsigned char>(i);
                unsigned char* outptr = k_affine.row<unsigned char>(i);

                memcpy(outptr, ptr, past_seqlen * k_affine.elemsize);
                memcpy(outptr + past_seqlen * k_affine.elemsize, ptrq, cur_seqlen * k_affine.elemsize);
            }
        }
        else
        {
            k_affine = cached_xk_blob_unpacked;
        }
    }
    else
    {
        int retk = k_gemm->forward(k_blob_unpacked, k_affine, opt_fp32);
        if (retk != 0)
            return retk;
    }

    Mat qk_cross(dst_seqlen, src_seqlen * num_heads, 4u, opt_fp32.blob_allocator);
    if (qk_cross.empty())
        return -100;

    std::vector<int> retqks;
    retqks.resize(num_heads);
    #pragma omp parallel for num_threads(opt_fp32.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        std::vector<Mat> qk_bottom_blobs(2);
        qk_bottom_blobs[0] = q_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
        qk_bottom_blobs[1] = k_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
        if (attn_mask)
        {
            const Mat& maskm = attn_mask_blob_unpacked.dims == 3 ? attn_mask_blob_unpacked.channel(i) : attn_mask_blob_unpacked;
            qk_bottom_blobs.push_back(maskm);
        }
        std::vector<Mat> qk_top_blobs(1);
        qk_top_blobs[0] = qk_cross.row_range(i * src_seqlen, src_seqlen);
        Option opt1 = opt_fp32;
        opt1.num_threads = 1;
        retqks[i] = qk_gemm->forward(qk_bottom_blobs, qk_top_blobs, opt1);
    }
    for (int i = 0; i < num_heads; i++)
    {
        if (retqks[i] != 0)
            return retqks[i];
    }

    q_affine.release();

    if (!kv_cache)
    {
        k_affine.release();
    }

    int retqk = qk_softmax->forward_inplace(qk_cross, opt_fp32);
    if (retqk != 0)
        return retqk;

    Mat v_affine;
    if (past_seqlen > 0)
    {
        if (q_blob_i == v_blob_i)
        {
            Mat v_affine_q;
            int retv = v_gemm->forward(v_blob_unpacked, v_affine_q, opt_fp32);
            if (retv != 0)
                return retv;

            v_affine.create(dst_seqlen, embed_dim, v_affine_q.elemsize, opt_fp32.workspace_allocator);
            if (v_affine.empty())
                return -100;

            for (int i = 0; i < embed_dim; i++)
            {
                const unsigned char* ptr = cached_xv_blob_unpacked.row<const unsigned char>(i);
                const unsigned char* ptrq = v_affine_q.row<const unsigned char>(i);
                unsigned char* outptr = v_affine.row<unsigned char>(i);

                memcpy(outptr, ptr, past_seqlen * v_affine.elemsize);
                memcpy(outptr + past_seqlen * v_affine.elemsize, ptrq, cur_seqlen * v_affine.elemsize);
            }
        }
        else
        {
            v_affine = cached_xv_blob_unpacked;
        }
    }
    else
    {
        int retv = v_gemm->forward(v_blob_unpacked, v_affine, opt_fp32);
        if (retv != 0)
            return retv;
    }

    Mat qkv_cross(src_seqlen, embed_dim_per_head * num_heads, 4u, opt_fp32.blob_allocator);
    if (qkv_cross.empty())
        return -100;

    std::vector<int> retqkvs;
    retqkvs.resize(num_heads);
    #pragma omp parallel for num_threads(opt_fp32.num_threads)
    for (int i = 0; i < num_heads; i++)
    {
        std::vector<Mat> qkv_bottom_blobs(2);
        qkv_bottom_blobs[0] = qk_cross.row_range(i * src_seqlen, src_seqlen);
        qkv_bottom_blobs[1] = v_affine.row_range(i * embed_dim_per_head, embed_dim_per_head);
        std::vector<Mat> qkv_top_blobs(1);
        qkv_top_blobs[0] = qkv_cross.row_range(i * embed_dim_per_head, embed_dim_per_head);
        Option opt1 = opt_fp32;
        opt1.num_threads = 1;
        retqkvs[i] = qkv_gemm->forward(qkv_bottom_blobs, qkv_top_blobs, opt1);
    }
    for (int i = 0; i < num_heads; i++)
    {
        if (retqkvs[i] != 0)
            return retqkvs[i];
    }

    if (!kv_cache)
    {
        v_affine.release();
    }

    int reto = o_gemm->forward(qkv_cross, top_blobs[0], opt_fp32);
    if (reto != 0)
        return reto;

    if (kv_cache)
    {
        top_blobs[1] = k_affine;
        top_blobs[2] = v_affine;
    }

    return 0;
}

} // namespace ncnn
