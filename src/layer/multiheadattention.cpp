// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "multiheadattention.h"

#include <float.h>

namespace ncnn {

MultiHeadAttention::MultiHeadAttention()
{
}

int MultiHeadAttention::load_param(const ParamDict& pd)
{
    embed_dim = pd.get(0, 0);
    num_heads = pd.get(1, 1);
    weight_data_size = pd.get(2, 0);
    kdim = pd.get(3, embed_dim);
    vdim = pd.get(4, embed_dim);
    attn_mask = pd.get(5, 0);
    scale = pd.get(6, 1.f / sqrtf(embed_dim / num_heads));
    kv_cache = pd.get(7, 0);
    int8_scale_term = pd.get(18, 0);

    return 0;
}

int MultiHeadAttention::load_model(const ModelBin& mb)
{
    const int qdim = weight_data_size / embed_dim;

    q_weight_data = mb.load(embed_dim * qdim, 0);
    if (q_weight_data.empty())
        return -100;

    q_bias_data = mb.load(embed_dim, 1);
    if (q_bias_data.empty())
        return -100;

    k_weight_data = mb.load(embed_dim * kdim, 0);
    if (k_weight_data.empty())
        return -100;

    k_bias_data = mb.load(embed_dim, 1);
    if (k_bias_data.empty())
        return -100;

    v_weight_data = mb.load(embed_dim * vdim, 0);
    if (v_weight_data.empty())
        return -100;

    v_bias_data = mb.load(embed_dim, 1);
    if (v_bias_data.empty())
        return -100;

    out_weight_data = mb.load(qdim * embed_dim, 0);
    if (out_weight_data.empty())
        return -100;

    out_bias_data = mb.load(qdim, 1);
    if (out_bias_data.empty())
        return -100;

#if NCNN_INT8
    if (int8_scale_term)
    {
        q_weight_data_int8_scales = mb.load(embed_dim, 1);
        k_weight_data_int8_scales = mb.load(embed_dim, 1);
        v_weight_data_int8_scales = mb.load(embed_dim, 1);
        out_weight_data_int8_scale = mb.load(1, 1)[0];
    }
#endif // NCNN_INT8

    return 0;
}

// refers to https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
int MultiHeadAttention::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
#endif

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

    const int src_seqlen = q_blob.h;
    const int cur_seqlen = k_blob.h;
    const int past_seqlen = kv_cache && !cached_xk_blob.empty() ? cached_xk_blob.w : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    const int embed_dim_per_head = embed_dim / num_heads;
    const int qdim = weight_data_size / embed_dim;

    // assert k_blob.h == v_blob.h

    Mat& top_blob = top_blobs[0];
    top_blob.create(qdim, src_seqlen, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    Mat xq(embed_dim_per_head, src_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xq.empty())
        return -100;

    // layout for efficient append
    Mat xk(embed_dim_per_head, dst_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xk.empty())
        return -100;

    // layout for efficient append
    Mat xv(embed_dim_per_head, dst_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xv.empty())
        return -100;

    Mat xqk(dst_seqlen, src_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xqk.empty())
        return -100;

    Mat xqkv(embed_dim_per_head, num_heads, src_seqlen, 4u, opt.workspace_allocator);
    if (xqkv.empty())
        return -100;

    if (kv_cache && !cached_xk_blob.empty() && q_blob_i != k_blob_i)
    {
        xk = cached_xk_blob;
    }
    if (kv_cache && !cached_xv_blob.empty() && q_blob_i != v_blob_i)
    {
        xv = cached_xv_blob;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        // xq = affine(q) * scale
        {
            Mat xqm = xq.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* outptr = xqm.row(i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* ptr = q_blob.row(i);
                    const float* kptr = (const float*)q_weight_data + qdim * (q * embed_dim_per_head + j);

                    float sum = q_bias_data[q * embed_dim_per_head + j];
                    for (int k = 0; k < qdim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    outptr[j] = sum * scale;
                }
            }
        }

        // xk = affine(k)
        if (kv_cache && !cached_xk_blob.empty() && q_blob_i != k_blob_i)
        {
            // pass
        }
        else
        {
            Mat xkm = xk.channel(q);

            if (past_seqlen > 0)
            {
                // reuse cached_xk
                memcpy(xkm, cached_xk_blob.channel(q), past_seqlen * embed_dim_per_head * sizeof(float));
            }

            for (int i = 0; i < cur_seqlen; i++)
            {
                float* xk_ptr = xkm.row(past_seqlen + i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* ptr = k_blob.row(i);
                    const float* kptr = (const float*)k_weight_data + kdim * (q * embed_dim_per_head + j);

                    float sum = k_bias_data[q * embed_dim_per_head + j];
                    for (int k = 0; k < kdim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    xk_ptr[j] = sum;
                }
            }
        }

        // xv = affine(v)
        if (kv_cache && !cached_xv_blob.empty() && q_blob_i != v_blob_i)
        {
            // pass
        }
        else
        {
            Mat xvm = xv.channel(q);

            if (past_seqlen > 0)
            {
                // reuse cached_xv
                memcpy(xvm, cached_xv_blob.channel(q), past_seqlen * embed_dim_per_head * sizeof(float));
            }

            for (int i = 0; i < cur_seqlen; i++)
            {
                float* xv_ptr = xvm.row(past_seqlen + i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* ptr = v_blob.row(i);
                    const float* kptr = (const float*)v_weight_data + vdim * (q * embed_dim_per_head + j);

                    float sum = v_bias_data[q * embed_dim_per_head + j];
                    for (int k = 0; k < vdim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    xv_ptr[j] = sum;
                }
            }
        }

        // xqk = xq * xk^T
        {
            const Mat xqm = xq.channel(q);
            const Mat xkm = xk.channel(q);
            Mat xqkm = xqk.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* outptr = xqkm.row(i);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    const float* qptr = xqm.row(i);
                    const float* kptr = xkm.row(j);

                    float sum = 0.f;
                    for (int k = 0; k < embed_dim_per_head; k++)
                    {
                        sum += *qptr++ * *kptr++;
                    }

                    outptr[j] = sum;
                }
            }
        }

        // xqk = xqk + mask
        if (attn_mask)
        {
            const Mat& maskm = attn_mask_blob.dims == 3 ? attn_mask_blob.channel(q) : attn_mask_blob;
            Mat xqkm = xqk.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                const float* mptr = maskm.row(i);
                float* outptr = xqkm.row(i);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    outptr[j] += mptr[j];
                }
            }
        }

        // softmax(xqk)
        {
            Mat xqkm = xqk.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* ptr = xqkm.row(i);

                float max = -FLT_MAX;
                for (int j = 0; j < dst_seqlen; j++)
                {
                    max = std::max(max, ptr[j]);
                }

                float sum = 0.f;
                for (int j = 0; j < dst_seqlen; j++)
                {
                    ptr[j] = (float)expf(ptr[j] - max);
                    sum += ptr[j];
                }

                for (int j = 0; j < dst_seqlen; j++)
                {
                    ptr[j] /= sum;
                }
            }
        }

        // xqkv = xqk * xv
        {
            const Mat xqkm = xqk.channel(q);
            const Mat xvm = xv.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* outptr = xqkv.channel(i).row(q);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* qkptr = xqkm.row(i);
                    const float* vptr = xvm.row(0) + j;

                    float sum = 0.f;
                    for (int k = 0; k < dst_seqlen; k++)
                    {
                        sum += *qkptr++ * *vptr;
                        vptr += embed_dim_per_head;
                    }

                    outptr[j] = sum;
                }
            }
        }
    }

    // out = affine(xqkv)
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < src_seqlen; i++)
    {
        float* outptr = top_blob.row(i);
        for (int j = 0; j < qdim; j++)
        {
            const float* ptr = xqkv.channel(i);
            const float* kptr = (const float*)out_weight_data + embed_dim * j;

            float sum = out_bias_data[j];
            for (int k = 0; k < embed_dim; k++)
            {
                sum += *ptr++ * *kptr++;
            }

            outptr[j] = sum;
        }
    }

    if (kv_cache)
    {
        // assert top_blobs.size() == 3
        top_blobs[1] = xk;
        top_blobs[2] = xv;
    }

    return 0;
}

void MultiHeadAttention::resolve_bottom_blob_index(int bottom_blob_count, int& q_blob_i, int& k_blob_i, int& v_blob_i, int& attn_mask_i, int& cached_xk_i, int& cached_xv_i) const
{
    if (kv_cache)
    {
        if (attn_mask)
        {
            // assert bottom_blob_count == 4/5/6
            if (bottom_blob_count == 4)
            {
                q_blob_i = 0;
                k_blob_i = 0;
                v_blob_i = 0;
                attn_mask_i = 1;
                cached_xk_i = 2;
                cached_xv_i = 3;
            }
            if (bottom_blob_count == 5)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 1;
                attn_mask_i = 2;
                cached_xk_i = 3;
                cached_xv_i = 4;
            }
            if (bottom_blob_count == 6)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 2;
                attn_mask_i = 3;
                cached_xk_i = 4;
                cached_xv_i = 5;
            }
        }
        else
        {
            // assert bottom_blob_count == 3/4/5
            if (bottom_blob_count == 3)
            {
                q_blob_i = 0;
                k_blob_i = 0;
                v_blob_i = 0;
                cached_xk_i = 1;
                cached_xv_i = 2;
            }
            if (bottom_blob_count == 4)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 1;
                cached_xk_i = 2;
                cached_xv_i = 3;
            }
            if (bottom_blob_count == 5)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 2;
                cached_xk_i = 3;
                cached_xv_i = 4;
            }
        }
    }
    else
    {
        if (attn_mask)
        {
            // assert bottom_blob_count == 2/3/4
            if (bottom_blob_count == 2)
            {
                q_blob_i = 0;
                k_blob_i = 0;
                v_blob_i = 0;
                attn_mask_i = 1;
            }
            if (bottom_blob_count == 3)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 1;
                attn_mask_i = 2;
            }
            if (bottom_blob_count == 4)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 2;
                attn_mask_i = 3;
            }
        }
        else
        {
            // assert bottom_blob_count == 1/2/3
            if (bottom_blob_count == 1)
            {
                q_blob_i = 0;
                k_blob_i = 0;
                v_blob_i = 0;
            }
            if (bottom_blob_count == 2)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 1;
            }
            if (bottom_blob_count == 3)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 2;
            }
        }
    }
}

#if NCNN_INT8
static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void dynamic_quantize_2d(const Mat& blob, Mat& blob_int8, float& scale, const Option& opt)
{
    blob_int8.create(blob.w, blob.h, (size_t)1u, 1, opt.workspace_allocator);

    float absmax = 0.f;
    for (int i = 0; i < blob_int8.h; i++)
    {
        const float* ptr = blob.row(i);

        for (int j = 0; j < blob_int8.w; j++)
        {
            absmax = std::max(absmax, (float)fabs(ptr[j]));
        }
    }

    scale = absmax == 0.f ? 1.f : 127.f / absmax;

    for (int i = 0; i < blob_int8.h; i++)
    {
        const float* ptr = blob.row(i);
        signed char* outptr = blob_int8.row<signed char>(i);

        for (int j = 0; j < blob_int8.w; j++)
        {
            outptr[j] = float2int8(ptr[j] * scale);
        }
    }
}

static void dynamic_quantize_2d_per_h(const Mat& blob, Mat& blob_int8, Mat& scales, const Option& opt)
{
    blob_int8.create(blob.w, blob.h, (size_t)1u, 1, opt.workspace_allocator);
    scales.create(blob.h, (size_t)4u, 1, opt.workspace_allocator);

    for (int i = 0; i < blob_int8.h; i++)
    {
        const float* ptr = blob.row(i);

        float absmax = 0.f;
        for (int j = 0; j < blob_int8.w; j++)
        {
            absmax = std::max(absmax, (float)fabs(ptr[j]));
        }

        scales[i] = absmax == 0.f ? 1.f : 127.f / absmax;
    }

    for (int i = 0; i < blob_int8.h; i++)
    {
        const float* ptr = blob.row(i);
        signed char* outptr = blob_int8.row<signed char>(i);
        const float scale = scales[i];

        for (int j = 0; j < blob_int8.w; j++)
        {
            outptr[j] = float2int8(ptr[j] * scale);
        }
    }
}

int MultiHeadAttention::forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

    const int src_seqlen = q_blob.h;
    const int cur_seqlen = k_blob.h;
    const int past_seqlen = kv_cache && !cached_xk_blob.empty() ? cached_xk_blob.w : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    const int embed_dim_per_head = embed_dim / num_heads;
    const int qdim = weight_data_size / embed_dim;

    // assert k_blob.h == v_blob.h

    Mat& top_blob = top_blobs[0];
    top_blob.create(qdim, src_seqlen, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    Mat xq(embed_dim_per_head, src_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xq.empty())
        return -100;

    // layout for efficient append
    Mat xk(embed_dim_per_head, dst_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xk.empty())
        return -100;

    // layout for efficient append
    Mat xv(embed_dim_per_head, dst_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xv.empty())
        return -100;

    Mat xqk(dst_seqlen, src_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xqk.empty())
        return -100;

    Mat xqkv(embed_dim_per_head, num_heads, src_seqlen, 4u, opt.workspace_allocator);
    if (xqkv.empty())
        return -100;

    // dynamic quantize q_blob
    Mat q_blob_int8;
    float q_blob_int8_scale;
    dynamic_quantize_2d(q_blob, q_blob_int8, q_blob_int8_scale, opt);

    // dynamic quantize k_blob
    Mat k_blob_int8;
    float k_blob_int8_scale;
    if (bottom_blobs.size() == 1 || (bottom_blobs.size() == 2 && attn_mask))
    {
        k_blob_int8 = q_blob_int8;
        k_blob_int8_scale = q_blob_int8_scale;
    }
    else
    {
        dynamic_quantize_2d(k_blob, k_blob_int8, k_blob_int8_scale, opt);
    }

    // dynamic quantize v_blob
    Mat v_blob_int8;
    float v_blob_int8_scale;
    if (bottom_blobs.size() == 1 || (bottom_blobs.size() == 2 && attn_mask))
    {
        v_blob_int8 = q_blob_int8;
        v_blob_int8_scale = q_blob_int8_scale;
    }
    else if (bottom_blobs.size() == 2 || (bottom_blobs.size() == 3 && attn_mask))
    {
        v_blob_int8 = k_blob_int8;
        v_blob_int8_scale = k_blob_int8_scale;
    }
    else
    {
        dynamic_quantize_2d(v_blob, v_blob_int8, v_blob_int8_scale, opt);
    }

    // NCNN_LOGE("%.4f %.4f", q_weight_data_int8_scale, q_blob_int8_scale);

    if (kv_cache && !cached_xk_blob.empty() && q_blob_i != k_blob_i)
    {
        xk = cached_xk_blob;
    }
    if (kv_cache && !cached_xv_blob.empty() && q_blob_i != v_blob_i)
    {
        xv = cached_xv_blob;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        // xq = affine(q) * scale
        {
            Mat outm = xq.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const signed char* ptr = q_blob_int8.row<const signed char>(i);
                    const signed char* kptr = (const signed char*)q_weight_data + qdim * (q * embed_dim_per_head + j);

                    int sum = 0;
                    for (int k = 0; k < qdim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }
                    const float q_descale = 1.f / (q_weight_data_int8_scales[q * embed_dim_per_head + j] * q_blob_int8_scale);
                    float sum_fp32 = sum * q_descale + q_bias_data[q * embed_dim_per_head + j];

                    outptr[j] = sum_fp32 * scale;
                }
            }
        }

        // xk = affine(k)
        if (kv_cache && !cached_xk_blob.empty() && q_blob_i != k_blob_i)
        {
            // pass
        }
        else
        {
            Mat xkm = xk.channel(q);

            if (past_seqlen > 0)
            {
                // reuse cached_xk
                memcpy(xkm, cached_xk_blob.channel(q), past_seqlen * embed_dim_per_head * sizeof(float));
            }

            for (int i = 0; i < cur_seqlen; i++)
            {
                float* xk_ptr = xkm.row(past_seqlen + i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const signed char* ptr = k_blob_int8.row<const signed char>(i);
                    const signed char* kptr = (const signed char*)k_weight_data + kdim * (q * embed_dim_per_head + j);

                    int sum = 0;
                    for (int k = 0; k < kdim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }
                    const float k_descale = 1.f / (k_weight_data_int8_scales[q * embed_dim_per_head + j] * k_blob_int8_scale);
                    float sum_fp32 = sum * k_descale + k_bias_data[q * embed_dim_per_head + j];

                    xk_ptr[j] = sum_fp32;
                }
            }
        }

        // xv = affine(v)
        if (kv_cache && !cached_xv_blob.empty() && q_blob_i != v_blob_i)
        {
            // pass
        }
        else
        {
            Mat xvm = xv.channel(q);

            if (past_seqlen > 0)
            {
                // reuse cached_xv
                memcpy(xvm, cached_xv_blob.channel(q), past_seqlen * embed_dim_per_head * sizeof(float));
            }

            for (int i = 0; i < cur_seqlen; i++)
            {
                float* xv_ptr = xvm.row(past_seqlen + i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const signed char* ptr = v_blob_int8.row<const signed char>(i);
                    const signed char* kptr = (const signed char*)v_weight_data + vdim * (q * embed_dim_per_head + j);

                    int sum = 0;
                    for (int k = 0; k < vdim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }
                    const float v_descale = 1.f / (v_weight_data_int8_scales[q * embed_dim_per_head + j] * v_blob_int8_scale);
                    float sum_fp32 = sum * v_descale + v_bias_data[q * embed_dim_per_head + j];

                    xv_ptr[j] = sum_fp32;
                }
            }
        }

        // xqk = xq * xk^T
        {
            const Mat xqm = xq.channel(q);
            const Mat xkm = xk.channel(q);
            Mat xqkm = xqk.channel(q);

            // dynamic quantize xqm per h
            Mat xqm_int8;
            Mat xqm_int8_scales;
            dynamic_quantize_2d_per_h(xqm, xqm_int8, xqm_int8_scales, opt);

            // dynamic quantize xkm
            Mat xkm_int8;
            float xkm_int8_scale;
            dynamic_quantize_2d(xkm, xkm_int8, xkm_int8_scale, opt);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* outptr = xqkm.row(i);
                const float xqk_descale = 1.f / (xqm_int8_scales[i] * xkm_int8_scale);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    const signed char* qptr = xqm_int8.row<const signed char>(i);
                    const signed char* kptr = xkm_int8.row<const signed char>(j);

                    int sum = 0;
                    for (int k = 0; k < embed_dim_per_head; k++)
                    {
                        sum += *qptr++ * *kptr++;
                    }
                    float sum_fp32 = sum * xqk_descale;

                    outptr[j] = sum_fp32;
                }
            }
        }

        // xqk = xqk + mask
        if (attn_mask)
        {
            const Mat& maskm = attn_mask_blob.dims == 3 ? attn_mask_blob.channel(q) : attn_mask_blob;
            Mat xqkm = xqk.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                const float* mptr = maskm.row(i);
                float* outptr = xqkm.row(i);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    outptr[j] += mptr[j];
                }
            }
        }

        // softmax(xqk)
        {
            Mat xqkm = xqk.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* ptr = xqkm.row(i);

                float max = -FLT_MAX;
                for (int j = 0; j < dst_seqlen; j++)
                {
                    max = std::max(max, ptr[j]);
                }

                float sum = 0.f;
                for (int j = 0; j < dst_seqlen; j++)
                {
                    ptr[j] = (float)expf(ptr[j] - max);
                    sum += ptr[j];
                }

                for (int j = 0; j < dst_seqlen; j++)
                {
                    ptr[j] /= sum;
                }
            }
        }

        // xqkv = xqk * xv
        {
            const Mat xqkm = xqk.channel(q);
            const Mat xvm = xv.channel(q);

            // dynamic quantize xqkm
            Mat xqkm_int8;
            Mat xqkm_int8_scales;
            dynamic_quantize_2d_per_h(xqkm, xqkm_int8, xqkm_int8_scales, opt);

            // dynamic quantize xvm per h
            Mat xvm_int8;
            float xvm_int8_scale;
            dynamic_quantize_2d(xvm, xvm_int8, xvm_int8_scale, opt);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* outptr = xqkv.channel(i).row(q);
                const float xqkv_descale = 1.f / (xqkm_int8_scales[i] * xvm_int8_scale);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const signed char* qkptr = xqkm_int8.row<const signed char>(i);
                    const signed char* vptr = xvm_int8.row<const signed char>(0) + j;

                    int sum = 0;
                    for (int k = 0; k < dst_seqlen; k++)
                    {
                        sum += *qkptr++ * *vptr;
                        vptr += embed_dim_per_head;
                    }
                    float sum_fp32 = sum * xqkv_descale;

                    outptr[j] = sum_fp32;
                }
            }
        }
    }

    // dynamic quantize xqkv
    Mat xqkv_int8;
    Mat xqkv_int8_scales;
    {
        xqkv_int8.create(xqkv.w, xqkv.h, xqkv.c, (size_t)1u, 1, opt.workspace_allocator);
        xqkv_int8_scales.create(src_seqlen, (size_t)4u, 1, opt.workspace_allocator);

        for (int i = 0; i < xqkv_int8.c; i++)
        {
            const float* ptr = xqkv.channel(i);

            float absmax = 0.f;
            for (int j = 0; j < xqkv_int8.w * xqkv_int8.h; j++)
            {
                absmax = std::max(absmax, (float)fabs(ptr[j]));
            }

            xqkv_int8_scales[i] = absmax == 0.f ? 1.f : 127.f / absmax;
        }

        for (int i = 0; i < xqkv_int8.c; i++)
        {
            const float* ptr = xqkv.channel(i);
            signed char* outptr = xqkv_int8.channel(i);

            for (int j = 0; j < xqkv_int8.w * xqkv_int8.h; j++)
            {
                outptr[j] = float2int8(ptr[j] * xqkv_int8_scales[i]);
            }
        }
    }

    // out = affine(xqkv)
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < src_seqlen; i++)
    {
        float* outptr = top_blob.row(i);

        for (int j = 0; j < qdim; j++)
        {
            const signed char* ptr = xqkv_int8.channel(i);
            const signed char* kptr = (const signed char*)out_weight_data + embed_dim * j;

            int sum = 0;
            for (int k = 0; k < embed_dim; k++)
            {
                sum += *ptr++ * *kptr++;
            }
            const float out_descale = 1.f / (out_weight_data_int8_scale * xqkv_int8_scales[i]);
            float sum_fp32 = sum * out_descale + out_bias_data[j];

            outptr[j] = sum_fp32;
        }
    }

    if (kv_cache)
    {
        // assert top_blobs.size() == 3
        top_blobs[1] = xk;
        top_blobs[2] = xv;
    }

    return 0;
}
#endif

} // namespace ncnn
