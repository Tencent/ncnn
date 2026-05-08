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

    //              | self-attention  cross-attention
    // w/o kvcache  | past(0) + cur   cur
    // with kvcache | past + cur      past

    const int src_seqlen = q_blob.h;
    const int cur_seqlen = k_blob.h;
    const int past_seqlen = kv_cache && !cached_xk_blob.empty() ? cached_xk_blob.w : 0;
    const int dst_seqlen = past_seqlen > 0 ? (q_blob_i == k_blob_i ? (past_seqlen + cur_seqlen) : past_seqlen) : cur_seqlen;

    const int embed_dim_per_head = embed_dim / num_heads;
    const int qdim = weight_data_size / embed_dim;

    // assert k_blob.h == v_blob.h

    Mat q_affine;
    {
        q_affine.create(src_seqlen, embed_dim, 4u, opt.workspace_allocator);
        if (q_affine.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < src_seqlen; i++)
        {
            const float* kptr = (const float*)q_weight_data;

            for (int j = 0; j < embed_dim; j++)
            {
                const float* ptr = q_blob.row(i);

                float sum = q_bias_data[j];
                for (int k = 0; k < qdim; k++)
                {
                    sum += *ptr++ * *kptr++;
                }

                q_affine.row(j)[i] = sum * scale;
            }
        }
    }

    Mat k_affine;
    if (past_seqlen > 0 && q_blob_i != k_blob_i)
    {
        k_affine = cached_xk_blob;
    }
    else
    {
        k_affine.create(dst_seqlen, embed_dim, 4u, opt.workspace_allocator);
        if (k_affine.empty())
            return -100;

        if (past_seqlen > 0)
        {
            // reuse cached_xk
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < embed_dim; i++)
            {
                memcpy(k_affine.row(i), cached_xk_blob.row(i), dst_seqlen * sizeof(float));
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < cur_seqlen; i++)
        {
            const float* kptr = (const float*)k_weight_data;

            for (int j = 0; j < embed_dim; j++)
            {
                const float* ptr = k_blob.row(i);

                float sum = k_bias_data[j];
                for (int k = 0; k < kdim; k++)
                {
                    sum += *ptr++ * *kptr++;
                }

                k_affine.row(j)[past_seqlen + i] = sum;
            }
        }
    }

    Mat v_affine;
    if (past_seqlen > 0 && q_blob_i != v_blob_i)
    {
        v_affine = cached_xv_blob;
    }
    else
    {
        v_affine.create(dst_seqlen, embed_dim, 4u, opt.workspace_allocator);
        if (v_affine.empty())
            return -100;

        if (past_seqlen > 0)
        {
            // reuse cached_xv
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < embed_dim; i++)
            {
                memcpy(v_affine.row(i), cached_xv_blob.row(i), dst_seqlen * sizeof(float));
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < cur_seqlen; i++)
        {
            const float* kptr = (const float*)v_weight_data;

            for (int j = 0; j < embed_dim; j++)
            {
                const float* ptr = v_blob.row(i);

                float sum = v_bias_data[j];
                for (int k = 0; k < vdim; k++)
                {
                    sum += *ptr++ * *kptr++;
                }

                v_affine.row(j)[past_seqlen + i] = sum;
            }
        }
    }

    Mat qk_cross;
    {
        qk_cross.create(dst_seqlen, src_seqlen, num_heads, 4u, opt.workspace_allocator);
        if (qk_cross.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const Mat q_affine_head = q_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
            const Mat k_affine_head = k_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
            Mat qk_cross_head = qk_cross.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* outptr = qk_cross_head.row(i);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    float sum = 0.f;
                    for (int l = 0; l < embed_dim_per_head; l++)
                    {
                        sum += q_affine_head.row(l)[i] * k_affine_head.row(l)[j];
                    }

                    outptr[j] = sum;
                }
            }
        }
    }

    if (attn_mask)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const Mat& maskm = attn_mask_blob.dims == 3 ? attn_mask_blob.channel(q) : attn_mask_blob;
            Mat qk_cross_head = qk_cross.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                const float* mptr = maskm.row(i);
                float* outptr = qk_cross_head.row(i);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    outptr[j] += mptr[j];
                }
            }
        }
    }

    // softmax(qk_cross)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            Mat qk_cross_head = qk_cross.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* ptr = qk_cross_head.row(i);

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
    }

    Mat qkv_cross;
    {
        qkv_cross.create(src_seqlen, embed_dim, 4u, opt.workspace_allocator);
        if (qkv_cross.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const Mat qk_cross_head = qk_cross.channel(q);
            const Mat v_affine_head = v_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
            Mat qkv_cross_head = qkv_cross.row_range(q * embed_dim_per_head, embed_dim_per_head);

            for (int i = 0; i < src_seqlen; i++)
            {
                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* qkptr = qk_cross_head.row(i);
                    const float* vptr = v_affine_head.row(j);

                    float sum = 0.f;
                    for (int k = 0; k < dst_seqlen; k++)
                    {
                        sum += *qkptr++ * *vptr++;
                    }

                    qkv_cross_head.row(j)[i] = sum;
                }
            }
        }
    }

    Mat& top_blob = top_blobs[0];
    {
        top_blob.create(qdim, src_seqlen, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < src_seqlen; i++)
        {
            const float* kptr = (const float*)out_weight_data;
            float* outptr = top_blob.row(i);

            for (int j = 0; j < qdim; j++)
            {
                float sum = out_bias_data[j];
                for (int k = 0; k < embed_dim; k++)
                {
                    sum += qkv_cross.row(k)[i] * *kptr++;
                }

                outptr[j] = sum;
            }
        }
    }

    if (kv_cache)
    {
        // assert top_blobs.size() == 3
        top_blobs[1] = k_affine;
        top_blobs[2] = v_affine;
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

static void dynamic_quantize_2d_per_w(const Mat& blob, Mat& blob_int8, Mat& scales, const Option& opt)
{
    blob_int8.create(blob.w, blob.h, (size_t)1u, 1, opt.workspace_allocator);
    scales.create(blob.w, (size_t)4u, 1, opt.workspace_allocator);

    scales.fill(0.f);
    for (int i = 0; i < blob_int8.h; i++)
    {
        const float* ptr = blob.row(i);

        for (int j = 0; j < blob_int8.w; j++)
        {
            scales[j] = std::max(scales[j], (float)fabs(ptr[j]));
        }
    }
    for (int i = 0; i < blob_int8.w; i++)
    {
        scales[i] = scales[i] == 0.f ? 1.f : 127.f / scales[i];
    }

    for (int i = 0; i < blob_int8.h; i++)
    {
        const float* ptr = blob.row(i);
        signed char* outptr = blob_int8.row<signed char>(i);

        for (int j = 0; j < blob_int8.w; j++)
        {
            outptr[j] = float2int8(ptr[j] * scales[j]);
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
    const int dst_seqlen = past_seqlen > 0 ? (q_blob_i == k_blob_i ? (past_seqlen + cur_seqlen) : past_seqlen) : cur_seqlen;

    const int embed_dim_per_head = embed_dim / num_heads;
    const int qdim = weight_data_size / embed_dim;

    // assert k_blob.h == v_blob.h

    Mat q_affine;
    {
        q_affine.create(src_seqlen, embed_dim, 4u, opt.workspace_allocator);
        if (q_affine.empty())
            return -100;

        // dynamic quantize q_blob
        Mat q_blob_int8;
        float q_blob_int8_scale;
        dynamic_quantize_2d(q_blob, q_blob_int8, q_blob_int8_scale, opt);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < src_seqlen; i++)
        {
            const signed char* kptr = (const signed char*)q_weight_data;

            for (int j = 0; j < embed_dim; j++)
            {
                const signed char* ptr = q_blob_int8.row<const signed char>(i);

                int sum = 0;
                for (int k = 0; k < qdim; k++)
                {
                    sum += *ptr++ * *kptr++;
                }
                const float q_descale = 1.f / (q_weight_data_int8_scales[j] * q_blob_int8_scale);
                float sum_fp32 = sum * q_descale + q_bias_data[j];

                q_affine.row(j)[i] = sum_fp32 * scale;
            }
        }
    }

    Mat k_affine;
    if (past_seqlen > 0 && q_blob_i != k_blob_i)
    {
        k_affine = cached_xk_blob;
    }
    else
    {
        k_affine.create(dst_seqlen, embed_dim, 4u, opt.workspace_allocator);
        if (k_affine.empty())
            return -100;

        if (past_seqlen > 0)
        {
            // reuse cached_xk
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < embed_dim; i++)
            {
                memcpy(k_affine.row(i), cached_xk_blob.row(i), dst_seqlen * sizeof(float));
            }
        }

        // dynamic quantize k_blob
        Mat k_blob_int8;
        float k_blob_int8_scale;
        dynamic_quantize_2d(k_blob, k_blob_int8, k_blob_int8_scale, opt);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < cur_seqlen; i++)
        {
            const signed char* kptr = (const signed char*)k_weight_data;

            for (int j = 0; j < embed_dim; j++)
            {
                const signed char* ptr = k_blob_int8.row<const signed char>(i);

                int sum = 0;
                for (int k = 0; k < kdim; k++)
                {
                    sum += *ptr++ * *kptr++;
                }
                const float k_descale = 1.f / (k_weight_data_int8_scales[j] * k_blob_int8_scale);
                float sum_fp32 = sum * k_descale + k_bias_data[j];

                k_affine.row(j)[past_seqlen + i] = sum_fp32;
            }
        }
    }

    Mat v_affine;
    if (past_seqlen > 0 && q_blob_i != v_blob_i)
    {
        v_affine = cached_xv_blob;
    }
    else
    {
        v_affine.create(dst_seqlen, embed_dim, 4u, opt.workspace_allocator);
        if (v_affine.empty())
            return -100;

        if (past_seqlen > 0)
        {
            // reuse cached_xv
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < embed_dim; i++)
            {
                memcpy(v_affine.row(i), cached_xv_blob.row(i), dst_seqlen * sizeof(float));
            }
        }

        // dynamic quantize v_blob
        Mat v_blob_int8;
        float v_blob_int8_scale;
        dynamic_quantize_2d(v_blob, v_blob_int8, v_blob_int8_scale, opt);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < cur_seqlen; i++)
        {
            const signed char* kptr = (const signed char*)v_weight_data;

            for (int j = 0; j < embed_dim; j++)
            {
                const signed char* ptr = v_blob_int8.row<const signed char>(i);

                int sum = 0;
                for (int k = 0; k < vdim; k++)
                {
                    sum += *ptr++ * *kptr++;
                }
                const float v_descale = 1.f / (v_weight_data_int8_scales[j] * v_blob_int8_scale);
                float sum_fp32 = sum * v_descale + v_bias_data[j];

                v_affine.row(j)[past_seqlen + i] = sum_fp32;
            }
        }
    }

    Mat qk_cross;
    {
        qk_cross.create(dst_seqlen, src_seqlen, num_heads, 4u, opt.workspace_allocator);
        if (qk_cross.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const Mat q_affine_head = q_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
            const Mat k_affine_head = k_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
            Mat qk_cross_head = qk_cross.channel(q);

            // dynamic quantize q_affine_head per w
            Mat q_affine_head_int8;
            Mat q_affine_head_int8_scales;
            dynamic_quantize_2d_per_w(q_affine_head, q_affine_head_int8, q_affine_head_int8_scales, opt);

            // dynamic quantize k_affine_head
            Mat k_affine_head_int8;
            float k_affine_head_int8_scale;
            dynamic_quantize_2d(k_affine_head, k_affine_head_int8, k_affine_head_int8_scale, opt);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* outptr = qk_cross_head.row(i);
                const float qk_descale = 1.f / (q_affine_head_int8_scales[i] * k_affine_head_int8_scale);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    int sum = 0;
                    for (int l = 0; l < embed_dim_per_head; l++)
                    {
                        signed char vq = q_affine_head_int8.row<const signed char>(l)[i];
                        signed char vk = k_affine_head_int8.row<const signed char>(l)[j];
                        sum += vq * vk;
                    }
                    float sum_fp32 = sum * qk_descale;

                    outptr[j] = sum_fp32;
                }
            }
        }
    }

    if (attn_mask)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const Mat& maskm = attn_mask_blob.dims == 3 ? attn_mask_blob.channel(q) : attn_mask_blob;
            Mat qk_cross_head = qk_cross.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                const float* mptr = maskm.row(i);
                float* outptr = qk_cross_head.row(i);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    outptr[j] += mptr[j];
                }
            }
        }
    }

    // softmax(qk_cross)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            Mat qk_cross_head = qk_cross.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* ptr = qk_cross_head.row(i);

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
    }

    Mat qkv_cross;
    {
        qkv_cross.create(src_seqlen, embed_dim, 4u, opt.workspace_allocator);
        if (qkv_cross.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const Mat qk_cross_head = qk_cross.channel(q);
            const Mat v_affine_head = v_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
            Mat qkv_cross_head = qkv_cross.row_range(q * embed_dim_per_head, embed_dim_per_head);

            // dynamic quantize qk_cross_head per h
            Mat qk_cross_head_int8;
            Mat qk_cross_head_int8_scales;
            dynamic_quantize_2d_per_h(qk_cross_head, qk_cross_head_int8, qk_cross_head_int8_scales, opt);

            // dynamic quantize v_affine_head
            Mat v_affine_head_int8;
            float v_affine_head_int8_scale;
            dynamic_quantize_2d(v_affine_head, v_affine_head_int8, v_affine_head_int8_scale, opt);

            for (int i = 0; i < src_seqlen; i++)
            {
                const float qkv_descale = 1.f / (qk_cross_head_int8_scales[i] * v_affine_head_int8_scale);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const signed char* qkptr = qk_cross_head_int8.row<const signed char>(i);
                    const signed char* vptr = v_affine_head_int8.row<const signed char>(j);

                    int sum = 0;
                    for (int k = 0; k < dst_seqlen; k++)
                    {
                        sum += *qkptr++ * *vptr++;
                    }
                    float sum_fp32 = sum * qkv_descale;

                    qkv_cross_head.row(j)[i] = sum_fp32;
                }
            }
        }
    }

    Mat& top_blob = top_blobs[0];
    {
        // dynamic quantize qkv_cross
        Mat qkv_cross_int8;
        Mat qkv_cross_int8_scales;
        dynamic_quantize_2d_per_w(qkv_cross, qkv_cross_int8, qkv_cross_int8_scales, opt);

        top_blob.create(qdim, src_seqlen, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < src_seqlen; i++)
        {
            const signed char* kptr = (const signed char*)out_weight_data;
            float* outptr = top_blob.row(i);

            for (int j = 0; j < qdim; j++)
            {
                int sum = 0;
                for (int k = 0; k < embed_dim; k++)
                {
                    signed char vqkv = qkv_cross_int8.row<const signed char>(k)[i];
                    sum += vqkv * *kptr++;
                }
                const float out_descale = 1.f / (out_weight_data_int8_scale * qkv_cross_int8_scales[i]);
                float sum_fp32 = sum * out_descale + out_bias_data[j];

                outptr[j] = sum_fp32;
            }
        }
    }

    if (kv_cache)
    {
        // assert top_blobs.size() == 3
        top_blobs[1] = k_affine;
        top_blobs[2] = v_affine;
    }

    return 0;
}
#endif

} // namespace ncnn
