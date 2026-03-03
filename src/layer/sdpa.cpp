// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa.h"

#include <float.h>

#include "cpu.h"

namespace ncnn {

SDPA::SDPA()
{
}

int SDPA::load_param(const ParamDict& pd)
{
    attn_mask = pd.get(5, 0);
    scale = pd.get(6, 0.f);
    kv_cache = pd.get(7, 0);
    int8_scale_term = pd.get(18, 0);

    return 0;
}

// refers to https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
int SDPA::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return forward_int8(bottom_blobs, top_blobs, opt);
    }
#endif

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

    // assert cur_key.w == embed_dim
    // assert cur_key.h == cur_value.h == cur_seqlen
    // assert cur_value.c == num_group
    // assert num_heads % num_group == 0

    const float _scale = scale == 0.f ? 1.f / sqrt(embed_dim) : scale;
    const int num_heads_per_group = num_heads / num_group;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    Mat qk_cross(dst_seqlen, src_seqlen, opt.num_threads, 4u, opt.workspace_allocator);
    if (qk_cross.empty())
        return -100;

    Mat key = cur_key;
    if (past_seqlen > 0)
    {
        key.create(embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
        if (key.empty())
            return -100;

        // concat
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

    Mat value = cur_value;
    if (past_seqlen > 0)
    {
        value.create(out_embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
        if (value.empty())
            return -100;

        // concat
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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat query_head = query.channel(q);
        const Mat key_head = key.channel(q / num_heads_per_group);
        const Mat value_head = value.channel(q / num_heads_per_group);
        Mat qk_cross_head = qk_cross.channel(get_omp_thread_num());
        Mat top_blob_head = top_blob.channel(q);

        // qk_cross
        {
            for (int i = 0; i < src_seqlen; i++)
            {
                const float* qptr = query_head.row(i);
                float* outptr = qk_cross_head.row(i);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    const float* kptr = key_head.row(j);

                    float sum = 0.f;
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += qptr[k] * kptr[k];
                    }

                    outptr[j] = sum * _scale;
                }
            }
        }

        if (attn_mask)
        {
            const Mat& maskm = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob;

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

        // softmax
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

        // qkv_cross
        {
            for (int i = 0; i < src_seqlen; i++)
            {
                const float* qkptr = qk_cross_head.row(i);
                float* outptr = top_blob_head.row(i);

                for (int j = 0; j < out_embed_dim; j++)
                {
                    float sum = 0.f;
                    for (int k = 0; k < dst_seqlen; k++)
                    {
                        sum += qkptr[k] * value_head.row(k)[j];
                    }

                    outptr[j] = sum;
                }
            }
        }
    }

    if (kv_cache)
    {
        // assert top_blobs.size() == 3
        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}

#if NCNN_INT8
static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void dynamic_quantize_2d(const Mat& blob, Mat& blob_int8, float& scale)
{
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

static void dynamic_quantize_2d_per_h(const Mat& blob, Mat& blob_int8, Mat& scales)
{
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

int SDPA::forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
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

    // assert cur_key.w == embed_dim
    // assert cur_key.h == cur_value.h == cur_seqlen
    // assert cur_value.c == num_group
    // assert num_heads % num_group == 0

    const float _scale = scale == 0.f ? 1.f / sqrt(embed_dim) : scale;
    const int num_heads_per_group = num_heads / num_group;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    Mat query_int8(embed_dim, src_seqlen, opt.num_threads, 1u, opt.workspace_allocator);
    if (query_int8.empty())
        return -100;

    Mat key_int8(embed_dim, dst_seqlen, opt.num_threads, 1u, opt.workspace_allocator);
    if (key_int8.empty())
        return -100;

    Mat value_int8(out_embed_dim, dst_seqlen, opt.num_threads, 1u, opt.workspace_allocator);
    if (value_int8.empty())
        return -100;

    Mat qk_cross(dst_seqlen, src_seqlen, opt.num_threads, 4u, opt.workspace_allocator);
    if (qk_cross.empty())
        return -100;

    Mat qk_cross_int8(dst_seqlen, src_seqlen, opt.num_threads, 1u, opt.workspace_allocator);
    if (qk_cross_int8.empty())
        return -100;

    Mat query_or_qk_cross_int8_scales(src_seqlen, 1, opt.num_threads, 4u, opt.workspace_allocator);
    if (query_or_qk_cross_int8_scales.empty())
        return -100;

    Mat key = cur_key;
    if (past_seqlen > 0)
    {
        key.create(embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
        if (key.empty())
            return -100;

        // concat
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

    Mat value = cur_value;
    if (past_seqlen > 0)
    {
        value.create(out_embed_dim, dst_seqlen, num_group, 4u, opt.blob_allocator);
        if (value.empty())
            return -100;

        // concat
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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat query_head = query.channel(q);
        const Mat key_head = key.channel(q / num_heads_per_group);
        const Mat value_head = value.channel(q / num_heads_per_group);
        Mat qk_cross_head = qk_cross.channel(get_omp_thread_num());
        Mat top_blob_head = top_blob.channel(q);

        // qk_cross
        {
            // dynamic quantize query_head per h
            Mat query_head_int8 = query_int8.channel(get_omp_thread_num());
            Mat query_head_int8_scales = query_or_qk_cross_int8_scales.channel(get_omp_thread_num());
            dynamic_quantize_2d_per_h(query_head, query_head_int8, query_head_int8_scales);

            // dynamic quantize key_head
            Mat key_head_int8 = key_int8.channel(get_omp_thread_num());
            float key_head_int8_scale;
            dynamic_quantize_2d(key_head, key_head_int8, key_head_int8_scale);

            for (int i = 0; i < src_seqlen; i++)
            {
                const signed char* qptr = query_head_int8.row<const signed char>(i);
                float* outptr = qk_cross_head.row(i);

                const float qk_descale = 1.f / (query_head_int8_scales[i] * key_head_int8_scale);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    const signed char* kptr = key_head_int8.row<const signed char>(j);

                    int sum = 0;
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += qptr[k] * kptr[k];
                    }
                    float sum_fp32 = sum * qk_descale;

                    outptr[j] = sum_fp32 * _scale;
                }
            }
        }

        if (attn_mask)
        {
            const Mat& maskm = attn_mask_blob.c > 1 ? attn_mask_blob.channel(q) : attn_mask_blob;

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

        // softmax
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

        // qkv_cross
        {
            // dynamic quantize qk_cross_head per h
            Mat qk_cross_head_int8 = qk_cross_int8.channel(get_omp_thread_num());
            Mat qk_cross_head_int8_scales = query_or_qk_cross_int8_scales.channel(get_omp_thread_num());
            dynamic_quantize_2d_per_h(qk_cross_head, qk_cross_head_int8, qk_cross_head_int8_scales);

            // dynamic quantize value_head
            Mat value_head_int8 = value_int8.channel(get_omp_thread_num());
            float value_head_int8_scale;
            dynamic_quantize_2d(value_head, value_head_int8, value_head_int8_scale);

            for (int i = 0; i < src_seqlen; i++)
            {
                const signed char* qkptr = qk_cross_head_int8.row<const signed char>(i);
                float* outptr = top_blob_head.row(i);

                const float qkv_descale = 1.f / (qk_cross_head_int8_scales[i] * value_head_int8_scale);

                for (int j = 0; j < out_embed_dim; j++)
                {
                    int sum = 0;
                    for (int k = 0; k < dst_seqlen; k++)
                    {
                        sum += qkptr[k] * value_head_int8.row<const signed char>(k)[j];
                    }
                    float sum_fp32 = sum * qkv_descale;

                    outptr[j] = sum_fp32;
                }
            }
        }
    }

    if (kv_cache)
    {
        // assert top_blobs.size() == 3
        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
