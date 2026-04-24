// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_x86.h"

#include "layer_type.h"

#include "cpu.h"

#include <float.h>
#include <math.h>

namespace ncnn {

SDPA_x86::SDPA_x86()
{
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int SDPA_x86::create_pipeline(const Option& /*_opt*/)
{
    if (int8_scale_term)
    {
        support_bf16_storage = false;
    }

    return 0;
}

int SDPA_x86::destroy_pipeline(const Option& /*_opt*/)
{
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

static void dynamic_quantize_blockwise(const float* src, signed char* dst, float* scales, int width)
{
    const int block_size = 32;
    int num_blocks = (width + block_size - 1) / block_size;
    for (int b = 0; b < num_blocks; b++)
    {
        int start = b * block_size;
        int end = start + block_size < width ? start + block_size : width;
        float absmax = 0.f;
        for (int i = start; i < end; i++)
        {
            absmax = std::max(absmax, (float)fabs(src[i]));
        }
        float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        scales[b] = scale;
        for (int i = start; i < end; i++)
        {
            dst[i] = float2int8(src[i] * scale);
        }
    }
}
#endif // NCNN_INT8

int SDPA_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& _opt) const
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
    const float _scale = scale == 0.f ? 1.f / sqrtf((float)embed_dim) : scale;

    const int BLOCK_M = 64;
    const int BLOCK_N = 64;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if NCNN_INT8
    if (int8_scale_term)
    {
        const int qk_num_blocks = (embed_dim + 31) / 32;
        const int v_num_blocks = (out_embed_dim + 31) / 32;

        Mat key_int8(embed_dim, dst_seqlen, num_group, 1u, opt.blob_allocator);
        Mat key_scales(qk_num_blocks, dst_seqlen, num_group, 4u, opt.blob_allocator);
        Mat value_int8(out_embed_dim, dst_seqlen, num_group, 1u, opt.blob_allocator);
        Mat value_scales(v_num_blocks, dst_seqlen, num_group, 4u, opt.blob_allocator);

        if (key_int8.empty() || key_scales.empty() || value_int8.empty() || value_scales.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < num_group; g++)
        {
            const Mat key_head = key.channel(g);
            Mat key_int8_head = key_int8.channel(g);
            Mat key_scales_head = key_scales.channel(g);
            for (int j = 0; j < dst_seqlen; j++)
            {
                dynamic_quantize_blockwise(key_head.row(j), key_int8_head.row<signed char>(j), key_scales_head.row(j), embed_dim);
            }

            const Mat value_head = value.channel(g);
            Mat value_int8_head = value_int8.channel(g);
            Mat value_scales_head = value_scales.channel(g);
            for (int j = 0; j < dst_seqlen; j++)
            {
                dynamic_quantize_blockwise(value_head.row(j), value_int8_head.row<signed char>(j), value_scales_head.row(j), out_embed_dim);
            }
        }

        Mat o_accum(out_embed_dim, BLOCK_M, opt.num_threads, 4u, opt.workspace_allocator);
        Mat s_vec(BLOCK_N, opt.num_threads, 4u, opt.workspace_allocator);
        Mat q_int8_tile(embed_dim, BLOCK_M, opt.num_threads, 1u, opt.workspace_allocator);
        Mat q_scales_tile(qk_num_blocks, BLOCK_M, opt.num_threads, 4u, opt.workspace_allocator);

        if (o_accum.empty() || s_vec.empty() || q_int8_tile.empty() || q_scales_tile.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const Mat query_head = query.channel(q);
            const Mat key_int8_head = key_int8.channel(q / num_heads_per_group);
            const Mat key_scales_head = key_scales.channel(q / num_heads_per_group);
            const Mat value_int8_head = value_int8.channel(q / num_heads_per_group);
            const Mat value_scales_head = value_scales.channel(q / num_heads_per_group);
            Mat top_blob_head = top_blob.channel(q);

            Mat mask_head;
            if (attn_mask)
            {
                const Mat& maskm = attn_mask_blob;
                if (maskm.dims == 3)
                {
                    mask_head = maskm.c > 1 ? maskm.channel(q) : maskm.channel(0);
                }
                else
                {
                    mask_head = maskm;
                }
            }

            Mat o_accum_head = o_accum.channel(get_omp_thread_num());
            float* s_vec_ptr = s_vec.row(get_omp_thread_num());
            Mat q_int8_tile_head = q_int8_tile.channel(get_omp_thread_num());
            Mat q_scales_tile_head = q_scales_tile.channel(get_omp_thread_num());

            for (int m_start = 0; m_start < src_seqlen; m_start += BLOCK_M)
            {
                int m_end = m_start + BLOCK_M < src_seqlen ? m_start + BLOCK_M : src_seqlen;
                int block_m = m_end - m_start;

                for (int i = 0; i < block_m; i++)
                {
                    dynamic_quantize_blockwise(query_head.row(m_start + i), q_int8_tile_head.row<signed char>(i), q_scales_tile_head.row(i), embed_dim);
                }

                for (int i = 0; i < block_m; i++)
                {
                    float* optr = o_accum_head.row(i);
                    for (int k = 0; k < out_embed_dim; k++)
                    {
                        optr[k] = 0.f;
                    }
                }

                float m_vec[64];
                float l_vec[64];
                for (int i = 0; i < block_m; i++)
                {
                    m_vec[i] = -FLT_MAX;
                    l_vec[i] = 0.f;
                }

                for (int n_start = 0; n_start < dst_seqlen; n_start += BLOCK_N)
                {
                    int n_end = n_start + BLOCK_N < dst_seqlen ? n_start + BLOCK_N : dst_seqlen;
                    int block_n = n_end - n_start;

                    for (int i = 0; i < block_m; i++)
                    {
                        const signed char* qptr = q_int8_tile_head.row<const signed char>(i);
                        const float* qscales = q_scales_tile_head.row(i);

                        for (int j = 0; j < block_n; j++)
                        {
                            const signed char* kptr = key_int8_head.row<const signed char>(n_start + j);
                            const float* kscales = key_scales_head.row(n_start + j);

                            float sum = 0.f;
                            for (int b = 0; b < qk_num_blocks; b++)
                            {
                                int k_start = b * 32;
                                int k_end = k_start + 32 < embed_dim ? k_start + 32 : embed_dim;
                                int block_sum = 0;
                                for (int k = k_start; k < k_end; k++)
                                {
                                    block_sum += qptr[k] * kptr[k];
                                }
                                sum += (float)block_sum / (qscales[b] * kscales[b]);
                            }
                            s_vec_ptr[j] = sum * _scale;
                        }

                        if (attn_mask)
                        {
                            const float* mptr = mask_head.row(m_start + i) + n_start;
                            for (int j = 0; j < block_n; j++)
                            {
                                s_vec_ptr[j] += mptr[j];
                            }
                        }

                        float m_new = m_vec[i];
                        for (int j = 0; j < block_n; j++)
                        {
                            m_new = std::max(m_new, s_vec_ptr[j]);
                        }

                        float scale_factor = expf(m_vec[i] - m_new);
                        float l_new = l_vec[i] * scale_factor;

                        float* optr = o_accum_head.row(i);
                        for (int k = 0; k < out_embed_dim; k++)
                        {
                            optr[k] *= scale_factor;
                        }

                        for (int j = 0; j < block_n; j++)
                        {
                            float p = expf(s_vec_ptr[j] - m_new);
                            l_new += p;

                            const signed char* vptr = value_int8_head.row<const signed char>(n_start + j);
                            const float* vscales = value_scales_head.row(n_start + j);
                            for (int vb = 0; vb < v_num_blocks; vb++)
                            {
                                float inv_scale = 1.f / vscales[vb];
                                int k_start = vb * 32;
                                int k_end = k_start + 32 < out_embed_dim ? k_start + 32 : out_embed_dim;
                                for (int k = k_start; k < k_end; k++)
                                {
                                    optr[k] += p * (float)vptr[k] * inv_scale;
                                }
                            }
                        }

                        m_vec[i] = m_new;
                        l_vec[i] = l_new;
                    }
                }

                for (int i = 0; i < block_m; i++)
                {
                    float* optr = o_accum_head.row(i);
                    float* outptr = top_blob_head.row(m_start + i);
                    float inv_l = 1.f / l_vec[i];
                    for (int k = 0; k < out_embed_dim; k++)
                    {
                        outptr[k] = optr[k] * inv_l;
                    }
                }
            }
        }

        if (kv_cache)
        {
            top_blobs[1] = key;
            top_blobs[2] = value;
        }

        return 0;
    }
#endif // NCNN_INT8

    Mat o_accum(out_embed_dim, BLOCK_M, opt.num_threads, 4u, opt.workspace_allocator);
    Mat s_vec(BLOCK_N, opt.num_threads, 4u, opt.workspace_allocator);

    if (o_accum.empty() || s_vec.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat query_head = query.channel(q);
        const Mat key_head = key.channel(q / num_heads_per_group);
        const Mat value_head = value.channel(q / num_heads_per_group);
        Mat top_blob_head = top_blob.channel(q);

        Mat mask_head;
        if (attn_mask)
        {
            const Mat& maskm = attn_mask_blob;
            if (maskm.dims == 3)
            {
                mask_head = maskm.c > 1 ? maskm.channel(q) : maskm.channel(0);
            }
            else
            {
                mask_head = maskm;
            }
        }

        Mat o_accum_head = o_accum.channel(get_omp_thread_num());
        float* s_vec_ptr = s_vec.row(get_omp_thread_num());

        for (int m_start = 0; m_start < src_seqlen; m_start += BLOCK_M)
        {
            int m_end = m_start + BLOCK_M < src_seqlen ? m_start + BLOCK_M : src_seqlen;
            int block_m = m_end - m_start;

            for (int i = 0; i < block_m; i++)
            {
                float* optr = o_accum_head.row(i);
                for (int k = 0; k < out_embed_dim; k++)
                {
                    optr[k] = 0.f;
                }
            }

            float m_vec[64];
            float l_vec[64];
            for (int i = 0; i < block_m; i++)
            {
                m_vec[i] = -FLT_MAX;
                l_vec[i] = 0.f;
            }

            for (int n_start = 0; n_start < dst_seqlen; n_start += BLOCK_N)
            {
                int n_end = n_start + BLOCK_N < dst_seqlen ? n_start + BLOCK_N : dst_seqlen;
                int block_n = n_end - n_start;

                for (int i = 0; i < block_m; i++)
                {
                    const float* qptr = query_head.row(m_start + i);

                    for (int j = 0; j < block_n; j++)
                    {
                        const float* kptr = key_head.row(n_start + j);
                        float sum = 0.f;
                        for (int k = 0; k < embed_dim; k++)
                        {
                            sum += qptr[k] * kptr[k];
                        }
                        s_vec_ptr[j] = sum * _scale;
                    }

                    if (attn_mask)
                    {
                        const float* mptr = mask_head.row(m_start + i) + n_start;
                        for (int j = 0; j < block_n; j++)
                        {
                            s_vec_ptr[j] += mptr[j];
                        }
                    }

                    float m_new = m_vec[i];
                    for (int j = 0; j < block_n; j++)
                    {
                        m_new = std::max(m_new, s_vec_ptr[j]);
                    }

                    float scale_factor = expf(m_vec[i] - m_new);
                    float l_new = l_vec[i] * scale_factor;

                    float* optr = o_accum_head.row(i);
                    for (int k = 0; k < out_embed_dim; k++)
                    {
                        optr[k] *= scale_factor;
                    }

                    for (int j = 0; j < block_n; j++)
                    {
                        float p = expf(s_vec_ptr[j] - m_new);
                        l_new += p;

                        const float* vptr = value_head.row(n_start + j);
                        for (int k = 0; k < out_embed_dim; k++)
                        {
                            optr[k] += p * vptr[k];
                        }
                    }

                    m_vec[i] = m_new;
                    l_vec[i] = l_new;
                }
            }

            for (int i = 0; i < block_m; i++)
            {
                float* optr = o_accum_head.row(i);
                float* outptr = top_blob_head.row(m_start + i);
                float inv_l = 1.f / l_vec[i];
                for (int k = 0; k < out_embed_dim; k++)
                {
                    outptr[k] = optr[k] * inv_l;
                }
            }
        }
    }

    if (kv_cache)
    {
        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}

} // namespace ncnn
