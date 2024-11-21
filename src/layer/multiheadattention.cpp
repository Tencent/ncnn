// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

    const Mat& q_blob = bottom_blobs[0];
    const Mat& k_blob = (bottom_blobs.size() == 1 || (bottom_blobs.size() == 2 && attn_mask)) ? q_blob : bottom_blobs[1];
    const Mat& v_blob = (bottom_blobs.size() == 1 || (bottom_blobs.size() == 2 && attn_mask)) ? q_blob : (bottom_blobs.size() == 2 || (bottom_blobs.size() == 3 && attn_mask)) ? k_blob : bottom_blobs[2];
    const Mat& attn_mask_blob = attn_mask ? bottom_blobs[bottom_blobs.size() - 1] : Mat();

    const int src_seqlen = q_blob.h;
    const int dst_seqlen = k_blob.h;
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
    Mat xk(embed_dim_per_head, dst_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xk.empty())
        return -100;
    Mat xv(dst_seqlen, embed_dim_per_head, num_heads, 4u, opt.workspace_allocator);
    if (xv.empty())
        return -100;

    Mat xqk(dst_seqlen, src_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xqk.empty())
        return -100;

    Mat xqkv(embed_dim_per_head, num_heads, src_seqlen, 4u, opt.workspace_allocator);
    if (xqkv.empty())
        return -100;

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
        {
            Mat outm = xk.channel(q);

            for (int i = 0; i < dst_seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* ptr = k_blob.row(i);
                    const float* kptr = (const float*)k_weight_data + kdim * (q * embed_dim_per_head + j);

                    float sum = k_bias_data[q * embed_dim_per_head + j];
                    for (int k = 0; k < kdim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    outptr[j] = sum;
                }
            }
        }

        // xv = affine(v)
        {
            Mat outm = xv.channel(q);

            for (int i = 0; i < embed_dim_per_head; i++)
            {
                for (int j = 0; j < dst_seqlen; j++)
                {
                    const float* ptr = v_blob.row(j);
                    const float* kptr = (const float*)v_weight_data + vdim * (q * embed_dim_per_head + i);

                    float sum = v_bias_data[q * embed_dim_per_head + i];
                    for (int k = 0; k < vdim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }

                    float* outptr = outm.row(i);

                    outptr[j] = sum;
                }
            }
        }

        // xqk = xq * xk
        // xq  (embed_dim_per_head, src_seqlen)
        // xk  (embed_dim_per_head, dst_seqlen)
        {
            const Mat xqm = xq.channel(q);
            const Mat xkm = xk.channel(q);

            Mat outm = xqk.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* outptr = outm.row(i);

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
            Mat outm = xqk.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                const float* mptr = maskm.row(i);
                float* outptr = outm.row(i);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    outptr[j] += mptr[j];
                }
            }
        }

        // softmax(xqk)
        {
            Mat outm = xqk.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* ptr = outm.row(i);

                float max = -FLT_MAX;
                for (int j = 0; j < dst_seqlen; j++)
                {
                    max = std::max(max, ptr[j]);
                }

                float sum = 0.f;
                for (int j = 0; j < dst_seqlen; j++)
                {
                    ptr[j] = (float)(expf(ptr[j] - max));
                    sum += ptr[j];
                }

                for (int j = 0; j < dst_seqlen; j++)
                {
                    ptr[j] /= sum;
                }
            }
        }

        // xqkv = xqk * xv
        // xqk (dst_seqlen, src_seqlen)
        // xv  (dst_seqlen, embed_dim_per_head)
        // out (embed_dim_per_head, num_heads, src_seqlen)
        {
            const Mat xqkm = xqk.channel(q);
            const Mat xvm = xv.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* outptr = xqkv.channel(i).row(q);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* qkptr = xqkm.row(i);
                    const float* vptr = xvm.row(j);

                    float sum = 0.f;
                    for (int k = 0; k < dst_seqlen; k++)
                    {
                        sum += *qkptr++ * *vptr++;
                    }

                    outptr[j] = sum;
                }
            }
        }
    }

    // out = affine(xqkv)
    // xqkv  (embed_dim, src_seqlen)
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
    const Mat& q_blob = bottom_blobs[0];
    const Mat& k_blob = (bottom_blobs.size() == 1 || (bottom_blobs.size() == 2 && attn_mask)) ? q_blob : bottom_blobs[1];
    const Mat& v_blob = (bottom_blobs.size() == 1 || (bottom_blobs.size() == 2 && attn_mask)) ? q_blob : (bottom_blobs.size() == 2 || (bottom_blobs.size() == 3 && attn_mask)) ? k_blob : bottom_blobs[2];
    const Mat& attn_mask_blob = attn_mask ? bottom_blobs[bottom_blobs.size() - 1] : Mat();

    const int src_seqlen = q_blob.h;
    const int dst_seqlen = k_blob.h;
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
    Mat xk(embed_dim_per_head, dst_seqlen, num_heads, 4u, opt.workspace_allocator);
    if (xk.empty())
        return -100;
    Mat xv(dst_seqlen, embed_dim_per_head, num_heads, 4u, opt.workspace_allocator);
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
    if (bottom_blobs.size() == 1)
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
    if (bottom_blobs.size() == 1)
    {
        v_blob_int8 = q_blob_int8;
        v_blob_int8_scale = q_blob_int8_scale;
    }
    else if (bottom_blobs.size() == 2)
    {
        v_blob_int8 = k_blob_int8;
        v_blob_int8_scale = k_blob_int8_scale;
    }
    else
    {
        dynamic_quantize_2d(v_blob, v_blob_int8, v_blob_int8_scale, opt);
    }

    // NCNN_LOGE("%.4f %.4f", q_weight_data_int8_scale, q_blob_int8_scale);

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
        {
            float* outptr = xk.channel(q);

            for (int i = 0; i < k_blob_int8.h; i++)
            {
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

                    *outptr++ = sum_fp32;
                }
            }
        }

        // xv = affine(v)
        {
            Mat outm = xv.channel(q);

            for (int i = 0; i < embed_dim_per_head; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < v_blob_int8.h; j++)
                {
                    const signed char* ptr = v_blob_int8.row<const signed char>(j);
                    const signed char* kptr = (const signed char*)v_weight_data + vdim * (q * embed_dim_per_head + i);

                    int sum = 0;
                    for (int k = 0; k < vdim; k++)
                    {
                        sum += *ptr++ * *kptr++;
                    }
                    const float v_descale = 1.f / (v_weight_data_int8_scales[q * embed_dim_per_head + i] * v_blob_int8_scale);
                    float sum_fp32 = sum * v_descale + v_bias_data[q * embed_dim_per_head + i];

                    *outptr++ = sum_fp32;
                }
            }
        }

        // xqk = xq * xk
        // xq  (embed_dim_per_head, src_seqlen)
        // xk  (embed_dim_per_head, dst_seqlen)
        {
            const Mat xqm = xq.channel(q);
            const Mat xkm = xk.channel(q);

            Mat outm = xqk.channel(q);

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
                float* outptr = outm.row(i);
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
            Mat outm = xqk.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                const float* mptr = maskm.row(i);
                float* outptr = outm.row(i);

                for (int j = 0; j < dst_seqlen; j++)
                {
                    outptr[j] += mptr[j];
                }
            }
        }

        // softmax(xqk)
        {
            Mat outm = xqk.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                float* ptr = outm.row(i);

                float max = -FLT_MAX;
                for (int j = 0; j < dst_seqlen; j++)
                {
                    max = std::max(max, ptr[j]);
                }

                float sum = 0.f;
                for (int j = 0; j < dst_seqlen; j++)
                {
                    ptr[j] = (float)(expf(ptr[j] - max));
                    sum += ptr[j];
                }

                for (int j = 0; j < dst_seqlen; j++)
                {
                    ptr[j] /= sum;
                }
            }
        }

        // xqkv = xqk * xv
        // xqk (dst_seqlen, src_seqlen)
        // xv  (dst_seqlen, embed_dim_per_head)
        // out (embed_dim_per_head, num_heads, src_seqlen)
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
                    const signed char* vptr = xvm_int8.row<const signed char>(j);

                    int sum = 0;
                    for (int k = 0; k < dst_seqlen; k++)
                    {
                        sum += *qkptr++ * *vptr++;
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
    // xqkv  (embed_dim, src_seqlen)
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

    return 0;
}
#endif

} // namespace ncnn
