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

#include "multiheadattention_arm.h"

#include <float.h>
#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

MultiHeadAttention_arm::MultiHeadAttention_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

#if __ARM_NEON
inline float sum_float32x4(float32x4_t _sum)
{
    float sum = 0.f;
#if __aarch64__
    sum += vaddvq_f32(_sum);
#else
    float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
    float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
    sum += vget_lane_f32(_ss2, 0);
#endif
    return sum;
}

inline float max_float32x4(float max, float32x4_t _max)
{
#if __aarch64__
    max = std::max(max, vmaxvq_f32(_max));
#else
    float32x2_t _max2 = vmax_f32(vget_low_f32(_max), vget_high_f32(_max));
    float32x2_t _mm2 = vpmax_f32(_max2, _max2);
    max = std::max(max, vget_lane_f32(_mm2, 0));
#endif
    return max;
}

#endif

int MultiHeadAttention_arm::create_pipeline(const Option& opt)
{
#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        return create_pipeline_bf16s(opt);
    }
#endif

    return 0;
}

int MultiHeadAttention_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& q_blob = bottom_blobs[0];

    size_t elemsize = q_blob.elemsize;
    int elempack = q_blob.elempack;
    int elembits = q_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blobs, top_blobs, opt);
        else
            return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    const Mat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const Mat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[2];

    const int seqlen = q_blob.h;
    const int embed_dim_per_head = embed_dim / num_head;
    const float inv_sqrt_embed_dim_per_head = 1.f / sqrt(embed_dim_per_head);

#if __ARM_NEON
    if (elempack == 4)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create(embed_dim, seqlen, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -1;

        Mat xq(embed_dim_per_head, seqlen, num_head, elemsize, elempack, opt.workspace_allocator);
        Mat xk(embed_dim_per_head, seqlen, num_head, elemsize, elempack, opt.workspace_allocator);
        Mat xv(seqlen, embed_dim_per_head, num_head, elemsize, elempack, opt.workspace_allocator);

        Mat xqk(seqlen * elempack, seqlen, num_head, elemsize, elempack, opt.workspace_allocator);

        Mat xqkv(embed_dim_per_head, num_head, seqlen, elemsize, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_head; q++)
        {
            // xq = affine(q) * inv_sqrt_embed_dim_per_head
            {
                float* outptr = (float*)xq.channel(q);

                const float* bptr0 = (const float*)q_bias_data + q * embed_dim_per_head;
                const float* kptr0 = (const float*)q_weight_data + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const float* bptr = bptr0;
                    const float* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const float* ptr = q_blob.row(i);

                        float32x4_t _sum = vdupq_n_f32(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vld1q_f32(ptr);
                            float32x4_t _k = vdupq_n_f32(*kptr);
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        float32x4_t _slope = vdupq_n_f32(inv_sqrt_embed_dim_per_head);
                        _sum = vmulq_f32(_sum, _slope);

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                        bptr++;
                    }
                }
            }

            // xk = affine(k)
            {
                float* outptr = (float*)xk.channel(q);

                const float* bptr0 = (const float*)k_bias_data + q * embed_dim_per_head;
                const float* kptr0 = (const float*)k_weight_data + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const float* bptr = bptr0;
                    const float* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const float* ptr = k_blob.row(i);

                        float32x4_t _sum = vdupq_n_f32(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vld1q_f32(ptr);
                            float32x4_t _k = vdupq_n_f32(*kptr);
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                        bptr++;
                    }
                }
            }

            // xv = affine(v)
            {
                float* outptr = (float*)xv.channel(q);

                const float* bptr = (const float*)v_bias_data + q * embed_dim_per_head;
                const float* kptr0 = (const float*)v_weight_data + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < embed_dim_per_head; i++)
                {
                    for (int j = 0; j < seqlen; j++)
                    {
                        const float* ptr = v_blob.row(j);
                        const float* kptr = kptr0;

                        float32x4_t _sum = vdupq_n_f32(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vld1q_f32(ptr);
                            float32x4_t _k = vdupq_n_f32(*kptr);
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                    }

                    bptr++;
                    kptr0 += embed_dim;
                }
            }

            // xqk = xq * xk
            // xq  (embed_dim_per_head, seqlen)
            // xk  (embed_dim_per_head, seqlen)
            {
                const Mat xqm = xq.channel(q);
                const Mat xkm = xk.channel(q);

                Mat outm = xqk.channel(q);

                Mat upxkm;
                convert_packing(xkm, upxkm, 1);

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = outm.row(i);

                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        const float* qptr = xqm.row(i);
                        const float* kptr = upxkm.row(j);

                        float32x4_t _sum = vdupq_n_f32(0.f);
                        for (int k = 0; k < embed_dim_per_head; k++)
                        {
                            float32x4_t _q = vld1q_f32(qptr);
                            float32x4_t _k = vdupq_n_f32(*kptr);
                            _sum = vmlaq_f32(_sum, _q, _k);
                            qptr += 4;
                            kptr += 1;
                        }

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                    }
                }
            }

            // softmax(xqk)
            {
                Mat outm = xqk.channel(q);
                for (int i = 0; i < seqlen; i++)
                {
                    float* ptr = outm.row(i);

                    float* ptr0 = ptr;
                    float32x4_t _max = vdupq_n_f32(-FLT_MAX);
                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        _max = vmaxq_f32(_max, _p);
                        ptr0 += 4;
                    }

                    ptr0 = ptr;
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        _p = exp_ps(vsubq_f32(_p, _max));
                        vst1q_f32(ptr0, _p);
                        _sum = vaddq_f32(_sum, _p);
                        ptr0 += 4;
                    }

                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = div_ps(_p, _sum);
                        vst1q_f32(ptr, _p);
                        ptr += 4;
                    }
                }
            }

            // xqkv = xqk * xv
            // xqk (seqlen, seqlen)
            // xv  (seqlen, embed_dim_per_head)
            // out (embed_dim_per_head, num_head, seqlen)
            {
                const Mat xqkm = xqk.channel(q);
                const Mat xvm = xv.channel(q);

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = xqkv.channel(i).row(q);

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const float* qkptr = xqkm.row(i);
                        const float* vptr = xvm.row(j);

                        float32x4_t _sum = vdupq_n_f32(0.f);
                        for (int k = 0; k < seqlen * elempack; k++)
                        {
                            float32x4_t _qk = vld1q_f32(qkptr);
                            float32x4_t _v = vdupq_n_f32(*vptr);
                            _sum = vmlaq_f32(_sum, _qk, _v);
                            qkptr += 4;
                            vptr += 1;
                        }

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                    }
                }
            }
        }

        // out = affine(xqkv)
        // xqkv  (embed_dim, seqlen)
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < seqlen; i++)
        {
            float* outptr = top_blob.row(i);

            const float* bptr = (const float*)out_bias_data;
            const float* kptr = (const float*)out_weight_data;

            for (int j = 0; j < embed_dim; j++)
            {
                const float* ptr = xqkv.channel(i);

                float32x4_t _sum = vdupq_n_f32(*bptr);
                for (int k = 0; k < embed_dim; k++)
                {
                    float32x4_t _val = vld1q_f32(ptr);
                    float32x4_t _k = vdupq_n_f32(*kptr);
                    _sum = vmlaq_f32(_sum, _val, _k);
                    ptr += 4;
                    kptr += 1;
                }

                vst1q_f32(outptr, _sum);
                outptr += 4;
                bptr++;
            }
        }

        return 0;
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(embed_dim, seqlen, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -1;

    Mat xq(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
    Mat xk(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
    Mat xv(seqlen, embed_dim_per_head, num_head, 4u, opt.workspace_allocator);

    Mat xqk(seqlen, seqlen, num_head, 4u, opt.workspace_allocator);

    Mat xqkv(embed_dim_per_head, num_head, seqlen, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_head; q++)
    {
        // xq = affine(q) * inv_sqrt_embed_dim_per_head
        {
            float* outptr = (float*)xq.channel(q);

            const float* bptr0 = (const float*)q_bias_data + q * embed_dim_per_head;
            const float* kptr0 = (const float*)q_weight_data + embed_dim * q * embed_dim_per_head;

            for (int i = 0; i < seqlen; i++)
            {
                const float* bptr = bptr0;
                const float* kptr = kptr0;

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* ptr = q_blob.row(i);

                    float sum = *bptr;
                    int k = 0;
                    float32x4_t _sum = vdupq_n_f32(0);
                    for (; k + 3 < embed_dim; k += 4)
                    {
                        float32x4_t _val = vld1q_f32(ptr);
                        float32x4_t _k = vld1q_f32(kptr);
                        _sum = vmlaq_f32(_sum, _val, _k);
                        ptr += 4;
                        kptr += 4;
                    }
                    sum += sum_float32x4(_sum);
                    for (; k < embed_dim; k++)
                    {
                        sum += *ptr * *kptr;
                        ptr++;
                        kptr++;
                    }

                    *outptr = sum * inv_sqrt_embed_dim_per_head;
                    outptr++;
                    bptr++;
                }
            }
        }

        // xk = affine(k)
        {
            float* outptr = (float*)xk.channel(q);

            const float* bptr0 = (const float*)k_bias_data + q * embed_dim_per_head;
            const float* kptr0 = (const float*)k_weight_data + embed_dim * q * embed_dim_per_head;

            for (int i = 0; i < seqlen; i++)
            {
                const float* bptr = bptr0;
                const float* kptr = kptr0;

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* ptr = k_blob.row(i);

                    float sum = *bptr;
                    int k = 0;
                    float32x4_t _sum = vdupq_n_f32(0);
                    for (; k + 3 < embed_dim; k += 4)
                    {
                        float32x4_t _val = vld1q_f32(ptr);
                        float32x4_t _k = vld1q_f32(kptr);
                        _sum = vmlaq_f32(_sum, _val, _k);
                        ptr += 4;
                        kptr += 4;
                    }
                    sum += sum_float32x4(_sum);
                    for (; k < embed_dim; k++)
                    {
                        sum += *ptr * *kptr;
                        ptr++;
                        kptr++;
                    }

                    *outptr = sum;
                    outptr++;
                    bptr++;
                }
            }
        }

        // xv = affine(v)
        {
            float* outptr = (float*)xv.channel(q);

            const float* bptr = (const float*)v_bias_data + q * embed_dim_per_head;
            const float* kptr0 = (const float*)v_weight_data + embed_dim * q * embed_dim_per_head;

            for (int i = 0; i < embed_dim_per_head; i++)
            {
                for (int j = 0; j < seqlen; j++)
                {
                    const float* ptr = v_blob.row(j);
                    const float* kptr = kptr0;

                    float sum = *bptr;
                    int k = 0;
                    float32x4_t _sum = vdupq_n_f32(0);
                    for (; k + 3 < embed_dim; k += 4)
                    {
                        float32x4_t _val = vld1q_f32(ptr);
                        float32x4_t _k = vld1q_f32(kptr);
                        _sum = vmlaq_f32(_sum, _val, _k);
                        ptr += 4;
                        kptr += 4;
                    }
                    sum += sum_float32x4(_sum);
                    for (; k < embed_dim; k++)
                    {
                        sum += *ptr * *kptr;
                        ptr++;
                        kptr++;
                    }

                    *outptr = sum;
                    outptr++;
                }

                bptr++;
                kptr0 += embed_dim;
            }
        }

        // xqk = xq * xk
        // xq  (embed_dim_per_head, seqlen)
        // xk  (embed_dim_per_head, seqlen)
        {
            const Mat xqm = xq.channel(q);
            const Mat xkm = xk.channel(q);

            Mat outm = xqk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < seqlen; j++)
                {
                    const float* qptr = xqm.row(i);
                    const float* kptr = xkm.row(j);

                    float sum = 0.f;
                    int k = 0;
                    float32x4_t _sum = vdupq_n_f32(0);
                    for (; k + 3 < embed_dim_per_head; k += 4)
                    {
                        float32x4_t _val = vld1q_f32(qptr);
                        float32x4_t _k = vld1q_f32(kptr);
                        _sum = vmlaq_f32(_sum, _val, _k);
                        qptr += 4;
                        kptr += 4;
                    }
                    sum += sum_float32x4(_sum);
                    for (; k < embed_dim_per_head; k++)
                    {
                        sum += *qptr * *kptr;
                        qptr++;
                        kptr++;
                    }

                    *outptr = sum;
                    outptr++;
                }
            }
        }

        // softmax(xqk)
        {
            Mat outm = xqk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* ptr = outm.row(i);

                float* ptr0 = ptr;
                float max = -FLT_MAX;
                int j = 0;
                float32x4_t _max = vdupq_n_f32(-FLT_MAX);
                for (; j + 3 < seqlen; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr0);
                    _max = vmaxq_f32(_max, _p);
                    ptr0 += 4;
                }
                max = max_float32x4(max, _max);
                for (; j < seqlen; j++)
                {
                    max = std::max(max, *ptr0);
                    ptr0++;
                }

                ptr0 = ptr;
                float sum = 0.f;
                j = 0;
                float32x4_t _sum = vdupq_n_f32(0.f);
                _max = vdupq_n_f32(max);
                for (; j + 3 < seqlen; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr0);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1q_f32(ptr0, _p);
                    _sum = vaddq_f32(_sum, _p);
                    ptr0 += 4;
                }
                sum += sum_float32x4(_sum);
                for (; j < seqlen; j++)
                {
                    *ptr0 = (float)(exp(*ptr0 - max));
                    sum += *ptr0;
                    ptr0++;
                }

                j = 0;
                _sum = vdupq_n_f32(sum);
                for (; j + 3 < seqlen; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    _p = div_ps(_p, _sum);
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                }
                for (; j < seqlen; j++)
                {
                    *ptr /= sum;
                    ptr++;
                }
            }
        }

        // xqkv = xqk * xv
        // xqk (seqlen, seqlen)
        // xv  (seqlen, embed_dim_per_head)
        // out (embed_dim_per_head, num_head, seqlen)
        {
            const Mat xqkm = xqk.channel(q);
            const Mat xvm = xv.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = xqkv.channel(i).row(q);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* qkptr = xqkm.row(i);
                    const float* vptr = xvm.row(j);

                    float sum = 0.f;
                    int k = 0;
                    float32x4_t _sum = vdupq_n_f32(0);
                    for (; k + 3 < seqlen; k += 4)
                    {
                        float32x4_t _val = vld1q_f32(qkptr);
                        float32x4_t _k = vld1q_f32(vptr);
                        _sum = vmlaq_f32(_sum, _val, _k);
                        qkptr += 4;
                        vptr += 4;
                    }
                    sum += sum_float32x4(_sum);
                    for (; k < seqlen; k++)
                    {
                        sum += *qkptr * *vptr;
                        qkptr++;
                        vptr++;
                    }

                    *outptr = sum;
                    outptr++;
                }
            }
        }
    }

    // out = affine(xqkv)
    // xqkv  (embed_dim, seqlen)
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < seqlen; i++)
    {
        float* outptr = top_blob.row(i);

        const float* bptr = (const float*)out_bias_data;
        const float* kptr = (const float*)out_weight_data;

        for (int j = 0; j < embed_dim; j++)
        {
            const float* ptr = xqkv.channel(i);

            float sum = *bptr;
            int k = 0;
            float32x4_t _sum = vdupq_n_f32(0);
            for (; k + 3 < embed_dim; k += 4)
            {
                float32x4_t _val = vld1q_f32(ptr);
                float32x4_t _k = vld1q_f32(kptr);
                _sum = vmlaq_f32(_sum, _val, _k);
                ptr += 4;
                kptr += 4;
            }
            sum += sum_float32x4(_sum);
            for (; k < embed_dim; k++)
            {
                sum += *ptr * *kptr;
                ptr++;
                kptr++;
            }

            *outptr = sum;
            outptr++;
            bptr++;
        }
    }

    return 0;

#endif // __ARM_NEON

    // fallback to naive implement
    return MultiHeadAttention::forward(bottom_blobs, top_blobs, opt);
}

#if NCNN_BF16
int MultiHeadAttention_arm::create_pipeline_bf16s(const Option& opt)
{
    ncnn::cast_float32_to_bfloat16(q_weight_data, q_weight_data_bf16, opt);
    ncnn::cast_float32_to_bfloat16(q_bias_data, q_bias_data_bf16, opt);
    ncnn::cast_float32_to_bfloat16(k_weight_data, k_weight_data_bf16, opt);
    ncnn::cast_float32_to_bfloat16(k_bias_data, k_bias_data_bf16, opt);
    ncnn::cast_float32_to_bfloat16(v_weight_data, v_weight_data_bf16, opt);
    ncnn::cast_float32_to_bfloat16(v_bias_data, v_bias_data_bf16, opt);
    ncnn::cast_float32_to_bfloat16(out_weight_data, out_weight_data_bf16, opt);
    ncnn::cast_float32_to_bfloat16(out_bias_data, out_bias_data_bf16, opt);

    if (opt.lightmode)
    {
        q_weight_data.release();
        q_bias_data.release();
        k_weight_data.release();
        k_bias_data.release();
        v_weight_data.release();
        v_bias_data.release();
        out_weight_data.release();
        out_bias_data.release();
    }

    return 0;
}

int MultiHeadAttention_arm::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& q_blob = bottom_blobs[0];

    size_t elemsize = q_blob.elemsize;
    int elempack = q_blob.elempack;
    int elembits = q_blob.elembits();

    const Mat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const Mat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[2];

    int seqlen = q_blob.h;
    const int embed_dim_per_head = embed_dim / num_head;
    const float inv_sqrt_embed_dim_per_head = 1.f / sqrt(embed_dim_per_head);

#if __ARM_NEON
    if (elempack == 1)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create(embed_dim, seqlen, 2u, opt.blob_allocator);
        if (top_blob.empty())
            return -1;

        Mat xq(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
        Mat xk(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
        Mat xv(seqlen, embed_dim_per_head, num_head, 4u, opt.workspace_allocator);

        Mat xqk(seqlen, seqlen, num_head, 4u, opt.workspace_allocator);

        Mat xqkv(embed_dim_per_head, num_head, seqlen, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_head; q++)
        {
            // xq = affine(q) * inv_sqrt_embed_dim_per_head
            {
                Mat outm = xq.channel(q);

                const unsigned short* bptr0 = (const unsigned short*)q_bias_data_bf16 + q * embed_dim_per_head;
                const unsigned short* kptr0 = (const unsigned short*)q_weight_data_bf16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = outm.row(i);

                    const unsigned short* bptr = bptr0;
                    const unsigned short* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const unsigned short* ptr = q_blob.row<unsigned short>(i);

                        float sum = bfloat16_to_float32(*bptr);
                        int k = 0;
                        float32x4_t _sum = vdupq_n_f32(0);
                        for (; k + 3 < embed_dim; k += 4)
                        {
                            float32x4_t _val = float2bfloat(vld1_u16(ptr));
                            float32x4_t _k = float2bfloat(vld1_u16(kptr));
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 4;
                        }
                        sum += sum_float32x4(_sum);
                        for (; k < embed_dim; k++)
                        {
                            sum += bfloat16_to_float32(*ptr) * bfloat16_to_float32(*kptr);
                            ptr++;
                            kptr++;
                        }

                        *outptr = sum * inv_sqrt_embed_dim_per_head;
                        outptr++;
                        bptr++;
                    }
                }
            }

            // xk = affine(k)
            {
                Mat outm = xk.channel(q);

                const unsigned short* bptr0 = (const unsigned short*)k_bias_data_bf16 + q * embed_dim_per_head;
                const unsigned short* kptr0 = (const unsigned short*)k_weight_data_bf16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = outm.row(i);

                    const unsigned short* bptr = bptr0;
                    const unsigned short* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const unsigned short* ptr = k_blob.row<unsigned short>(i);

                        float sum = bfloat16_to_float32(*bptr);
                        int k = 0;
                        float32x4_t _sum = vdupq_n_f32(0);
                        for (; k + 3 < embed_dim; k += 4)
                        {
                            float32x4_t _val = float2bfloat(vld1_u16(ptr));
                            float32x4_t _k = float2bfloat(vld1_u16(kptr));
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 4;
                        }
                        sum += sum_float32x4(_sum);
                        for (; k < embed_dim; k++)
                        {
                            sum += bfloat16_to_float32(*ptr) * bfloat16_to_float32(*kptr);
                            ptr++;
                            kptr++;
                        }

                        *outptr = sum;
                        outptr++;
                        bptr++;
                    }
                }
            }

            // xv = affine(v)
            {
                Mat outm = xv.channel(q);

                const unsigned short* bptr = (const unsigned short*)v_bias_data_bf16 + q * embed_dim_per_head;
                const unsigned short* kptr0 = (const unsigned short*)v_weight_data_bf16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < embed_dim_per_head; i++)
                {
                    float* outptr = outm.row(i);

                    for (int j = 0; j < seqlen; j++)
                    {
                        const unsigned short* ptr = v_blob.row<unsigned short>(j);
                        const unsigned short* kptr = kptr0;

                        float sum = bfloat16_to_float32(*bptr);
                        int k = 0;
                        float32x4_t _sum = vdupq_n_f32(0);
                        for (; k + 3 < embed_dim; k += 4)
                        {
                            float32x4_t _val = float2bfloat(vld1_u16(ptr));
                            float32x4_t _k = float2bfloat(vld1_u16(kptr));
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 4;
                        }
                        sum += sum_float32x4(_sum);
                        for (; k < embed_dim; k++)
                        {
                            sum += bfloat16_to_float32(*ptr) * bfloat16_to_float32(*kptr);
                            ptr++;
                            kptr++;
                        }

                        *outptr = sum;
                        outptr++;
                    }

                    bptr++;
                    kptr0 += embed_dim;
                }
            }

            // xqk = xq * xk
            // xq  (embed_dim_per_head, seqlen)
            // xk  (embed_dim_per_head, seqlen)
            {
                const Mat xqm = xq.channel(q);
                const Mat xkm = xk.channel(q);

                Mat outm = xqk.channel(q);

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = outm.row(i);

                    for (int j = 0; j < seqlen; j++)
                    {
                        const float* qptr = xqm.row(i);
                        const float* kptr = xkm.row(j);

                        float sum = 0.f;
                        int k = 0;
                        float32x4_t _sum = vdupq_n_f32(0);
                        for (; k + 3 < embed_dim_per_head; k += 4)
                        {
                            float32x4_t _val = vld1q_f32(qptr);
                            float32x4_t _k = vld1q_f32(kptr);
                            _sum = vmlaq_f32(_sum, _val, _k);
                            qptr += 4;
                            kptr += 4;
                        }
                        sum += sum_float32x4(_sum);
                        for (; k < embed_dim_per_head; k++)
                        {
                            sum += *qptr * *kptr;
                            qptr++;
                            kptr++;
                        }

                        *outptr = sum;
                        outptr++;
                    }
                }
            }

            // softmax(xqk)
            {
                Mat outm = xqk.channel(q);

                for (int i = 0; i < seqlen; i++)
                {
                    float* ptr = outm.row(i);

                    float* ptr0 = ptr;
                    float max = -FLT_MAX;
                    int j = 0;
                    float32x4_t _max = vdupq_n_f32(-FLT_MAX);
                    for (; j + 3 < seqlen; j += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        _max = vmaxq_f32(_max, _p);
                        ptr0 += 4;
                    }
                    max = max_float32x4(max, _max);
                    for (; j < seqlen; j++)
                    {
                        max = std::max(max, (float)(*ptr0));
                        ptr0++;
                    }

                    ptr0 = ptr;
                    float sum = 0.f;
                    j = 0;
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    _max = vdupq_n_f32(max);
                    for (; j + 3 < seqlen; j += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        _p = exp_ps(vsubq_f32(_p, _max));
                        vst1q_f32(ptr0, _p);
                        _sum = vaddq_f32(_sum, _p);
                        ptr0 += 4;
                    }
                    sum += sum_float32x4(_sum);
                    for (; j < seqlen; j++)
                    {
                        *ptr0 = (float)(exp(*ptr0 - max));
                        sum += *ptr0;
                        ptr0++;
                    }

                    j = 0;
                    _sum = vdupq_n_f32(sum);
                    for (; j + 3 < seqlen; j += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = div_ps(_p, _sum);
                        vst1q_f32(ptr, _p);
                        ptr += 4;
                    }
                    for (; j < seqlen; j++)
                    {
                        *ptr /= sum;
                        ptr++;
                    }
                }
            }

            // xqkv = xqk * xv
            // xqk (seqlen, seqlen)
            // xv  (seqlen, embed_dim_per_head)
            // out (embed_dim_per_head, num_head, seqlen)
            {
                const Mat xqkm = xqk.channel(q);
                const Mat xvm = xv.channel(q);

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = xqkv.channel(i).row(q);

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const float* qkptr = xqkm.row(i);
                        const float* vptr = xvm.row(j);

                        float sum = 0.f;
                        int k = 0;
                        float32x4_t _sum = vdupq_n_f32(0);
                        for (; k + 3 < seqlen; k += 4)
                        {
                            float32x4_t _val = vld1q_f32(qkptr);
                            float32x4_t _k = vld1q_f32(vptr);
                            _sum = vmlaq_f32(_sum, _val, _k);
                            qkptr += 4;
                            vptr += 4;
                        }
                        sum += sum_float32x4(_sum);
                        for (; k < seqlen; k++)
                        {
                            sum += *qkptr * *vptr;
                            qkptr++;
                            vptr++;
                        }

                        *outptr = sum;
                        outptr++;
                    }
                }
            }
        }

        // out = affine(xqkv)
        // xqkv  (embed_dim, seqlen)
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < seqlen; i++)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(i);

            const unsigned short* bptr = (const unsigned short*)out_bias_data_bf16;
            const unsigned short* kptr = (const unsigned short*)out_weight_data_bf16;

            for (int j = 0; j < embed_dim; j++)
            {
                const float* ptr = xqkv.channel(i);

                float sum = bfloat16_to_float32(*bptr);
                int k = 0;
                float32x4_t _sum = vdupq_n_f32(0);
                for (; k + 3 < embed_dim; k += 4)
                {
                    float32x4_t _val = vld1q_f32(ptr);
                    float32x4_t _k = float2bfloat(vld1_u16(kptr));
                    _sum = vmlaq_f32(_sum, _val, _k);
                    ptr += 4;
                    kptr += 4;
                }
                sum += sum_float32x4(_sum);
                for (; k < embed_dim; k++)
                {
                    sum += *ptr * bfloat16_to_float32(*kptr);
                    ptr++;
                    kptr++;
                }

                *outptr = float32_to_bfloat16(sum);
                outptr++;
                bptr++;
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create(embed_dim, seqlen, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -1;

        Mat xq(embed_dim_per_head, seqlen, num_head, 16u, 4, opt.workspace_allocator);
        Mat xk(embed_dim_per_head, seqlen, num_head, 16u, 4, opt.workspace_allocator);
        Mat xv(seqlen, embed_dim_per_head, num_head, 16u, 4, opt.workspace_allocator);

        Mat xqk(seqlen * elempack, seqlen, num_head, 16u, 4, opt.workspace_allocator);

        Mat xqkv(embed_dim_per_head, num_head, seqlen, 16u, 4, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_head; q++)
        {
            // xq = affine(q) * inv_sqrt_embed_dim_per_head
            {
                float* outptr = (float*)xq.channel(q);

                const unsigned short* bptr0 = (const unsigned short*)q_bias_data_bf16 + q * embed_dim_per_head;
                const unsigned short* kptr0 = (const unsigned short*)q_weight_data_bf16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const unsigned short* bptr = bptr0;
                    const unsigned short* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const unsigned short* ptr = q_blob.row<unsigned short>(i);

                        float32x4_t _sum = vdupq_n_f32(bfloat16_to_float32(*bptr));
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = float2bfloat(vld1_u16(ptr));
                            float32x4_t _k = vdupq_n_f32(bfloat16_to_float32(*kptr));
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        float32x4_t _slope = vdupq_n_f32(inv_sqrt_embed_dim_per_head);
                        _sum = vmulq_f32(_sum, _slope);

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                        bptr++;
                    }
                }
            }

            // xk = affine(k)
            {
                float* outptr = (float*)xk.channel(q);

                const unsigned short* bptr0 = (const unsigned short*)k_bias_data_bf16 + q * embed_dim_per_head;
                const unsigned short* kptr0 = (const unsigned short*)k_weight_data_bf16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const unsigned short* bptr = bptr0;
                    const unsigned short* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const unsigned short* ptr = k_blob.row<unsigned short>(i);

                        float32x4_t _sum = vdupq_n_f32(bfloat16_to_float32(*bptr));
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = float2bfloat(vld1_u16(ptr));
                            float32x4_t _k = vdupq_n_f32(bfloat16_to_float32(*kptr));
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                        bptr++;
                    }
                }
            }

            // xv = affine(v)
            {
                float* outptr = (float*)xv.channel(q);

                const unsigned short* bptr = (const unsigned short*)v_bias_data_bf16 + q * embed_dim_per_head;
                const unsigned short* kptr0 = (const unsigned short*)v_weight_data_bf16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < embed_dim_per_head; i++)
                {
                    for (int j = 0; j < seqlen; j++)
                    {
                        const unsigned short* ptr = v_blob.row<unsigned short>(j);
                        const unsigned short* kptr = kptr0;

                        float32x4_t _sum = vdupq_n_f32(bfloat16_to_float32(*bptr));
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = float2bfloat(vld1_u16(ptr));
                            float32x4_t _k = vdupq_n_f32(bfloat16_to_float32(*kptr));
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                    }

                    bptr++;
                    kptr0 += embed_dim;
                }
            }

            // xqk = xq * xk
            // xq  (embed_dim_per_head, seqlen)
            // xk  (embed_dim_per_head, seqlen)
            {
                const Mat xqm = xq.channel(q);
                const Mat xkm = xk.channel(q);

                Mat outm = xqk.channel(q);

                Mat upxkm;
                convert_packing(xkm, upxkm, 1);

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = outm.row(i);

                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        const float* qptr = xqm.row(i);
                        const float* kptr = upxkm.row(j);

                        float32x4_t _sum = vdupq_n_f32(0.f);
                        for (int k = 0; k < embed_dim_per_head; k++)
                        {
                            float32x4_t _q = vld1q_f32(qptr);
                            float32x4_t _k = vdupq_n_f32(*kptr);
                            _sum = vmlaq_f32(_sum, _q, _k);
                            qptr += 4;
                            kptr += 1;
                        }

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                    }
                }
            }

            // softmax(xqk)
            {
                Mat outm = xqk.channel(q);
                for (int i = 0; i < seqlen; i++)
                {
                    float* ptr = outm.row(i);

                    float* ptr0 = ptr;
                    float32x4_t _max = vdupq_n_f32(-FLT_MAX);
                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        _max = vmaxq_f32(_max, _p);
                        ptr0 += 4;
                    }

                    ptr0 = ptr;
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        _p = exp_ps(vsubq_f32(_p, _max));
                        vst1q_f32(ptr0, _p);
                        _sum = vaddq_f32(_sum, _p);
                        ptr0 += 4;
                    }

                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = div_ps(_p, _sum);
                        vst1q_f32(ptr, _p);
                        ptr += 4;
                    }
                }
            }

            // xqkv = xqk * xv
            // xqk (seqlen, seqlen)
            // xv  (seqlen, embed_dim_per_head)
            // out (embed_dim_per_head, num_head, seqlen)
            {
                const Mat xqkm = xqk.channel(q);
                const Mat xvm = xv.channel(q);

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = xqkv.channel(i).row(q);

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const float* qkptr = xqkm.row(i);
                        const float* vptr = xvm.row(j);

                        float32x4_t _sum = vdupq_n_f32(0.f);
                        for (int k = 0; k < seqlen * elempack; k++)
                        {
                            float32x4_t _qk = vld1q_f32(qkptr);
                            float32x4_t _v = vdupq_n_f32(*vptr);
                            _sum = vmlaq_f32(_sum, _qk, _v);
                            qkptr += 4;
                            vptr += 1;
                        }

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                    }
                }
            }
        }

        // out = affine(xqkv)
        // xqkv  (embed_dim, seqlen)
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < seqlen; i++)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(i);

            const unsigned short* bptr = (const unsigned short*)out_bias_data_bf16;
            const unsigned short* kptr = (const unsigned short*)out_weight_data_bf16;

            for (int j = 0; j < embed_dim; j++)
            {
                const float* ptr = xqkv.channel(i);

                float32x4_t _sum = vdupq_n_f32(bfloat16_to_float32(*bptr));
                for (int k = 0; k < embed_dim; k++)
                {
                    float32x4_t _val = vld1q_f32(ptr);
                    float32x4_t _k = vdupq_n_f32(bfloat16_to_float32(*kptr));
                    _sum = vmlaq_f32(_sum, _val, _k);
                    ptr += 4;
                    kptr += 1;
                }

                vst1_u16(outptr, bfloat2float(_sum));
                outptr += 4;
                bptr++;
            }
        }

        return 0;
    }

#endif

    Mat& top_blob = top_blobs[0];
    top_blob.create(embed_dim, seqlen, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -1;

    Mat xq(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
    Mat xk(embed_dim_per_head, seqlen, num_head, 4u, opt.workspace_allocator);
    Mat xv(seqlen, embed_dim_per_head, num_head, 4u, opt.workspace_allocator);

    Mat xqk(seqlen, seqlen, num_head, 4u, opt.workspace_allocator);

    Mat xqkv(embed_dim_per_head, num_head, seqlen, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_head; q++)
    {
        // xq = affine(q) * inv_sqrt_embed_dim_per_head
        {
            Mat outm = xq.channel(q);

            const unsigned short* bptr0 = (const unsigned short*)q_bias_data_bf16 + q * embed_dim_per_head;
            const unsigned short* kptr0 = (const unsigned short*)q_weight_data_bf16 + embed_dim * q * embed_dim_per_head;

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                const unsigned short* bptr = bptr0;
                const unsigned short* kptr = kptr0;

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const unsigned short* ptr = q_blob.row<unsigned short>(i);

                    float sum = bfloat16_to_float32(*bptr);
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += bfloat16_to_float32(*ptr) * bfloat16_to_float32(*kptr);
                        ptr++;
                        kptr++;
                    }

                    *outptr = sum * inv_sqrt_embed_dim_per_head;
                    outptr++;
                    bptr++;
                }
            }
        }

        // xk = affine(k)
        {
            Mat outm = xk.channel(q);

            const unsigned short* bptr0 = (const unsigned short*)k_bias_data_bf16 + q * embed_dim_per_head;
            const unsigned short* kptr0 = (const unsigned short*)k_weight_data_bf16 + embed_dim * q * embed_dim_per_head;

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                const unsigned short* bptr = bptr0;
                const unsigned short* kptr = kptr0;

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const unsigned short* ptr = k_blob.row<unsigned short>(i);

                    float sum = bfloat16_to_float32(*bptr);
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += bfloat16_to_float32(*ptr) * bfloat16_to_float32(*kptr);
                        ptr++;
                        kptr++;
                    }

                    *outptr = sum;
                    outptr++;
                    bptr++;
                }
            }
        }

        // xv = affine(v)
        {
            Mat outm = xv.channel(q);

            const unsigned short* bptr = (const unsigned short*)v_bias_data_bf16 + q * embed_dim_per_head;
            const unsigned short* kptr0 = (const unsigned short*)v_weight_data_bf16 + embed_dim * q * embed_dim_per_head;

            for (int i = 0; i < embed_dim_per_head; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < seqlen; j++)
                {
                    const unsigned short* ptr = v_blob.row<unsigned short>(j);
                    const unsigned short* kptr = kptr0;

                    float sum = bfloat16_to_float32(*bptr);
                    for (int k = 0; k < embed_dim; k++)
                    {
                        sum += bfloat16_to_float32(*ptr) * bfloat16_to_float32(*kptr);
                        ptr++;
                        kptr++;
                    }

                    *outptr = sum;
                    outptr++;
                }

                bptr++;
                kptr0 += embed_dim;
            }
        }

        // xqk = xq * xk
        // xq  (embed_dim_per_head, seqlen)
        // xk  (embed_dim_per_head, seqlen)
        {
            const Mat xqm = xq.channel(q);
            const Mat xkm = xk.channel(q);

            Mat outm = xqk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = outm.row(i);

                for (int j = 0; j < seqlen; j++)
                {
                    const float* qptr = xqm.row(i);
                    const float* kptr = xkm.row(j);

                    float sum = 0.f;
                    for (int k = 0; k < embed_dim_per_head; k++)
                    {
                        sum += *qptr * *kptr;
                        qptr++;
                        kptr++;
                    }

                    *outptr = sum;
                    outptr++;
                }
            }
        }

        // softmax(xqk)
        {
            Mat outm = xqk.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* ptr = outm.row(i);

                float* ptr0 = ptr;
                float max = -FLT_MAX;
                for (int j = 0; j < seqlen; j++)
                {
                    max = std::max(max, (float)(*ptr0));
                    ptr0++;
                }

                ptr0 = ptr;
                float sum = 0.f;
                for (int j = 0; j < seqlen; j++)
                {
                    *ptr0 = (float)(exp(*ptr0 - max));
                    sum += *ptr0;
                    ptr0++;
                }

                for (int j = 0; j < seqlen; j++)
                {
                    *ptr /= sum;
                    ptr++;
                }
            }
        }

        // xqkv = xqk * xv
        // xqk (seqlen, seqlen)
        // xv  (seqlen, embed_dim_per_head)
        // out (embed_dim_per_head, num_head, seqlen)
        {
            const Mat xqkm = xqk.channel(q);
            const Mat xvm = xv.channel(q);

            for (int i = 0; i < seqlen; i++)
            {
                float* outptr = xqkv.channel(i).row(q);

                for (int j = 0; j < embed_dim_per_head; j++)
                {
                    const float* qkptr = xqkm.row(i);
                    const float* vptr = xvm.row(j);

                    float sum = 0.f;
                    for (int k = 0; k < seqlen; k++)
                    {
                        sum += *qkptr * *vptr;
                        qkptr++;
                        vptr++;
                    }

                    *outptr = sum;
                    outptr++;
                }
            }
        }
    }

    // out = affine(xqkv)
    // xqkv  (embed_dim, seqlen)
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < seqlen; i++)
    {
        unsigned short* outptr = top_blob.row<unsigned short>(i);

        const unsigned short* bptr = (const unsigned short*)out_bias_data_bf16;
        const unsigned short* kptr = (const unsigned short*)out_weight_data_bf16;

        for (int j = 0; j < embed_dim; j++)
        {
            const float* ptr = xqkv.channel(i);

            float sum = bfloat16_to_float32(*bptr);
            for (int k = 0; k < embed_dim; k++)
            {
                sum += *ptr * bfloat16_to_float32(*kptr);
                ptr++;
                kptr++;
            }

            *outptr = float32_to_bfloat16(sum);
            outptr++;
            bptr++;
        }
    }

    return 0;
}

#endif

} // namespace ncnn
