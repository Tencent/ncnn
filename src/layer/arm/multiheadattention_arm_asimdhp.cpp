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
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

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

inline __fp16 sum_float16x4(float16x4_t _sum)
{
    float16x4_t _ss2 = vpadd_f16(_sum, _sum);
    _ss2 = vpadd_f16(_ss2, _ss2);
    __fp16 sum = vget_lane_f16(_ss2, 0);
    return sum;
}

inline __fp16 max_float16x4(__fp16 max, float16x4_t _max)
{
#if __aarch64__
    max = std::max(max, vmaxv_f16(_max));
#else
    float16x4_t _mm2 = vpmax_f16(_max, _max);
    _mm2 = vpmax_f16(_mm2, _mm2);
    max = std::max(max, vget_lane_f16(_mm2, 0));
#endif
    return max;
}

static inline float16x4_t div_ps_float16x4(float16x4_t a, float16x4_t b)
{
#if __aarch64__
    return vdiv_f16(a, b);
#else
    float16x4_t reciprocal = vrecpe_f16(b);
    reciprocal = vmul_f16(vrecps_f16(b, reciprocal), reciprocal);
    return vmul_f16(a, reciprocal);
#endif
}

static inline float16x8_t div_ps_float16x8(float16x8_t a, float16x8_t b)
{
#if __aarch64__
    return vdivq_f16(a, b);
#else
    float16x4_t reciprocal = vrecpeq_f16(b);
    reciprocal = vmulq_f16(vrecpsq_f16(b, reciprocal), reciprocal);
    return vmulq_f16(a, reciprocal);
#endif
}

int MultiHeadAttention_arm::create_pipeline_fp16s(const Option& opt)
{
    ncnn::cast_float32_to_float16(q_weight_data, q_weight_data_fp16, opt);
    ncnn::cast_float32_to_float16(q_bias_data, q_bias_data_fp16, opt);
    ncnn::cast_float32_to_float16(k_weight_data, k_weight_data_fp16, opt);
    ncnn::cast_float32_to_float16(k_bias_data, k_bias_data_fp16, opt);
    ncnn::cast_float32_to_float16(v_weight_data, v_weight_data_fp16, opt);
    ncnn::cast_float32_to_float16(v_bias_data, v_bias_data_fp16, opt);
    ncnn::cast_float32_to_float16(out_weight_data, out_weight_data_fp16, opt);
    ncnn::cast_float32_to_float16(out_bias_data, out_bias_data_fp16, opt);

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

int MultiHeadAttention_arm::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

                const __fp16* bptr0 = (const __fp16*)q_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)q_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = outm.row(i);

                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = q_blob.row<__fp16>(i);

                        float sum = *bptr;
                        int k = 0;
                        float32x4_t _sum = vdupq_n_f32(0);
                        for (; k + 3 < embed_dim; k += 4)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(ptr));
                            float32x4_t _k = vcvt_f32_f16(vld1_f16(kptr));
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
                Mat outm = xk.channel(q);

                const __fp16* bptr0 = (const __fp16*)k_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)k_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = outm.row(i);

                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = k_blob.row<__fp16>(i);

                        float sum = *bptr;
                        int k = 0;
                        float32x4_t _sum = vdupq_n_f32(0);
                        for (; k + 3 < embed_dim; k += 4)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(ptr));
                            float32x4_t _k = vcvt_f32_f16(vld1_f16(kptr));
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
                Mat outm = xv.channel(q);

                const __fp16* bptr = (const __fp16*)v_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)v_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < embed_dim_per_head; i++)
                {
                    float* outptr = outm.row(i);

                    for (int j = 0; j < seqlen; j++)
                    {
                        const __fp16* ptr = v_blob.row<__fp16>(j);
                        const __fp16* kptr = kptr0;

                        float sum = *bptr;
                        int k = 0;
                        float32x4_t _sum = vdupq_n_f32(0);
                        for (; k + 3 < embed_dim; k += 4)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(ptr));
                            float32x4_t _k = vcvt_f32_f16(vld1_f16(kptr));
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
            __fp16* outptr = top_blob.row<__fp16>(i);

            const __fp16* bptr = (const __fp16*)out_bias_data_fp16;
            const __fp16* kptr = (const __fp16*)out_weight_data_fp16;

            for (int j = 0; j < embed_dim; j++)
            {
                const float* ptr = xqkv.channel(i);

                float sum = *bptr;
                int k = 0;
                float32x4_t _sum = vdupq_n_f32(0);
                for (; k + 3 < embed_dim; k += 4)
                {
                    float32x4_t _val = vld1q_f32(ptr);
                    float32x4_t _k = vcvt_f32_f16(vld1_f16(kptr));
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

                const __fp16* bptr0 = (const __fp16*)q_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)q_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = q_blob.row<__fp16>(i);

                        float32x4_t _sum = vdupq_n_f32(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(ptr));
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

                const __fp16* bptr0 = (const __fp16*)k_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)k_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = k_blob.row<__fp16>(i);

                        float32x4_t _sum = vdupq_n_f32(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(ptr));
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

                const __fp16* bptr = (const __fp16*)v_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)v_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < embed_dim_per_head; i++)
                {
                    for (int j = 0; j < seqlen; j++)
                    {
                        const __fp16* ptr = v_blob.row<__fp16>(j);
                        const __fp16* kptr = kptr0;

                        float32x4_t _sum = vdupq_n_f32(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(ptr));
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
            __fp16* outptr = top_blob.row<__fp16>(i);

            const __fp16* bptr = (const __fp16*)out_bias_data_fp16;
            const __fp16* kptr = (const __fp16*)out_weight_data_fp16;

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

                vst1_f16(outptr, vcvt_f16_f32(_sum));
                outptr += 4;
                bptr++;
            }
        }

        return 0;
    }

    if (elempack == 8)
    {
        seqlen *= 2;
        elempack = 4;

        Mat q_blob_p4, k_blob_p4, v_blob_p4;
        convert_packing(q_blob, q_blob_p4, 4);
        convert_packing(k_blob, k_blob_p4, 4);
        convert_packing(v_blob, v_blob_p4, 4);

        Mat top_blob_p4;
        top_blob_p4.create(embed_dim, seqlen, 8u, 4, opt.blob_allocator);
        if (top_blob_p4.empty())
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

                const __fp16* bptr0 = (const __fp16*)q_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)q_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = q_blob_p4.row<__fp16>(i);

                        float32x4_t _sum = vdupq_n_f32(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(ptr));
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

                const __fp16* bptr0 = (const __fp16*)k_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)k_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = k_blob_p4.row<__fp16>(i);

                        float32x4_t _sum = vdupq_n_f32(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(ptr));
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

                const __fp16* bptr = (const __fp16*)v_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)v_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < embed_dim_per_head; i++)
                {
                    for (int j = 0; j < seqlen; j++)
                    {
                        const __fp16* ptr = v_blob_p4.row<__fp16>(j);
                        const __fp16* kptr = kptr0;

                        float32x4_t _sum = vdupq_n_f32(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(ptr));
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
            __fp16* outptr = top_blob_p4.row<__fp16>(i);

            const __fp16* bptr = (const __fp16*)out_bias_data_fp16;
            const __fp16* kptr = (const __fp16*)out_weight_data_fp16;

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

                vst1_f16(outptr, vcvt_f16_f32(_sum));
                outptr += 4;
                bptr++;
            }
        }

        Mat& top_blob = top_blobs[0];
        convert_packing(top_blob_p4, top_blob, 8);

        return 0;
    }

    return 0;
}

int MultiHeadAttention_arm::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

    if (elempack == 1)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create(embed_dim, seqlen, 2u, opt.blob_allocator);
        if (top_blob.empty())
            return -1;

        Mat xq(embed_dim_per_head, seqlen, num_head, 2u, opt.workspace_allocator);
        Mat xk(embed_dim_per_head, seqlen, num_head, 2u, opt.workspace_allocator);
        Mat xv(seqlen, embed_dim_per_head, num_head, 2u, opt.workspace_allocator);

        Mat xqk(seqlen, seqlen, num_head, 2u, opt.workspace_allocator);

        Mat xqkv(embed_dim_per_head, num_head, seqlen, 2u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_head; q++)
        {
            // xq = affine(q) * inv_sqrt_embed_dim_per_head
            {
                Mat outm = xq.channel(q);

                const __fp16* bptr0 = (const __fp16*)q_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)q_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    __fp16* outptr = outm.row<__fp16>(i);

                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = q_blob.row<__fp16>(i);

                        __fp16 sum = *bptr;
                        int k = 0;
                        float16x4_t _sum = vdup_n_f16(0.f);
                        for (; k + 3 < embed_dim; k += 4)
                        {
                            float16x4_t _val = vld1_f16(ptr);
                            float16x4_t _k = vld1_f16(kptr);
                            _sum = vfma_f16(_sum, _val, _k);
                            ptr += 4;
                            kptr += 4;
                        }
                        sum += sum_float16x4(_sum);
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
                Mat outm = xk.channel(q);

                const __fp16* bptr0 = (const __fp16*)k_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)k_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    __fp16* outptr = outm.row<__fp16>(i);

                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = k_blob.row<__fp16>(i);

                        __fp16 sum = *bptr;
                        int k = 0;
                        float16x4_t _sum = vdup_n_f16(0);
                        for (; k + 3 < embed_dim; k += 4)
                        {
                            float16x4_t _val = vld1_f16(ptr);
                            float16x4_t _k = vld1_f16(kptr);
                            _sum = vfma_f16(_sum, _val, _k);
                            ptr += 4;
                            kptr += 4;
                        }
                        sum += sum_float16x4(_sum);
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
                Mat outm = xv.channel(q);

                const __fp16* bptr = (const __fp16*)v_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)v_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < embed_dim_per_head; i++)
                {
                    __fp16* outptr = outm.row<__fp16>(i);

                    for (int j = 0; j < seqlen; j++)
                    {
                        const __fp16* ptr = v_blob.row<__fp16>(j);
                        const __fp16* kptr = kptr0;

                        __fp16 sum = *bptr;
                        int k = 0;
                        float16x4_t _sum = vdup_n_f16(0);
                        for (; k + 3 < embed_dim; k += 4)
                        {
                            float16x4_t _val = vld1_f16(ptr);
                            float16x4_t _k = vld1_f16(kptr);
                            _sum = vfma_f16(_sum, _val, _k);
                            ptr += 4;
                            kptr += 4;
                        }
                        sum += sum_float16x4(_sum);
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
                    __fp16* outptr = outm.row<__fp16>(i);

                    for (int j = 0; j < seqlen; j++)
                    {
                        const __fp16* qptr = xqm.row<__fp16>(i);
                        const __fp16* kptr = xkm.row<__fp16>(j);

                        __fp16 sum = 0.f;
                        int k = 0;
                        float16x4_t _sum = vdup_n_f16(0);
                        for (; k + 3 < embed_dim_per_head; k += 4)
                        {
                            float16x4_t _val = vld1_f16(qptr);
                            float16x4_t _k = vld1_f16(kptr);
                            _sum = vfma_f16(_sum, _val, _k);
                            qptr += 4;
                            kptr += 4;
                        }
                        sum += sum_float16x4(_sum);
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
                    __fp16* ptr = outm.row<__fp16>(i);

                    __fp16* ptr0 = ptr;
                    __fp16 max = -FLT_MAX;
                    int j = 0;
                    float16x4_t _max = vdup_n_f16(-FLT_MAX);
                    for (; j + 3 < seqlen; j += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr0);
                        _max = vmax_f16(_max, _p);
                        ptr0 += 4;
                    }
                    max = max_float16x4(max, _max);
                    for (; j < seqlen; j++)
                    {
                        max = std::max(max, *ptr0);
                        ptr0++;
                    }

                    ptr0 = ptr;
                    __fp16 sum = 0.f;
                    j = 0;
                    float16x4_t _sum = vdup_n_f16(0.f);
                    _max = vdup_n_f16(max);
                    for (; j + 3 < seqlen; j += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr0);
                        _p = exp_ps(vsub_f16(_p, _max));
                        vst1_f16(ptr0, _p);
                        _sum = vadd_f16(_sum, _p);
                        ptr0 += 4;
                    }
                    sum += sum_float16x4(_sum);
                    for (; j < seqlen; j++)
                    {
                        *ptr0 = exp(*ptr0 - max);
                        sum += *ptr0;
                        ptr0++;
                    }

                    j = 0;
                    _sum = vdup_n_f16(sum);
                    for (; j + 3 < seqlen; j += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        _p = div_ps_float16x4(_p, _sum);
                        vst1_f16(ptr, _p);
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
                    __fp16* outptr = xqkv.channel(i).row<__fp16>(q);

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* qkptr = xqkm.row<__fp16>(i);
                        const __fp16* vptr = xvm.row<__fp16>(j);

                        __fp16 sum = 0.f;
                        int k = 0;
                        float16x4_t _sum = vdup_n_f16(0);
                        for (; k + 3 < seqlen; k += 4)
                        {
                            float16x4_t _val = vld1_f16(qkptr);
                            float16x4_t _k = vld1_f16(vptr);
                            _sum = vfma_f16(_sum, _val, _k);
                            qkptr += 4;
                            vptr += 4;
                        }
                        sum += sum_float16x4(_sum);
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
            __fp16* outptr = top_blob.row<__fp16>(i);

            const __fp16* bptr = (const __fp16*)out_bias_data_fp16;
            const __fp16* kptr = (const __fp16*)out_weight_data_fp16;

            for (int j = 0; j < embed_dim; j++)
            {
                const __fp16* ptr = (__fp16*)xqkv.channel(i);

                __fp16 sum = *bptr;
                int k = 0;
                float16x4_t _sum = vdup_n_f16(0);
                for (; k + 3 < embed_dim; k += 4)
                {
                    float16x4_t _val = vld1_f16(ptr);
                    float16x4_t _k = vld1_f16(kptr);
                    _sum = vfma_f16(_sum, _val, _k);
                    ptr += 4;
                    kptr += 4;
                }
                sum += sum_float16x4(_sum);
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
    }

    if (elempack == 4)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create(embed_dim, seqlen, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -1;

        Mat xq(embed_dim_per_head, seqlen, num_head, 8u, 4, opt.workspace_allocator);
        Mat xk(embed_dim_per_head, seqlen, num_head, 8u, 4, opt.workspace_allocator);
        Mat xv(seqlen, embed_dim_per_head, num_head, 8u, 4, opt.workspace_allocator);

        Mat xqk(4 * seqlen, seqlen, num_head, 8u, 4, opt.workspace_allocator);

        Mat xqkv(embed_dim_per_head, num_head, seqlen, 8u, 4, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_head; q++)
        {
            // xq = affine(q) * inv_sqrt_embed_dim_per_head
            {
                __fp16* outptr = (__fp16*)xq.channel(q);

                const __fp16* bptr0 = (const __fp16*)q_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)q_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = q_blob.row<__fp16>(i);

                        float16x4_t _sum = vdup_n_f16(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float16x4_t _val = vld1_f16(ptr);
                            float16x4_t _k = vdup_n_f16(*kptr);
                            _sum = vfma_f16(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        float16x4_t _slope = vdup_n_f16(inv_sqrt_embed_dim_per_head);
                        _sum = vmul_f16(_sum, _slope);

                        vst1_f16(outptr, _sum);
                        outptr += 4;
                        bptr++;
                    }
                }
            }

            // xk = affine(k)
            {
                __fp16* outptr = (__fp16*)xk.channel(q);

                const __fp16* bptr0 = (const __fp16*)k_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)k_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = k_blob.row<__fp16>(i);

                        float16x4_t _sum = vdup_n_f16(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float16x4_t _val = vld1_f16(ptr);
                            float16x4_t _k = vdup_n_f16(*kptr);
                            _sum = vfma_f16(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        vst1_f16(outptr, _sum);
                        outptr += 4;
                        bptr++;
                    }
                }
            }

            // xv = affine(v)
            {
                __fp16* outptr = (__fp16*)xv.channel(q);

                const __fp16* bptr = (const __fp16*)v_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)v_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < embed_dim_per_head; i++)
                {
                    for (int j = 0; j < seqlen; j++)
                    {
                        const __fp16* ptr = v_blob.row<__fp16>(j);
                        const __fp16* kptr = kptr0;

                        float16x4_t _sum = vdup_n_f16(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float16x4_t _val = vld1_f16(ptr);
                            float16x4_t _k = vdup_n_f16(*kptr);
                            _sum = vfma_f16(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        vst1_f16(outptr, _sum);
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
                    __fp16* outptr = outm.row<__fp16>(i);

                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        const __fp16* qptr = xqm.row<__fp16>(i);
                        const __fp16* kptr = upxkm.row<__fp16>(j);

                        float16x4_t _sum = vdup_n_f16(0.f);
                        for (int k = 0; k < embed_dim_per_head; k++)
                        {
                            float16x4_t _q = vld1_f16(qptr);
                            float16x4_t _k = vdup_n_f16(*kptr);
                            _sum = vfma_f16(_sum, _q, _k);
                            qptr += 4;
                            kptr += 1;
                        }

                        vst1_f16(outptr, _sum);
                        outptr += 4;
                    }
                }
            }

            // softmax(xqk)
            {
                Mat outm = xqk.channel(q);
                for (int i = 0; i < seqlen; i++)
                {
                    __fp16* ptr = outm.row<__fp16>(i);

                    __fp16* ptr0 = ptr;
                    float16x4_t _max = vdup_n_f16(-FLT_MAX);
                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float16x4_t _p = vld1_f16(ptr0);
                        _max = vmax_f16(_max, _p);
                        ptr0 += 4;
                    }

                    ptr0 = ptr;
                    float16x4_t _sum = vdup_n_f16(0.f);
                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float16x4_t _p = vld1_f16(ptr0);
                        _p = exp_ps(vsub_f16(_p, _max));
                        vst1_f16(ptr0, _p);
                        _sum = vadd_f16(_sum, _p);
                        ptr0 += 4;
                    }

                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        _p = div_ps_float16x4(_p, _sum);
                        vst1_f16(ptr, _p);
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
                    __fp16* outptr = xqkv.channel(i).row<__fp16>(q);

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* qkptr = xqkm.row<__fp16>(i);
                        const __fp16* vptr = xvm.row<__fp16>(j);

                        float16x4_t _sum = vdup_n_f16(0.f);
                        for (int k = 0; k < seqlen * elempack; k++)
                        {
                            float16x4_t _qk = vld1_f16(qkptr);
                            float16x4_t _v = vdup_n_f16(*vptr);
                            _sum = vfma_f16(_sum, _qk, _v);
                            qkptr += 4;
                            vptr += 1;
                        }

                        vst1_f16(outptr, _sum);
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
            __fp16* outptr = top_blob.row<__fp16>(i);

            const __fp16* bptr = (const __fp16*)out_bias_data_fp16;
            const __fp16* kptr = (const __fp16*)out_weight_data_fp16;

            for (int j = 0; j < embed_dim; j++)
            {
                const __fp16* ptr = (__fp16*)xqkv.channel(i);

                float16x4_t _sum = vdup_n_f16(*bptr);
                for (int k = 0; k < embed_dim; k++)
                {
                    float16x4_t _val = vld1_f16(ptr);
                    float16x4_t _k = vdup_n_f16(*kptr);
                    _sum = vfma_f16(_sum, _val, _k);
                    ptr += 4;
                    kptr += 1;
                }

                vst1_f16(outptr, _sum);
                outptr += 4;
                bptr++;
            }
        }

        return 0;
    }

    if (elempack == 8)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create(embed_dim, seqlen, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -1;

        Mat xq(embed_dim_per_head, seqlen, num_head, 16u, 8, opt.workspace_allocator);
        Mat xk(embed_dim_per_head, seqlen, num_head, 16u, 8, opt.workspace_allocator);
        Mat xv(seqlen, embed_dim_per_head, num_head, 16u, 8, opt.workspace_allocator);

        Mat xqk(8 * seqlen, seqlen, num_head, 16u, 8, opt.workspace_allocator);

        Mat xqkv(embed_dim_per_head, num_head, seqlen, 16u, 8, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_head; q++)
        {
            // xq = affine(q) * inv_sqrt_embed_dim_per_head
            {
                __fp16* outptr = (__fp16*)xq.channel(q);

                const __fp16* bptr0 = (const __fp16*)q_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)q_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = q_blob.row<__fp16>(i);

                        float16x8_t _sum = vdupq_n_f16(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float16x8_t _val = vld1q_f16(ptr);
                            float16x8_t _k = vdupq_n_f16(*kptr);
                            _sum = vfmaq_f16(_sum, _val, _k);
                            ptr += 8;
                            kptr += 1;
                        }

                        float16x8_t _slope = vdupq_n_f16(inv_sqrt_embed_dim_per_head);
                        _sum = vmulq_f16(_sum, _slope);

                        vst1q_f16(outptr, _sum);
                        outptr += 8;
                        bptr++;
                    }
                }
            }

            // xk = affine(k)
            {
                __fp16* outptr = (__fp16*)xk.channel(q);

                const __fp16* bptr0 = (const __fp16*)k_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)k_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < seqlen; i++)
                {
                    const __fp16* bptr = bptr0;
                    const __fp16* kptr = kptr0;

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* ptr = k_blob.row<__fp16>(i);

                        float16x8_t _sum = vdupq_n_f16(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float16x8_t _val = vld1q_f16(ptr);
                            float16x8_t _k = vdupq_n_f16(*kptr);
                            _sum = vfmaq_f16(_sum, _val, _k);
                            ptr += 8;
                            kptr += 1;
                        }

                        vst1q_f16(outptr, _sum);
                        outptr += 8;
                        bptr++;
                    }
                }
            }

            // xv = affine(v)
            {
                __fp16* outptr = (__fp16*)xv.channel(q);

                const __fp16* bptr = (const __fp16*)v_bias_data_fp16 + q * embed_dim_per_head;
                const __fp16* kptr0 = (const __fp16*)v_weight_data_fp16 + embed_dim * q * embed_dim_per_head;

                for (int i = 0; i < embed_dim_per_head; i++)
                {
                    for (int j = 0; j < seqlen; j++)
                    {
                        const __fp16* ptr = v_blob.row<__fp16>(j);
                        const __fp16* kptr = kptr0;

                        float16x8_t _sum = vdupq_n_f16(*bptr);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float16x8_t _val = vld1q_f16(ptr);
                            float16x8_t _k = vdupq_n_f16(*kptr);
                            _sum = vfmaq_f16(_sum, _val, _k);
                            ptr += 8;
                            kptr += 1;
                        }

                        vst1q_f16(outptr, _sum);
                        outptr += 8;
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
                    __fp16* outptr = outm.row<__fp16>(i);

                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        const __fp16* qptr = xqm.row<__fp16>(i);
                        const __fp16* kptr = upxkm.row<__fp16>(j);

                        float16x8_t _sum = vdupq_n_f16(0.f);
                        for (int k = 0; k < embed_dim_per_head; k++)
                        {
                            float16x8_t _q = vld1q_f16(qptr);
                            float16x8_t _k = vdupq_n_f16(*kptr);
                            _sum = vfmaq_f16(_sum, _q, _k);
                            qptr += 8;
                            kptr += 1;
                        }

                        vst1q_f16(outptr, _sum);
                        outptr += 8;
                    }
                }
            }

            // softmax(xqk)
            {
                Mat outm = xqk.channel(q);
                for (int i = 0; i < seqlen; i++)
                {
                    __fp16* ptr = outm.row<__fp16>(i);

                    __fp16* ptr0 = ptr;
                    float16x8_t _max = vdupq_n_f16(-FLT_MAX);
                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float16x8_t _p = vld1q_f16(ptr0);
                        _max = vmaxq_f16(_max, _p);
                        ptr0 += 8;
                    }

                    ptr0 = ptr;
                    float16x8_t _sum = vdupq_n_f16(0.f);
                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float16x8_t _p = vld1q_f16(ptr0);
                        _p = exp_ps(vsubq_f16(_p, _max));
                        vst1q_f16(ptr0, _p);
                        _sum = vaddq_f16(_sum, _p);
                        ptr0 += 8;
                    }

                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        _p = div_ps_float16x8(_p, _sum);
                        vst1q_f16(ptr, _p);
                        ptr += 8;
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
                    __fp16* outptr = xqkv.channel(i).row<__fp16>(q);

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const __fp16* qkptr = xqkm.row<__fp16>(i);
                        const __fp16* vptr = xvm.row<__fp16>(j);

                        float16x8_t _sum = vdupq_n_f16(0.f);
                        for (int k = 0; k < seqlen * elempack; k++)
                        {
                            float16x8_t _qk = vld1q_f16(qkptr);
                            float16x8_t _v = vdupq_n_f16(*vptr);
                            _sum = vfmaq_f16(_sum, _qk, _v);
                            qkptr += 8;
                            vptr += 1;
                        }

                        vst1q_f16(outptr, _sum);
                        outptr += 8;
                    }
                }
            }
        }

        // out = affine(xqkv)
        // xqkv  (embed_dim, seqlen)
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < seqlen; i++)
        {
            __fp16* outptr = top_blob.row<__fp16>(i);

            const __fp16* bptr = (const __fp16*)out_bias_data_fp16;
            const __fp16* kptr = (const __fp16*)out_weight_data_fp16;

            for (int j = 0; j < embed_dim; j++)
            {
                const __fp16* ptr = (__fp16*)xqkv.channel(i);

                float16x8_t _sum = vdupq_n_f16(*bptr);
                for (int k = 0; k < embed_dim; k++)
                {
                    float16x8_t _val = vld1q_f16(ptr);
                    float16x8_t _k = vdupq_n_f16(*kptr);
                    _sum = vfmaq_f16(_sum, _val, _k);
                    ptr += 8;
                    kptr += 1;
                }

                vst1q_f16(outptr, _sum);
                outptr += 8;
                bptr++;
            }
        }

        return 0;
    }

    return 0;
}

#endif

} // namespace ncnn
