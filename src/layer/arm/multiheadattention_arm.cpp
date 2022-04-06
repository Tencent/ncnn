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
#include "neon_mathfun.h"

#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

MultiHeadAttention_arm::MultiHeadAttention_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int MultiHeadAttention_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& q_blob = bottom_blobs[0];
    const Mat& k_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[1];
    const Mat& v_blob = bottom_blobs.size() == 1 ? q_blob : bottom_blobs[2];

    size_t elemsize = q_blob.elemsize;
    int elempack = q_blob.elempack;

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
                Mat outm = xq.channel(q);

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = outm.row(i);

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const float* ptr = q_blob.row(i);
                        const float* kptr = (const float*)q_weight_data + embed_dim * (q * embed_dim_per_head + j);

                        float32x4_t _sum = vdupq_n_f32(q_bias_data[q * embed_dim_per_head + j]);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vld1q_f32(ptr);
                            float32x4_t _k = vdupq_n_f32(kptr[0]);
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        float32x4_t _slope = vdupq_n_f32(inv_sqrt_embed_dim_per_head);
                        _sum = vmulq_f32(_sum, _slope);

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                    }
                }
            }

            // xk = affine(k)
            {
                Mat outm = xk.channel(q);

                for (int i = 0; i < seqlen; i++)
                {
                    float* outptr = outm.row(i);

                    for (int j = 0; j < embed_dim_per_head; j++)
                    {
                        const float* ptr = k_blob.row(i);
                        const float* kptr = (const float*)k_weight_data + embed_dim * (q * embed_dim_per_head + j);

                        float32x4_t _sum = vdupq_n_f32(k_bias_data[q * embed_dim_per_head + j]);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vld1q_f32(ptr);
                            float32x4_t _k = vdupq_n_f32(kptr[0]);
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                    }
                }
            }

            // xv = affine(v)
            {
                Mat outm = xv.channel(q);

                for (int i = 0; i < embed_dim_per_head; i++)
                {
                    float* outptr = outm.row(i);

                    for (int j = 0; j < seqlen; j++)
                    {
                        const float* ptr = v_blob.row(j);
                        const float* kptr = (const float*)v_weight_data + embed_dim * (q * embed_dim_per_head + i);

                        float32x4_t _sum = vdupq_n_f32(v_bias_data[q * embed_dim_per_head + i]);
                        for (int k = 0; k < embed_dim; k++)
                        {
                            float32x4_t _val = vld1q_f32(ptr);
                            float32x4_t _k = vdupq_n_f32(kptr[0]);
                            _sum = vmlaq_f32(_sum, _val, _k);
                            ptr += 4;
                            kptr += 1;
                        }

                        vst1q_f32(outptr, _sum);
                        outptr += 4;
                    }
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
                            float32x4_t _k = vdupq_n_f32(kptr[0]);
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

                    float32x4_t _max = vdupq_n_f32(-FLT_MAX);
                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr + j * 4);
                        _max = vmaxq_f32(_max, _p);
                    }

                    float32x4_t _sum = vdupq_n_f32(0.f);
                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr + j * 4);
                        _p = exp_ps(vsubq_f32(_p, _max));
                        vst1q_f32(ptr + j * 4, _p);
                        _sum = vaddq_f32(_sum, _p);
                    }

                    for (int j = 0; j < seqlen * elempack; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr + j * 4);
#if __aarch64__
                        _p = vdivq_f32(_p, _sum);
#else
                        _p = div_ps(_p, _sum);
#endif
                        vst1q_f32(ptr + j * 4, _p);
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
                            float32x4_t _v = vdupq_n_f32(vptr[0]);
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

            for (int j = 0; j < embed_dim; j++)
            {
                const float* ptr = xqkv.channel(i);
                const float* kptr = (const float*)out_weight_data + embed_dim * j;

                float32x4_t _sum = vdupq_n_f32(out_bias_data[j]);
                for (int k = 0; k < embed_dim; k++)
                {
                    float32x4_t _val = vld1q_f32(ptr);
                    float32x4_t _k = vdupq_n_f32(kptr[0]);
                    _sum = vmlaq_f32(_sum, _val, _k);
                    ptr += 4;
                    kptr += 1;
                }

                vst1q_f32(outptr, _sum);
                outptr += 4;
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    // fallback to native implement
    return MultiHeadAttention::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
