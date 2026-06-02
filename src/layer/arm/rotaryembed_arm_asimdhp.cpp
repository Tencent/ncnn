// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"

namespace ncnn {

#if NCNN_ARM82
int RotaryEmbed_arm::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;
    const int half = embed_dim / 2;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            const __fp16* cos_ptr = cos_cache.row<const __fp16>(i);
            const __fp16* sin_ptr = sin_cache.row<const __fp16>(i);

            if (interleaved)
            {
                const __fp16* ptr = head.row<const __fp16>(i);
                __fp16* outptr = out_head.row<__fp16>(i);

                int j = 0;
                for (; j + 3 < half; j += 4)
                {
                    float16x4x2_t _p = vld2_f16(ptr); // _p.val[0]=x0 lanes, _p.val[1]=x1 lanes
                    float32x4_t _x0 = vcvt_f32_f16(_p.val[0]);
                    float32x4_t _x1 = vcvt_f32_f16(_p.val[1]);
                    float32x4_t _c = vcvt_f32_f16(vld1_f16(cos_ptr));
                    float32x4_t _s = vcvt_f32_f16(vld1_f16(sin_ptr));

                    float16x4x2_t _out;
                    _out.val[0] = vcvt_f16_f32(vmlsq_f32(vmulq_f32(_x0, _c), _x1, _s)); // x0*c - x1*s
                    _out.val[1] = vcvt_f16_f32(vmlaq_f32(vmulq_f32(_x1, _c), _x0, _s)); // x0*s + x1*c
                    vst2_f16(outptr, _out);

                    ptr += 8;
                    outptr += 8;
                    cos_ptr += 4;
                    sin_ptr += 4;
                }
                for (; j < half; j++)
                {
                    const float x0 = (float)ptr[0];
                    const float x1 = (float)ptr[1];
                    const float cos_val = (float)*cos_ptr++;
                    const float sin_val = (float)*sin_ptr++;
                    outptr[0] = (__fp16)(x0 * cos_val - x1 * sin_val);
                    outptr[1] = (__fp16)(x0 * sin_val + x1 * cos_val);
                    ptr += 2;
                    outptr += 2;
                }
            }
            else
            {
                const __fp16* ptr0 = head.row<const __fp16>(i);
                const __fp16* ptr1 = ptr0 + half;
                __fp16* outptr0 = out_head.row<__fp16>(i);
                __fp16* outptr1 = outptr0 + half;

                int j = 0;
                for (; j + 3 < half; j += 4)
                {
                    float32x4_t _x0 = vcvt_f32_f16(vld1_f16(ptr0));
                    float32x4_t _x1 = vcvt_f32_f16(vld1_f16(ptr1));
                    float32x4_t _c = vcvt_f32_f16(vld1_f16(cos_ptr));
                    float32x4_t _s = vcvt_f32_f16(vld1_f16(sin_ptr));

                    float32x4_t _y0 = vmlsq_f32(vmulq_f32(_x0, _c), _x1, _s); // x0*c - x1*s
                    float32x4_t _y1 = vmlaq_f32(vmulq_f32(_x1, _c), _x0, _s); // x1*c + x0*s

                    vst1_f16(outptr0, vcvt_f16_f32(_y0));
                    vst1_f16(outptr1, vcvt_f16_f32(_y1));

                    ptr0 += 4;
                    ptr1 += 4;
                    cos_ptr += 4;
                    sin_ptr += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
                for (; j < half; j++)
                {
                    const float x0 = (float)*ptr0++;
                    const float x1 = (float)*ptr1++;
                    const float cos_val = (float)*cos_ptr++;
                    const float sin_val = (float)*sin_ptr++;
                    *outptr0++ = (__fp16)(x0 * cos_val - x1 * sin_val);
                    *outptr1++ = (__fp16)(x0 * sin_val + x1 * cos_val);
                }
            }
        }
    }

    return 0;
}

int RotaryEmbed_arm::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;
    const int half = embed_dim / 2;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            const __fp16* cos_ptr = cos_cache.row<const __fp16>(i);
            const __fp16* sin_ptr = sin_cache.row<const __fp16>(i);

            if (interleaved)
            {
                const __fp16* ptr = head.row<const __fp16>(i);
                __fp16* outptr = out_head.row<__fp16>(i);

                int j = 0;
                for (; j + 7 < half; j += 8)
                {
                    float16x8x2_t _p = vld2q_f16(ptr); // _p.val[0]=x0 lanes, _p.val[1]=x1 lanes
                    float16x8_t _c = vld1q_f16(cos_ptr);
                    float16x8_t _s = vld1q_f16(sin_ptr);

                    float16x8x2_t _out;
                    _out.val[0] = vfmsq_f16(vmulq_f16(_p.val[0], _c), _p.val[1], _s); // x0*c - x1*s
                    _out.val[1] = vfmaq_f16(vmulq_f16(_p.val[1], _c), _p.val[0], _s); // x0*s + x1*c
                    vst2q_f16(outptr, _out);

                    ptr += 16;
                    outptr += 16;
                    cos_ptr += 8;
                    sin_ptr += 8;
                }
                for (; j < half; j++)
                {
                    const float x0 = (float)ptr[0];
                    const float x1 = (float)ptr[1];
                    const float cos_val = (float)*cos_ptr++;
                    const float sin_val = (float)*sin_ptr++;
                    outptr[0] = (__fp16)(x0 * cos_val - x1 * sin_val);
                    outptr[1] = (__fp16)(x0 * sin_val + x1 * cos_val);
                    ptr += 2;
                    outptr += 2;
                }
            }
            else
            {
                const __fp16* ptr0 = head.row<const __fp16>(i);
                const __fp16* ptr1 = ptr0 + half;
                __fp16* outptr0 = out_head.row<__fp16>(i);
                __fp16* outptr1 = outptr0 + half;

                int j = 0;
                for (; j + 7 < half; j += 8)
                {
                    float16x8_t _x0 = vld1q_f16(ptr0);
                    float16x8_t _x1 = vld1q_f16(ptr1);
                    float16x8_t _c = vld1q_f16(cos_ptr);
                    float16x8_t _s = vld1q_f16(sin_ptr);

                    float16x8_t _y0 = vfmsq_f16(vmulq_f16(_x0, _c), _x1, _s); // x0*c - x1*s
                    float16x8_t _y1 = vfmaq_f16(vmulq_f16(_x1, _c), _x0, _s); // x1*c + x0*s

                    vst1q_f16(outptr0, _y0);
                    vst1q_f16(outptr1, _y1);

                    ptr0 += 8;
                    ptr1 += 8;
                    cos_ptr += 8;
                    sin_ptr += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
                for (; j < half; j++)
                {
                    const float x0 = (float)*ptr0++;
                    const float x1 = (float)*ptr1++;
                    const float cos_val = (float)*cos_ptr++;
                    const float sin_val = (float)*sin_ptr++;
                    *outptr0++ = (__fp16)(x0 * cos_val - x1 * sin_val);
                    *outptr1++ = (__fp16)(x0 * sin_val + x1 * cos_val);
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_ARM82

} // namespace ncnn
