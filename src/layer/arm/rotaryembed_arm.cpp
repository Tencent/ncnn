// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

RotaryEmbed_arm::RotaryEmbed_arm()
{
#if __ARM_NEON
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON
}

int RotaryEmbed_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && bottom_blobs[0].elembits() == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blobs, top_blobs, opt);
        else
            return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif

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
            const float* cos_ptr = cos_cache.row(i);
            const float* sin_ptr = sin_cache.row(i);

            if (interleaved)
            {
                const float* ptr = head.row(i);
                float* outptr = out_head.row(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < half; j += 4)
                {
                    float32x4x2_t _p = vld2q_f32(ptr); // _p.val[0]=x0 lanes, _p.val[1]=x1 lanes
                    float32x4_t _c = vld1q_f32(cos_ptr);
                    float32x4_t _s = vld1q_f32(sin_ptr);

                    float32x4x2_t _out;
                    _out.val[0] = vmlsq_f32(vmulq_f32(_p.val[0], _c), _p.val[1], _s); // x0*c - x1*s
                    _out.val[1] = vmlaq_f32(vmulq_f32(_p.val[1], _c), _p.val[0], _s); // x0*s + x1*c
                    vst2q_f32(outptr, _out);

                    ptr += 8;
                    outptr += 8;
                    cos_ptr += 4;
                    sin_ptr += 4;
                }
#endif // __ARM_NEON
                for (; j < half; j++)
                {
                    const float x0 = ptr[0];
                    const float x1 = ptr[1];
                    const float cos_val = *cos_ptr++;
                    const float sin_val = *sin_ptr++;
                    outptr[0] = x0 * cos_val - x1 * sin_val;
                    outptr[1] = x0 * sin_val + x1 * cos_val;
                    ptr += 2;
                    outptr += 2;
                }
            }
            else
            {
                const float* ptr0 = head.row(i);
                const float* ptr1 = ptr0 + half;
                float* outptr0 = out_head.row(i);
                float* outptr1 = outptr0 + half;

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < half; j += 4)
                {
                    float32x4_t _x0 = vld1q_f32(ptr0);
                    float32x4_t _x1 = vld1q_f32(ptr1);
                    float32x4_t _c = vld1q_f32(cos_ptr);
                    float32x4_t _s = vld1q_f32(sin_ptr);

                    float32x4_t _y0 = vmlsq_f32(vmulq_f32(_x0, _c), _x1, _s); // x0*c - x1*s
                    float32x4_t _y1 = vmlaq_f32(vmulq_f32(_x1, _c), _x0, _s); // x1*c + x0*s

                    vst1q_f32(outptr0, _y0);
                    vst1q_f32(outptr1, _y1);

                    ptr0 += 4;
                    ptr1 += 4;
                    cos_ptr += 4;
                    sin_ptr += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
#endif // __ARM_NEON
                for (; j < half; j++)
                {
                    const float x0 = *ptr0++;
                    const float x1 = *ptr1++;
                    const float cos_val = *cos_ptr++;
                    const float sin_val = *sin_ptr++;
                    *outptr0++ = x0 * cos_val - x1 * sin_val;
                    *outptr1++ = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
