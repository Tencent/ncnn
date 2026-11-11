// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swiglu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "swiglu_arm_math.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

SwiGLU_arm::SwiGLU_arm()
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
static void swiglu_vector_f32(const float* gateptr, const float* upptr, float* outptr, int size, float scale, float offset, float limit)
{
    const bool use_limit = limit > 0.f;

    int i = 0;
    float32x4_t _scale = vdupq_n_f32(scale);
    float32x4_t _offset = vdupq_n_f32(offset);
    float32x4_t _limit = vdupq_n_f32(limit);
    float32x4_t _mlimit = vdupq_n_f32(-limit);
#if __aarch64__
    for (; i + 15 < size; i += 16)
    {
        float32x4_t _g0 = vld1q_f32(gateptr);
        float32x4_t _g1 = vld1q_f32(gateptr + 4);
        float32x4_t _g2 = vld1q_f32(gateptr + 8);
        float32x4_t _g3 = vld1q_f32(gateptr + 12);
        float32x4_t _u0 = vld1q_f32(upptr);
        float32x4_t _u1 = vld1q_f32(upptr + 4);
        float32x4_t _u2 = vld1q_f32(upptr + 8);
        float32x4_t _u3 = vld1q_f32(upptr + 12);
        vst1q_f32(outptr, swiglu_f32x4(_g0, _u0, _scale, _offset, _limit, _mlimit, use_limit));
        vst1q_f32(outptr + 4, swiglu_f32x4(_g1, _u1, _scale, _offset, _limit, _mlimit, use_limit));
        vst1q_f32(outptr + 8, swiglu_f32x4(_g2, _u2, _scale, _offset, _limit, _mlimit, use_limit));
        vst1q_f32(outptr + 12, swiglu_f32x4(_g3, _u3, _scale, _offset, _limit, _mlimit, use_limit));
        gateptr += 16;
        upptr += 16;
        outptr += 16;
    }
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _g0 = vld1q_f32(gateptr);
        float32x4_t _g1 = vld1q_f32(gateptr + 4);
        float32x4_t _u0 = vld1q_f32(upptr);
        float32x4_t _u1 = vld1q_f32(upptr + 4);
        vst1q_f32(outptr, swiglu_f32x4(_g0, _u0, _scale, _offset, _limit, _mlimit, use_limit));
        vst1q_f32(outptr + 4, swiglu_f32x4(_g1, _u1, _scale, _offset, _limit, _mlimit, use_limit));
        gateptr += 8;
        upptr += 8;
        outptr += 8;
    }
#endif // __aarch64__
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _g = vld1q_f32(gateptr);
        float32x4_t _u = vld1q_f32(upptr);
        vst1q_f32(outptr, swiglu_f32x4(_g, _u, _scale, _offset, _limit, _mlimit, use_limit));
        gateptr += 4;
        upptr += 4;
        outptr += 4;
    }
    for (; i < size; i++)
    {
        float g = gateptr[0];
        float u = upptr[0];
        if (use_limit)
        {
            g = std::min(g, limit);
            u = std::min(std::max(u, -limit), limit);
        }
        outptr[0] = g / (1.f + expf(-scale * g)) * (u + offset);
        gateptr++;
        upptr++;
        outptr++;
    }
}
#endif // __ARM_NEON

int SwiGLU_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() != 2)
        return -100;

    const Mat& gate = bottom_blobs[0];
    const Mat& up = bottom_blobs[1];

    if (gate.dims != up.dims || gate.w != up.w || gate.h != up.h || gate.d != up.d || gate.c != up.c || gate.elempack != up.elempack || gate.elembits() != up.elembits())
        return -100;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(gate, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if NCNN_ARM82 || NCNN_BF16
    const int elembits = gate.elembits();
#endif

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(gate, up, top_blob, opt);
        else
            return forward_fp16s(gate, up, top_blob, opt);
    }
#endif // NCNN_ARM82

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(gate, up, top_blob, opt);
#endif // NCNN_BF16

    return forward_fp32(gate, up, top_blob, opt);
}

int SwiGLU_arm::forward_fp32(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
{
    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;

    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offset = ii % nn_block * block_size;
        const int len = std::min(size - offset, block_size);

        const float* gateptr = (const float*)gate.channel(q) + offset;
        const float* upptr = (const float*)up.channel(q) + offset;
        float* outptr = (float*)top_blob.channel(q) + offset;

#if __ARM_NEON
        swiglu_vector_f32(gateptr, upptr, outptr, len, sigmoid_scale, up_offset, activation_limit);
#else
        for (int i = 0; i < len; i++)
        {
            float g = gateptr[i];
            float u = upptr[i];
            if (activation_limit > 0.f)
            {
                g = std::min(g, activation_limit);
                u = std::min(std::max(u, -activation_limit), activation_limit);
            }
            outptr[i] = g / (1.f + expf(-sigmoid_scale * g)) * (u + up_offset);
        }
#endif // __ARM_NEON
    }

    return 0;
}

#if NCNN_BF16
int SwiGLU_arm::forward_bf16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
{
    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;

    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offset = ii % nn_block * block_size;
        const int len = std::min(size - offset, block_size);

        const unsigned short* gateptr = (const unsigned short*)gate.channel(q) + offset;
        const unsigned short* upptr = (const unsigned short*)up.channel(q) + offset;
        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + offset;

        const bool use_limit = activation_limit > 0.f;

        int i = 0;
#if __ARM_NEON
        float32x4_t _scale = vdupq_n_f32(sigmoid_scale);
        float32x4_t _offset = vdupq_n_f32(up_offset);
        float32x4_t _limit = vdupq_n_f32(activation_limit);
        float32x4_t _mlimit = vdupq_n_f32(-activation_limit);
#if __aarch64__
        for (; i + 7 < len; i += 8)
        {
            uint16x8_t _g = vld1q_u16(gateptr);
            uint16x8_t _u = vld1q_u16(upptr);
            float32x4_t _g0 = swiglu_f32x4(bfloat2float(vget_low_u16(_g)), bfloat2float(vget_low_u16(_u)), _scale, _offset, _limit, _mlimit, use_limit);
            float32x4_t _g1 = swiglu_f32x4(bfloat2float(vget_high_u16(_g)), bfloat2float(vget_high_u16(_u)), _scale, _offset, _limit, _mlimit, use_limit);
            vst1q_u16(outptr, vcombine_u16(float2bfloat(_g0), float2bfloat(_g1)));
            gateptr += 8;
            upptr += 8;
            outptr += 8;
        }
#endif // __aarch64__
        for (; i + 3 < len; i += 4)
        {
            float32x4_t _g = swiglu_f32x4(bfloat2float(vld1_u16(gateptr)), bfloat2float(vld1_u16(upptr)), _scale, _offset, _limit, _mlimit, use_limit);
            vst1_u16(outptr, float2bfloat(_g));
            gateptr += 4;
            upptr += 4;
            outptr += 4;
        }
#endif // __ARM_NEON
        for (; i < len; i++)
        {
            float g = bfloat16_to_float32(gateptr[0]);
            float u = bfloat16_to_float32(upptr[0]);
            if (use_limit)
            {
                g = std::min(g, activation_limit);
                u = std::min(std::max(u, -activation_limit), activation_limit);
            }
            outptr[0] = float32_to_bfloat16(g / (1.f + expf(-sigmoid_scale * g)) * (u + up_offset));
            gateptr++;
            upptr++;
            outptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
