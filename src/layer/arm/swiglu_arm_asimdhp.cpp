// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swiglu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "swiglu_arm_math.h"
#endif // __ARM_NEON

#include "arm_usability.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int SwiGLU_arm::forward_fp16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
{
    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;
    const float climit = activation_limit > 0.f ? activation_limit : 65504.f;
    const float mlimit = -climit;

    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offset = ii % nn_block * block_size;
        const int len = std::min(size - offset, block_size);

        const __fp16* gateptr = (const __fp16*)gate.channel(q) + offset;
        const __fp16* upptr = (const __fp16*)up.channel(q) + offset;
        __fp16* outptr = (__fp16*)top_blob.channel(q) + offset;

        const float32x4_t _scale = vdupq_n_f32(sigmoid_scale);
        const float32x4_t _offset = vdupq_n_f32(up_offset);
        const float32x4_t _limit = vdupq_n_f32(climit);
        const float32x4_t _mlimit = vdupq_n_f32(mlimit);

        int i = 0;
        for (; i + 7 < len; i += 8)
        {
            float16x8_t _g = vld1q_f16(gateptr);
            float16x8_t _u = vld1q_f16(upptr);
            float32x4_t _out0 = swiglu_f32x4(vcvt_f32_f16(vget_low_f16(_g)), vcvt_f32_f16(vget_low_f16(_u)), _scale, _offset, _limit, _mlimit);
            float32x4_t _out1 = swiglu_f32x4(vcvt_f32_f16(vget_high_f16(_g)), vcvt_f32_f16(vget_high_f16(_u)), _scale, _offset, _limit, _mlimit);
            vst1q_f16(outptr, vcombine_f16(vcvt_f16_f32(_out0), vcvt_f16_f32(_out1)));
            gateptr += 8;
            upptr += 8;
            outptr += 8;
        }
        for (; i + 3 < len; i += 4)
        {
            float32x4_t _out = swiglu_f32x4(vcvt_f32_f16(vld1_f16(gateptr)), vcvt_f32_f16(vld1_f16(upptr)), _scale, _offset, _limit, _mlimit);
            vst1_f16(outptr, vcvt_f16_f32(_out));
            gateptr += 4;
            upptr += 4;
            outptr += 4;
        }
        for (; i < len; i++)
        {
            float g = (float)gateptr[0];
            float u = (float)upptr[0];
            g = std::min(g, climit);
            u = std::min(std::max(u, mlimit), climit);
            outptr[0] = (__fp16)(g / (1.f + expf(-sigmoid_scale * g)) * (u + up_offset));
            gateptr++;
            upptr++;
            outptr++;
        }
    }

    return 0;
}

int SwiGLU_arm::forward_fp16sa(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
{
    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;
    const float climit = activation_limit > 0.f ? activation_limit : 65504.f;
    const float mlimit = -climit;

    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offset = ii % nn_block * block_size;
        const int len = std::min(size - offset, block_size);

        const __fp16* gateptr = (const __fp16*)gate.channel(q) + offset;
        const __fp16* upptr = (const __fp16*)up.channel(q) + offset;
        __fp16* outptr = (__fp16*)top_blob.channel(q) + offset;

        const float16x8_t _scale = vdupq_n_f16(sigmoid_scale);
        const float16x8_t _offset = vdupq_n_f16(up_offset);
        const float16x8_t _limit = vdupq_n_f16(climit);
        const float16x8_t _mlimit = vdupq_n_f16(mlimit);
        const float16x8_t _one = vdupq_n_f16(1.f);

        int i = 0;
        for (; i + 15 < len; i += 16)
        {
            float16x8_t _g0 = vld1q_f16(gateptr);
            float16x8_t _g1 = vld1q_f16(gateptr + 8);
            float16x8_t _u0 = vld1q_f16(upptr);
            float16x8_t _u1 = vld1q_f16(upptr + 8);
            vst1q_f16(outptr, swiglu_f16x8(_g0, _u0, _scale, _offset, _limit, _mlimit, _one));
            vst1q_f16(outptr + 8, swiglu_f16x8(_g1, _u1, _scale, _offset, _limit, _mlimit, _one));
            gateptr += 16;
            upptr += 16;
            outptr += 16;
        }
        for (; i + 7 < len; i += 8)
        {
            vst1q_f16(outptr, swiglu_f16x8(vld1q_f16(gateptr), vld1q_f16(upptr), _scale, _offset, _limit, _mlimit, _one));
            gateptr += 8;
            upptr += 8;
            outptr += 8;
        }
        const float16x4_t _scale4 = vdup_n_f16(sigmoid_scale);
        const float16x4_t _offset4 = vdup_n_f16(up_offset);
        const float16x4_t _limit4 = vdup_n_f16(climit);
        const float16x4_t _mlimit4 = vdup_n_f16(mlimit);
        const float16x4_t _one4 = vdup_n_f16(1.f);
        for (; i + 3 < len; i += 4)
        {
            vst1_f16(outptr, swiglu_f16x4(vld1_f16(gateptr), vld1_f16(upptr), _scale4, _offset4, _limit4, _mlimit4, _one4));
            gateptr += 4;
            upptr += 4;
            outptr += 4;
        }
        for (; i < len; i++)
        {
            float g = (float)gateptr[0];
            float u = (float)upptr[0];
            g = std::min(g, climit);
            u = std::min(std::max(u, mlimit), climit);
            outptr[0] = (__fp16)(g / (1.f + expf(-sigmoid_scale * g)) * (u + up_offset));
            gateptr++;
            upptr++;
            outptr++;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
