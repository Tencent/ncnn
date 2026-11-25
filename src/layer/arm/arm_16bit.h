// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ARM_16BIT_H
#define ARM_16BIT_H

#include "mat.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

struct ArmFP16Converter
{
    static float load1(const unsigned short* ptr)
    {
        return float16_to_float32(*ptr);
    }

    static void store1(unsigned short* ptr, float v)
    {
        *ptr = float32_to_float16(v);
    }

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    static float32x4_t load4(const unsigned short* ptr)
    {
        return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr)));
    }

    static void store4(unsigned short* ptr, float32x4_t v)
    {
        vst1_u16(ptr, vreinterpret_u16_f16(vcvt_f16_f32(v)));
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
};

#if NCNN_BF16
struct ArmBF16Converter
{
    static float load1(const unsigned short* ptr)
    {
        return bfloat16_to_float32(*ptr);
    }

    static void store1(unsigned short* ptr, float v)
    {
        *ptr = float32_to_bfloat16(v);
    }

#if __ARM_NEON
    static float32x4_t load4(const unsigned short* ptr)
    {
        return bfloat2float(vld1_u16(ptr));
    }

    static void store4(unsigned short* ptr, float32x4_t v)
    {
        vst1_u16(ptr, float2bfloat(v));
    }
#endif // __ARM_NEON
};
#endif // NCNN_BF16

} // namespace ncnn

#endif // ARM_16BIT_H
