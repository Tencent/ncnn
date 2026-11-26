// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_arm.h"

#include <float.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
// fp16 storage element access for the decode kernels, which compute in fp32
// these overloads must be declared before the kernel templates
static inline float sdpa_cvt(__fp16 v)
{
    return (float)v;
}

static inline void sdpa_store1(__fp16* p, float v)
{
    *p = (__fp16)v;
}

static inline float32x4_t sdpa_load4(const __fp16* p)
{
    return vcvt_f32_f16(vld1_f16(p));
}

static inline float32x4_t sdpa_load4_dup(const __fp16* p)
{
    return vdupq_n_f32((float)*p);
}

static inline void sdpa_store4(__fp16* p, float32x4_t _v)
{
    vst1_f16(p, vcvt_f16_f32(_v));
}

#include "sdpa_decode.h"

int SDPA_arm::forward_decode_fp16s(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt) const
{
    return sdpa_decode<__fp16, __fp16>(query, key, value, attn_mask_blob, top_blob, scale, opt);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
