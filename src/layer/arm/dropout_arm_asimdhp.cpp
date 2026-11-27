// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "dropout_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Dropout_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;
        float32x4_t _scale = vdupq_n_f32(scale);
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(vmulq_f32(_p0, _scale)), vcvt_f16_f32(vmulq_f32(_p1, _scale))));
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            vst1_f16(ptr, vcvt_f16_f32(vmulq_f32(_p, _scale)));
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr = (__fp16)((float)*ptr * scale);
            ptr++;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
