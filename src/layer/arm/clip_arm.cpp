//
// Created by RogerOu on 2018/9/28.
//

#include "clip_arm.h"


#define __ARM_NEON

#ifdef __ARM_NEON

#include <arm_neon.h>

#endif // __ARM_NEON


namespace ncnn {
    DEFINE_LAYER_CREATOR(Clip_arm)

    int Clip_arm::forward_inplace(Mat &bottom_top_blob, const Option &opt) const {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

#ifdef __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remian = size;
#endif

#ifdef __ARM_NEON
        float32x4_t maxf32 = vmovq_n_f32(max);
        float32x4_t minf32 = vmovq_n_f32(min);
#endif

#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < channels; ++i) {
            float *channel_ptr = bottom_top_blob.channel(i);
#ifdef __ARM_NEON
#ifdef __aarch64__
            for (; nn > 0; --nn) {
                float32x4_t clip_f32 = vld1q_f32(channel_ptr);
                float32x4_t clip_min_f32 = vmaxq_f32(minf32, clip_f32);
                float32x4_t clip_max_f32 = vminq_f32(maxf32, clip_min_f32);
                vst1_f32(channel_ptr, clip_max_f32);
                channel_ptr += 4;
            }
#else
            if (nn > 0) {
                asm volatile(
                "0:"
                "pld        [%1,    #128]           \n"
                "vld1.f32   {d0-d1},    [%1:128]    \n"

                "vmax.f32   q1, %q4,    q0          \n"
                "vmin.f32   q2, %q5,    q1          \n"

                "subs       %0,          #1          \n"
                "vst1.f32   {d4-d5},    [%1:128]!   \n"

                "bne        0b                      \n"

                :"=r"(nn),              //%0
                "=r"(channel_ptr)       //%1
                :"0"(nn),
                "1"(channel_ptr),
                "w"(minf32),            //%q4
                "w"(maxf32)             //%q5
                :"cc", "memory", "q0", "q1", "q2"
                );

            }
#endif      // __aarch64__
#endif      // __ARM_NEON

            for (; remain > 0; --remain) {

                if (*channel_ptr < min) {
                    *channel_ptr = min;
                }

                if (*channel_ptr > max) {
                    *channel_ptr = max;
                }

                ++channel_ptr;
            }

        }

        return 0;
    }
}