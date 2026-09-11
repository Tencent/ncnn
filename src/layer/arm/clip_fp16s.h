// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int clip_fp16s_asimdhp(Mat& bottom_top_blob, float min, float max, const Option& opt);
#endif

static int clip_fp16s(Mat& bottom_top_blob, float min, float max, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
        return clip_fp16s_asimdhp(bottom_top_blob, min, max, opt);
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        __fp16 min_fp16 = min;
        __fp16 max_fp16 = max;

        int i = 0;
#if __ARM_FEATURE_SVE
        const int packn = svcnth();
        const svbool_t _pg = svptrue_b16();
        const svfloat16_t _min = svdup_n_f16(min_fp16);
        const svfloat16_t _max = svdup_n_f16(max_fp16);
        for (; i + packn * 4 <= size; i += packn * 4)
        {
            svfloat16_t _p0 = svld1_f16(_pg, ptr);
            svfloat16_t _p1 = svld1_f16(_pg, ptr + packn);
            svfloat16_t _p2 = svld1_f16(_pg, ptr + packn * 2);
            svfloat16_t _p3 = svld1_f16(_pg, ptr + packn * 3);
            _p0 = svmax_f16_x(_pg, _p0, _min);
            _p1 = svmax_f16_x(_pg, _p1, _min);
            _p2 = svmax_f16_x(_pg, _p2, _min);
            _p3 = svmax_f16_x(_pg, _p3, _min);
            _p0 = svmin_f16_x(_pg, _p0, _max);
            _p1 = svmin_f16_x(_pg, _p1, _max);
            _p2 = svmin_f16_x(_pg, _p2, _max);
            _p3 = svmin_f16_x(_pg, _p3, _max);
            svst1_f16(_pg, ptr, _p0);
            svst1_f16(_pg, ptr + packn, _p1);
            svst1_f16(_pg, ptr + packn * 2, _p2);
            svst1_f16(_pg, ptr + packn * 3, _p3);
            ptr += packn * 4;
        }
        for (; i + packn <= size; i += packn)
        {
            svfloat16_t _p = svld1_f16(_pg, ptr);
            _p = svmax_f16_x(_pg, _p, _min);
            _p = svmin_f16_x(_pg, _p, _max);
            svst1_f16(_pg, ptr, _p);
            ptr += packn;
        }
        if (i < size)
        {
            const svbool_t _pg1 = svwhilelt_b16((unsigned int)i, (unsigned int)size);
            svfloat16_t _p = svld1_f16(_pg1, ptr);
            _p = svmax_f16_x(_pg1, _p, _min);
            _p = svmin_f16_x(_pg1, _p, _max);
            svst1_f16(_pg1, ptr, _p);
        }
#else
        float16x8_t _min = vdupq_n_f16(min_fp16);
        float16x8_t _max = vdupq_n_f16(max_fp16);
        for (; i + 31 < size; i += 32)
        {
#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "prfm   pldl1keep, [%0, #512]   \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                "fmax   v0.8h, v0.8h, %2.8h     \n"
                "fmax   v1.8h, v1.8h, %2.8h     \n"
                "fmax   v2.8h, v2.8h, %2.8h     \n"
                "fmax   v3.8h, v3.8h, %2.8h     \n"
                "fmin   v0.8h, v0.8h, %3.8h     \n"
                "fmin   v1.8h, v1.8h, %3.8h     \n"
                "fmin   v2.8h, v2.8h, %3.8h     \n"
                "fmin   v3.8h, v3.8h, %3.8h     \n"
                "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                : "=r"(ptr) // %0
                : "0"(ptr),
                "w"(_min), // %2
                "w"(_max)  // %3
                : "memory", "v0", "v1", "v2", "v3");
#else  // NCNN_GNU_INLINE_ASM
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            _p0 = vmaxq_f16(_p0, _min);
            _p1 = vmaxq_f16(_p1, _min);
            _p2 = vmaxq_f16(_p2, _min);
            _p3 = vmaxq_f16(_p3, _min);
            _p0 = vminq_f16(_p0, _max);
            _p1 = vminq_f16(_p1, _max);
            _p2 = vminq_f16(_p2, _max);
            _p3 = vminq_f16(_p3, _max);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            ptr += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            _p0 = vmaxq_f16(_p0, _min);
            _p1 = vmaxq_f16(_p1, _min);
            _p0 = vminq_f16(_p0, _max);
            _p1 = vminq_f16(_p1, _max);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = vmaxq_f16(_p, _min);
            _p = vminq_f16(_p, _max);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = vmax_f16(_p, vget_low_f16(_min));
            _p = vmin_f16(_p, vget_low_f16(_max));
            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = *ptr;
            if (v < min_fp16)
                v = min_fp16;

            if (v > max_fp16)
                v = max_fp16;

            *ptr = v;
            ptr++;
        }
#endif // __ARM_FEATURE_SVE
    }

    return 0;
#else
    return 0;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
}
