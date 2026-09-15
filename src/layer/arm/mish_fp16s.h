// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int mish_fp16s_asimdhp(Mat& bottom_top_blob, const Option& opt);
int mish_fp16sa_asimdhp(Mat& bottom_top_blob, const Option& opt);
#endif

static int mish_fp16s(Mat& bottom_top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
        return mish_fp16s_asimdhp(bottom_top_blob, opt);
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;
    int elempack = bottom_top_blob.elempack;

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                _p = vmulq_f32(_p, tanh_ps(log_ps(vaddq_f32(exp_ps(_p), vdupq_n_f32(1.f)))));
                vst1_f16(ptr, vcvt_f16_f32(_p));

                ptr += 4;
            }
        }

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _p = vmulq_f32(_p, tanh_ps(log_ps(vaddq_f32(exp_ps(_p), vdupq_n_f32(1.f)))));
            vst1_f16(ptr, vcvt_f16_f32(_p));

            ptr += 4;
        }
        for (; i < size; i++)
        {
            float v = (float)*ptr;
            v = v * tanhf(logf(expf(v) + 1.f));
            *ptr = (__fp16)v;
            ptr++;
        }
    }

    return 0;

#else
    (void)bottom_top_blob;
    (void)opt;
    return 0;
#endif
}

static int mish_fp16sa(Mat& bottom_top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
        return mish_fp16sa_asimdhp(bottom_top_blob, opt);
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;
    int elempack = bottom_top_blob.elempack;

    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float16x8_t _p = vld1q_f16(ptr);
                _p = vmulq_f16(_p, tanh_ps_f16(log_ps_f16(vaddq_f16(exp_ps_f16(_p), vdupq_n_f16(1.f)))));
                vst1q_f16(ptr, _p);

                ptr += 8;
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float16x4_t _p = vld1_f16(ptr);
                _p = vmul_f16(_p, tanh_ps_f16(log_ps_f16(vadd_f16(exp_ps_f16(_p), vdup_n_f16(1.f)))));
                vst1_f16(ptr, _p);

                ptr += 4;
            }
        }

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = vmul_f16(_p, tanh_ps_f16(log_ps_f16(vadd_f16(exp_ps_f16(_p), vdup_n_f16(1.f)))));
            vst1_f16(ptr, _p);

            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = *ptr;
            v = v * (__fp16)tanhf(logf(expf(v) + 1.f));
            *ptr = v;
            ptr++;
        }
    }

    return 0;

#else
    (void)bottom_top_blob;
    (void)opt;
    return 0;
#endif
}
