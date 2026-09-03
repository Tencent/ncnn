// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int gelu_fp16s_asimdhp(Mat& bottom_top_blob, int fast_gelu, const Option& opt);
int gelu_fp16sa_asimdhp(Mat& bottom_top_blob, int fast_gelu, const Option& opt);
#endif

static int gelu_fp16s(Mat& bottom_top_blob, int fast_gelu, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
        return gelu_fp16s_asimdhp(bottom_top_blob, fast_gelu, opt);
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

    if (fast_gelu)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob.channel(q);

            int i = 0;

            for (; i + 3 < size; i += 4)
            {
                float32x4_t _pLoad = vcvt_f32_f16(vld1_f16(ptr));
                float32x4_t _blob = fast_gelu_ps(_pLoad);
                vst1_f16(ptr, vcvt_f16_f32(_blob));
                ptr += 4;
            }

            for (; i < size; i++)
            {
                float v = (float)*ptr;
                v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
                *ptr = (__fp16)v;
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob.channel(q);

            int i = 0;

            for (; i + 3 < size; i += 4)
            {
                float32x4_t _pLoad = vcvt_f32_f16(vld1_f16(ptr));
                float32x4_t _blob = gelu_ps(_pLoad);
                vst1_f16(ptr, vcvt_f16_f32(_blob));
                ptr += 4;
            }

            for (; i < size; i++)
            {
                float v = (float)*ptr;
                v = 0.5f * v * erfcf(-0.70710678f * v);
                *ptr = (__fp16)v;
                ptr++;
            }
        }
    }

    return 0;

#else
    (void)bottom_top_blob;
    (void)fast_gelu;
    (void)opt;
    return 0;
#endif
}

static int gelu_fp16sa(Mat& bottom_top_blob, int fast_gelu, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
        return gelu_fp16sa_asimdhp(bottom_top_blob, fast_gelu, opt);
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

    if (fast_gelu)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob.channel(q);

            int i = 0;

            for (; i + 7 < size; i += 8)
            {
                float16x8_t _pLoad = vld1q_f16(ptr);
                float16x8_t _blob = fast_gelu_ps_f16(_pLoad);
                vst1q_f16(ptr, _blob);
                ptr += 8;
            }

            for (; i < size; i++)
            {
                *ptr = (__fp16)0.5f * *ptr * (__fp16)(1.0f + tanhf((__fp16)0.79788452f * (*ptr + (__fp16)0.044715f * *ptr * *ptr * *ptr)));
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob.channel(q);

            int i = 0;

            for (; i + 7 < size; i += 8)
            {
                float16x8_t _pLoad = vld1q_f16(ptr);
                float16x8_t _blob = gelu_ps_f16(_pLoad);
                vst1q_f16(ptr, _blob);
                ptr += 8;
            }

            for (; i < size; i++)
            {
                float v = (float)*ptr;
                v = 0.5f * v * erfcf(-0.70710678f * v);
                *ptr = (__fp16)v;
                ptr++;
            }
        }
    }

    return 0;

#else
    (void)bottom_top_blob;
    (void)fast_gelu;
    (void)opt;
    return 0;
#endif
}

