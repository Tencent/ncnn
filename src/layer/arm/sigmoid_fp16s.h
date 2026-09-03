// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int sigmoid_fp16s_asimdhp(Mat& bottom_top_blob, const Option& opt);
int sigmoid_fp16sa_asimdhp(Mat& bottom_top_blob, const Option& opt);
#endif

static int sigmoid_fp16s(Mat& bottom_top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
        return sigmoid_fp16s_asimdhp(bottom_top_blob, opt);
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

        int i = 0;
        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p01 = vld1q_f16(ptr);
            float16x8_t _p23 = vld1q_f16(ptr + 8);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p01));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p01));
            float32x4_t _p2 = vcvt_f32_f16(vget_low_f16(_p23));
            float32x4_t _p3 = vcvt_f32_f16(vget_high_f16(_p23));
            _p0 = sigmoid_ps(_p0);
            _p1 = sigmoid_ps(_p1);
            _p2 = sigmoid_ps(_p2);
            _p3 = sigmoid_ps(_p3);
            _p01 = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
            _p23 = vcombine_f16(vcvt_f16_f32(_p2), vcvt_f16_f32(_p3));
            vst1q_f16(ptr, _p01);
            vst1q_f16(ptr + 8, _p23);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            _p0 = sigmoid_ps(_p0);
            _p1 = sigmoid_ps(_p1);
            _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _p = sigmoid_ps(_p);
            vst1_f16(ptr, vcvt_f16_f32(_p));
            ptr += 4;
        }
        for (; i < size; i++)
        {
            float v = (float)*ptr;
            v = 1.f / (1.f + expf(-v));
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

static int sigmoid_fp16sa(Mat& bottom_top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
        return sigmoid_fp16sa_asimdhp(bottom_top_blob, opt);
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

        int i = 0;
        for (; i + 31 < size; i += 32)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            float16x8_t _p2 = vld1q_f16(ptr + 16);
            float16x8_t _p3 = vld1q_f16(ptr + 24);
            _p0 = sigmoid_ps_f16(_p0);
            _p1 = sigmoid_ps_f16(_p1);
            _p2 = sigmoid_ps_f16(_p2);
            _p3 = sigmoid_ps_f16(_p3);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            vst1q_f16(ptr + 16, _p2);
            vst1q_f16(ptr + 24, _p3);
            ptr += 32;
        }
        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            _p0 = sigmoid_ps_f16(_p0);
            _p1 = sigmoid_ps_f16(_p1);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = sigmoid_ps_f16(_p);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = sigmoid_ps_f16(_p);
            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = *ptr;
            v = 1.f / (1.f + expf(-v));
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

