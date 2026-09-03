// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void dequantize_fp16s(const int* intptr, __fp16* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("dequantize_fp16s %d %d   %d %d", scale_data_size, bias_data_size, elemcount, elempack);

    float scale = scale_data[0];
    float32x4_t _scale0 = vdupq_n_f32(scale);
    float32x4_t _scale1 = _scale0;
    if (scale_data_size > 1)
    {
        if (elempack == 8)
        {
            _scale0 = vld1q_f32((const float*)scale_data);
            _scale1 = vld1q_f32((const float*)scale_data + 4);
        }
        if (elempack == 4)
        {
            _scale0 = vld1q_f32((const float*)scale_data);
            _scale1 = _scale0;
        }
    }

    if (bias_data_size == 0)
    {
        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
            _v0 = vmulq_f32(_v0, _scale0);
            _v1 = vmulq_f32(_v1, _scale1);
            vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
            _v = vmulq_f32(_v, _scale0);
            vst1_f16(ptr, vcvt_f16_f32(_v));
            intptr += 4;
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr = (__fp16)(*intptr * scale);
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
        float32x4_t _bias0 = vdupq_n_f32(bias);
        float32x4_t _bias1 = _bias0;
        if (bias_data_size > 1)
        {
            if (elempack == 8)
            {
                _bias0 = vld1q_f32((const float*)bias_data);
                _bias1 = vld1q_f32((const float*)bias_data + 4);
            }
            if (elempack == 4)
            {
                _bias0 = vld1q_f32((const float*)bias_data);
                _bias1 = _bias0;
            }
        }

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
            _v0 = vfmaq_f32(_bias0, _v0, _scale0);
            _v1 = vfmaq_f32(_bias1, _v1, _scale1);
            vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
            _v = vfmaq_f32(_bias0, _v, _scale0);
            vst1_f16(ptr, vcvt_f16_f32(_v));
            intptr += 4;
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr = (__fp16)(*intptr * scale + bias);
            intptr++;
            ptr++;
        }
    }
}
#endif

