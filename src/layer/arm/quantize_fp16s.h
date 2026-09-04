// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void quantize_fp16s_asimdhp(const unsigned short* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack);
void quantize_pack4to8_fp16s_asimdhp(const unsigned short* ptr0, const unsigned short* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount);
void quantize_pack4to1_fp16s_asimdhp(const unsigned short* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount);
void quantize_fp16sa_asimdhp(const unsigned short* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack);
void quantize_pack4to1_fp16sa_asimdhp(const unsigned short* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount);
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void quantize_fp16s(const __fp16* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize_fp16s %d   %d %d", scale_data_size, elemcount, elempack);

    float scale = scale_data[0];
    float32x4_t _scale = vdupq_n_f32(scale);
    if (scale_data_size > 1)
    {
        if (elempack == 4)
        {
            _scale = vld1q_f32((const float*)scale_data);
        }
    }

    int i = 0;
    for (; i + 15 < size; i += 16)
    {
        float16x8_t _v01 = vld1q_f16(ptr);
        float16x8_t _v23 = vld1q_f16(ptr + 8);
        float32x4_t _v0 = vcvt_f32_f16(vget_low_f16(_v01));
        float32x4_t _v1 = vcvt_f32_f16(vget_high_f16(_v01));
        float32x4_t _v2 = vcvt_f32_f16(vget_low_f16(_v23));
        float32x4_t _v3 = vcvt_f32_f16(vget_high_f16(_v23));
        _v0 = vmulq_f32(_v0, _scale);
        _v1 = vmulq_f32(_v1, _scale);
        _v2 = vmulq_f32(_v2, _scale);
        _v3 = vmulq_f32(_v3, _scale);
        vst1q_s8(s8ptr, vcombine_s8(float2int8(_v0, _v1), float2int8(_v2, _v3)));
        ptr += 16;
        s8ptr += 16;
    }
    for (; i + 7 < size; i += 8)
    {
        float16x8_t _v01 = vld1q_f16(ptr);
        float32x4_t _v0 = vcvt_f32_f16(vget_low_f16(_v01));
        float32x4_t _v1 = vcvt_f32_f16(vget_high_f16(_v01));
        _v0 = vmulq_f32(_v0, _scale);
        _v1 = vmulq_f32(_v1, _scale);
        vst1_s8(s8ptr, float2int8(_v0, _v1));
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _v = vcvt_f32_f16(vld1_f16(ptr));
        _v = vmulq_f32(_v, _scale);
        int8x8_t v = float2int8(_v, _v);
        s8ptr[0] = vget_lane_s8(v, 0);
        s8ptr[1] = vget_lane_s8(v, 1);
        s8ptr[2] = vget_lane_s8(v, 2);
        s8ptr[3] = vget_lane_s8(v, 3);
        ptr += 4;
        s8ptr += 4;
    }
    for (; i < size; i++)
    {
        float v = (float)(*ptr) * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

static void quantize_pack4to8_fp16s(const __fp16* ptr0, const __fp16* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to8_fp16s %d   %d", scale_data_size, elemcount);

    float scale = scale_data[0];
    float32x4_t _scale0 = vdupq_n_f32(scale);
    float32x4_t _scale1 = _scale0;
    if (scale_data_size > 1)
    {
        _scale0 = vld1q_f32((const float*)scale_data);
        _scale1 = vld1q_f32((const float*)scale_data + 4);
    }

    int i = 0;
    for (; i + 1 < elemcount; i += 2)
    {
        float16x8_t _v02 = vld1q_f16(ptr0);
        float16x8_t _v13 = vld1q_f16(ptr1);
        float32x4_t _v0 = vcvt_f32_f16(vget_low_f16(_v02));
        float32x4_t _v1 = vcvt_f32_f16(vget_low_f16(_v13));
        float32x4_t _v2 = vcvt_f32_f16(vget_high_f16(_v02));
        float32x4_t _v3 = vcvt_f32_f16(vget_high_f16(_v13));
        _v0 = vmulq_f32(_v0, _scale0);
        _v1 = vmulq_f32(_v1, _scale1);
        _v2 = vmulq_f32(_v2, _scale0);
        _v3 = vmulq_f32(_v3, _scale1);
        vst1q_s8(s8ptr, vcombine_s8(float2int8(_v0, _v1), float2int8(_v2, _v3)));
        ptr0 += 8;
        ptr1 += 8;
        s8ptr += 16;
    }
    for (; i < elemcount; i++)
    {
        float32x4_t _v0 = vcvt_f32_f16(vld1_f16(ptr0));
        float32x4_t _v1 = vcvt_f32_f16(vld1_f16(ptr1));
        _v0 = vmulq_f32(_v0, _scale0);
        _v1 = vmulq_f32(_v1, _scale1);
        vst1_s8(s8ptr, float2int8(_v0, _v1));
        ptr0 += 4;
        ptr1 += 4;
        s8ptr += 8;
    }
}

static void quantize_pack4to1_fp16s(const __fp16* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to1_fp16s %d   %d", scale_data_size, elemcount);

    float scale = scale_data[0];
    float32x4_t _scale = vdupq_n_f32(scale);
    if (scale_data_size > 1)
    {
        _scale = vld1q_f32((const float*)scale_data);
    }

    int i = 0;
    for (; i + 7 < elemcount; i += 8)
    {
        float16x8_t _v01 = vld1q_f16(ptr);
        float16x8_t _v23 = vld1q_f16(ptr + 8);
        float16x8_t _v45 = vld1q_f16(ptr + 16);
        float16x8_t _v67 = vld1q_f16(ptr + 24);
        float32x4_t _v0 = vcvt_f32_f16(vget_low_f16(_v01));
        float32x4_t _v1 = vcvt_f32_f16(vget_high_f16(_v01));
        float32x4_t _v2 = vcvt_f32_f16(vget_low_f16(_v23));
        float32x4_t _v3 = vcvt_f32_f16(vget_high_f16(_v23));
        float32x4_t _v4 = vcvt_f32_f16(vget_low_f16(_v45));
        float32x4_t _v5 = vcvt_f32_f16(vget_high_f16(_v45));
        float32x4_t _v6 = vcvt_f32_f16(vget_low_f16(_v67));
        float32x4_t _v7 = vcvt_f32_f16(vget_high_f16(_v67));
        _v0 = vmulq_f32(_v0, _scale);
        _v1 = vmulq_f32(_v1, _scale);
        _v2 = vmulq_f32(_v2, _scale);
        _v3 = vmulq_f32(_v3, _scale);
        _v4 = vmulq_f32(_v4, _scale);
        _v5 = vmulq_f32(_v5, _scale);
        _v6 = vmulq_f32(_v6, _scale);
        _v7 = vmulq_f32(_v7, _scale);
        int8x8_t v0 = float2int8(_v0, _v1);
        int8x8_t v1 = float2int8(_v2, _v3);
        int8x8_t v2 = float2int8(_v4, _v5);
        int8x8_t v3 = float2int8(_v6, _v7);
        int8x16_t v01 = vcombine_s8(v0, v1);
        int8x16_t v23 = vcombine_s8(v2, v3);
        int8x16x2_t v0213 = vuzpq_s8(v01, v23);
        int8x16x2_t v0123 = vuzpq_s8(v0213.val[0], v0213.val[1]);
        vst1_s8(s8ptr0, vget_low_s8(v0123.val[0]));
        vst1_s8(s8ptr1, vget_high_s8(v0123.val[0]));
        vst1_s8(s8ptr2, vget_low_s8(v0123.val[1]));
        vst1_s8(s8ptr3, vget_high_s8(v0123.val[1]));
        ptr += 32;
        s8ptr0 += 8;
        s8ptr1 += 8;
        s8ptr2 += 8;
        s8ptr3 += 8;
    }
    for (; i < elemcount; i++)
    {
        float32x4_t _v = vcvt_f32_f16(vld1_f16(ptr));
        _v = vmulq_f32(_v, _scale);
        int8x8_t v = float2int8(_v, _v);
        s8ptr0[0] = vget_lane_s8(v, 0);
        s8ptr1[0] = vget_lane_s8(v, 1);
        s8ptr2[0] = vget_lane_s8(v, 2);
        s8ptr3[0] = vget_lane_s8(v, 3);
        ptr += 4;
        s8ptr0 += 1;
        s8ptr1 += 1;
        s8ptr2 += 1;
        s8ptr3 += 1;
    }
}

static void quantize_fp16sa(const __fp16* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize_fp16sa %d   %d %d", scale_data_size, elemcount, elempack);

    __fp16 scale = (__fp16)scale_data[0];
    float16x4_t _scale0 = vdup_n_f16(scale);
    float16x4_t _scale1 = _scale0;
    if (scale_data_size > 1)
    {
        if (elempack == 8)
        {
            _scale0 = vcvt_f16_f32(vld1q_f32((const float*)scale_data));
            _scale1 = vcvt_f16_f32(vld1q_f32((const float*)scale_data + 4));
        }
        if (elempack == 4)
        {
            _scale0 = vcvt_f16_f32(vld1q_f32((const float*)scale_data));
            _scale1 = _scale0;
        }
    }
    float16x8_t _scale = vcombine_f16(_scale0, _scale1);

    int i = 0;
    for (; i + 7 < size; i += 8)
    {
        float16x8_t _v = vld1q_f16(ptr);
        _v = vmulq_f16(_v, _scale);
        vst1_s8(s8ptr, float2int8(_v));
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float16x4_t _v = vld1_f16(ptr);
        _v = vmul_f16(_v, _scale0);
        int8x8_t v = float2int8(vcombine_f16(_v, _v));
        s8ptr[0] = vget_lane_s8(v, 0);
        s8ptr[1] = vget_lane_s8(v, 1);
        s8ptr[2] = vget_lane_s8(v, 2);
        s8ptr[3] = vget_lane_s8(v, 3);
        ptr += 4;
        s8ptr += 4;
    }
    for (; i < size; i++)
    {
        __fp16 v = *ptr * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

static void quantize_pack4to1_fp16sa(const __fp16* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to1_fp16sa %d   %d", scale_data_size, elemcount);

    __fp16 scale = (__fp16)scale_data[0];
    float16x4_t _scale = vdup_n_f16(scale);
    if (scale_data_size > 1)
    {
        _scale = vcvt_f16_f32(vld1q_f32((const float*)scale_data));
    }
    float16x8_t _scale01 = vcombine_f16(_scale, _scale);

    int i = 0;
    for (; i + 7 < elemcount; i += 8)
    {
        float16x8_t _v01 = vld1q_f16(ptr);
        float16x8_t _v23 = vld1q_f16(ptr + 8);
        float16x8_t _v45 = vld1q_f16(ptr + 16);
        float16x8_t _v67 = vld1q_f16(ptr + 24);
        _v01 = vmulq_f16(_v01, _scale01);
        _v23 = vmulq_f16(_v23, _scale01);
        _v45 = vmulq_f16(_v45, _scale01);
        _v67 = vmulq_f16(_v67, _scale01);
        int8x8_t v0 = float2int8(_v01);
        int8x8_t v1 = float2int8(_v23);
        int8x8_t v2 = float2int8(_v45);
        int8x8_t v3 = float2int8(_v67);
        int8x16_t v01 = vcombine_s8(v0, v1);
        int8x16_t v23 = vcombine_s8(v2, v3);
        int8x16x2_t v0213 = vuzpq_s8(v01, v23);
        int8x16x2_t v0123 = vuzpq_s8(v0213.val[0], v0213.val[1]);
        vst1_s8(s8ptr0, vget_low_s8(v0123.val[0]));
        vst1_s8(s8ptr1, vget_high_s8(v0123.val[0]));
        vst1_s8(s8ptr2, vget_low_s8(v0123.val[1]));
        vst1_s8(s8ptr3, vget_high_s8(v0123.val[1]));
        ptr += 32;
        s8ptr0 += 8;
        s8ptr1 += 8;
        s8ptr2 += 8;
        s8ptr3 += 8;
    }
    for (; i < elemcount; i++)
    {
        float16x4_t _v = vld1_f16(ptr);
        _v = vmul_f16(_v, _scale);
        int8x8_t v = float2int8(vcombine_f16(_v, _v));
        s8ptr0[0] = vget_lane_s8(v, 0);
        s8ptr1[0] = vget_lane_s8(v, 1);
        s8ptr2[0] = vget_lane_s8(v, 2);
        s8ptr3[0] = vget_lane_s8(v, 3);
        ptr += 4;
        s8ptr0 += 1;
        s8ptr1 += 1;
        s8ptr2 += 1;
        s8ptr3 += 1;
    }
}
#endif

static void quantize_fp16s(const unsigned short* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    quantize_fp16s((const __fp16*)ptr, s8ptr, scale_data, elemcount, elempack);
#elif NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__
    quantize_fp16s_asimdhp(ptr, s8ptr, scale_data, elemcount, elempack);
#else
    (void)ptr;
    (void)s8ptr;
    (void)scale_data;
    (void)elemcount;
    (void)elempack;
#endif
}

static void quantize_pack4to8_fp16s(const unsigned short* ptr0, const unsigned short* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    quantize_pack4to8_fp16s((const __fp16*)ptr0, (const __fp16*)ptr1, s8ptr, scale_data, elemcount);
#elif NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__
    quantize_pack4to8_fp16s_asimdhp(ptr0, ptr1, s8ptr, scale_data, elemcount);
#else
    (void)ptr0;
    (void)ptr1;
    (void)s8ptr;
    (void)scale_data;
    (void)elemcount;
#endif
}

static void quantize_pack4to1_fp16s(const unsigned short* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    quantize_pack4to1_fp16s((const __fp16*)ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data, elemcount);
#elif NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__
    quantize_pack4to1_fp16s_asimdhp(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data, elemcount);
#else
    (void)ptr;
    (void)s8ptr0;
    (void)s8ptr1;
    (void)s8ptr2;
    (void)s8ptr3;
    (void)scale_data;
    (void)elemcount;
#endif
}

static void quantize_fp16sa(const unsigned short* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    quantize_fp16sa((const __fp16*)ptr, s8ptr, scale_data, elemcount, elempack);
#elif NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__
    quantize_fp16sa_asimdhp(ptr, s8ptr, scale_data, elemcount, elempack);
#else
    (void)ptr;
    (void)s8ptr;
    (void)scale_data;
    (void)elemcount;
    (void)elempack;
#endif
}

static void quantize_pack4to1_fp16sa(const unsigned short* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    quantize_pack4to1_fp16sa((const __fp16*)ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data, elemcount);
#elif NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__
    quantize_pack4to1_fp16sa_asimdhp(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data, elemcount);
#else
    (void)ptr;
    (void)s8ptr0;
    (void)s8ptr1;
    (void)s8ptr2;
    (void)s8ptr3;
    (void)scale_data;
    (void)elemcount;
#endif
}
