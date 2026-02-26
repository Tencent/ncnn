// Copyright 2018 Tencent
// Copyright 2019 BUG1989
// SPDX-License-Identifier: BSD-3-Clause

#include "quantize_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

Quantize_arm::Quantize_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void quantize(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize %d   %d %d", scale_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __ARM_NEON
    float32x4_t _scale = vdupq_n_f32(scale);
    if (scale_data_size > 1)
    {
        if (elempack == 4)
        {
            _scale = vld1q_f32((const float*)scale_data);
        }
    }
#endif // __ARM_NEON

    int i = 0;
#if __ARM_NEON
    for (; i + 15 < size; i += 16)
    {
        float32x4_t _v0 = vld1q_f32(ptr);
        float32x4_t _v1 = vld1q_f32(ptr + 4);
        float32x4_t _v2 = vld1q_f32(ptr + 8);
        float32x4_t _v3 = vld1q_f32(ptr + 12);
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
        float32x4_t _v0 = vld1q_f32(ptr);
        float32x4_t _v1 = vld1q_f32(ptr + 4);
        _v0 = vmulq_f32(_v0, _scale);
        _v1 = vmulq_f32(_v1, _scale);
        vst1_s8(s8ptr, float2int8(_v0, _v1));
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _v = vld1q_f32(ptr);
        _v = vmulq_f32(_v, _scale);
        int8x8_t v = float2int8(_v, _v);
        s8ptr[0] = vget_lane_s8(v, 0);
        s8ptr[1] = vget_lane_s8(v, 1);
        s8ptr[2] = vget_lane_s8(v, 2);
        s8ptr[3] = vget_lane_s8(v, 3);
        ptr += 4;
        s8ptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        float v = *ptr * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

#if __ARM_NEON
static void quantize_pack4to8(const float* ptr0, const float* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to8 %d   %d", scale_data_size, elemcount);

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
        float32x4_t _v0 = vld1q_f32(ptr0);
        float32x4_t _v1 = vld1q_f32(ptr1);
        float32x4_t _v2 = vld1q_f32(ptr0 + 4);
        float32x4_t _v3 = vld1q_f32(ptr1 + 4);
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
        float32x4_t _v0 = vld1q_f32(ptr0);
        float32x4_t _v1 = vld1q_f32(ptr1);
        _v0 = vmulq_f32(_v0, _scale0);
        _v1 = vmulq_f32(_v1, _scale1);
        vst1_s8(s8ptr, float2int8(_v0, _v1));
        ptr0 += 4;
        ptr1 += 4;
        s8ptr += 8;
    }
}

static void quantize_pack4to1(const float* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to1 %d   %d", scale_data_size, elemcount);

    float scale = scale_data[0];
    float32x4_t _scale = vdupq_n_f32(scale);
    if (scale_data_size > 1)
    {
        _scale = vld1q_f32((const float*)scale_data);
    }

    int i = 0;
    for (; i + 7 < elemcount; i += 8)
    {
        float32x4_t _v0 = vld1q_f32(ptr);
        float32x4_t _v1 = vld1q_f32(ptr + 4);
        float32x4_t _v2 = vld1q_f32(ptr + 8);
        float32x4_t _v3 = vld1q_f32(ptr + 12);
        float32x4_t _v4 = vld1q_f32(ptr + 16);
        float32x4_t _v5 = vld1q_f32(ptr + 20);
        float32x4_t _v6 = vld1q_f32(ptr + 24);
        float32x4_t _v7 = vld1q_f32(ptr + 28);
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
        float32x4_t _v = vld1q_f32(ptr);
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
#endif // __ARM_NEON

int Quantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blob, top_blob, opt);
#endif

    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        int out_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            out_elempack = w * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outw = w * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const float* ptr = (const float*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;

            // assert scale_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            quantize(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        int out_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            out_elempack = h * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outh = h * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ARM_NEON
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* ptr0 = bottom_blob.row(i * 2);
                const float* ptr1 = bottom_blob.row(i * 2 + 1);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8(ptr0, ptr1, s8ptr, scale_data_i, w);
            }
        }
        if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr0 = top_blob.row<signed char>(i * 4);
                signed char* s8ptr1 = top_blob.row<signed char>(i * 4 + 1);
                signed char* s8ptr2 = top_blob.row<signed char>(i * 4 + 2);
                signed char* s8ptr3 = top_blob.row<signed char>(i * 4 + 3);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_i, w);
            }
        }
#endif // __ARM_NEON
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize(ptr, s8ptr, scale_data_i, w, elempack);
            }
        }
    }

    if (dims == 3)
    {
        int out_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            out_elempack = channels * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outc = channels * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ARM_NEON
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* ptr0 = bottom_blob.channel(q * 2);
                const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8(ptr0, ptr1, s8ptr, scale_data_q, w * h);
            }
        }
        if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr0 = top_blob.channel(q * 4);
                signed char* s8ptr1 = top_blob.channel(q * 4 + 1);
                signed char* s8ptr2 = top_blob.channel(q * 4 + 2);
                signed char* s8ptr3 = top_blob.channel(q * 4 + 3);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_q, w * h);
            }
        }
#endif // __ARM_NEON
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize(ptr, s8ptr, scale_data_q, w * h, elempack);
            }
        }
    }

    return 0;
}

#if NCNN_BF16
static void quantize_bf16s(const unsigned short* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize_bf16s %d   %d %d", scale_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __ARM_NEON
    float32x4_t _scale = vdupq_n_f32(scale);
    if (scale_data_size > 1)
    {
        if (elempack == 4)
        {
            _scale = vld1q_f32((const float*)scale_data);
        }
    }
#endif // __ARM_NEON

    int i = 0;
#if __ARM_NEON
    for (; i + 15 < size; i += 16)
    {
        uint16x8_t _v01 = vld1q_u16(ptr);
        uint16x8_t _v23 = vld1q_u16(ptr + 8);
        float32x4_t _v0 = bfloat2float(vget_low_u16(_v01));
        float32x4_t _v1 = bfloat2float(vget_high_u16(_v01));
        float32x4_t _v2 = bfloat2float(vget_low_u16(_v23));
        float32x4_t _v3 = bfloat2float(vget_high_u16(_v23));
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
        uint16x8_t _v01 = vld1q_u16(ptr);
        float32x4_t _v0 = bfloat2float(vget_low_u16(_v01));
        float32x4_t _v1 = bfloat2float(vget_high_u16(_v01));
        _v0 = vmulq_f32(_v0, _scale);
        _v1 = vmulq_f32(_v1, _scale);
        vst1_s8(s8ptr, float2int8(_v0, _v1));
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _v = bfloat2float(vld1_u16(ptr));
        _v = vmulq_f32(_v, _scale);
        int8x8_t v = float2int8(_v, _v);
        s8ptr[0] = vget_lane_s8(v, 0);
        s8ptr[1] = vget_lane_s8(v, 1);
        s8ptr[2] = vget_lane_s8(v, 2);
        s8ptr[3] = vget_lane_s8(v, 3);
        ptr += 4;
        s8ptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        float v = bfloat16_to_float32(*ptr) * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

#if __ARM_NEON
static void quantize_pack4to8_bf16s(const unsigned short* ptr0, const unsigned short* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to8_bf16s %d   %d", scale_data_size, elemcount);

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
        uint16x8_t _v02 = vld1q_u16(ptr0);
        uint16x8_t _v13 = vld1q_u16(ptr1);
        float32x4_t _v0 = bfloat2float(vget_low_u16(_v02));
        float32x4_t _v1 = bfloat2float(vget_low_u16(_v13));
        float32x4_t _v2 = bfloat2float(vget_high_u16(_v02));
        float32x4_t _v3 = bfloat2float(vget_high_u16(_v13));
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
        float32x4_t _v0 = bfloat2float(vld1_u16(ptr0));
        float32x4_t _v1 = bfloat2float(vld1_u16(ptr1));
        _v0 = vmulq_f32(_v0, _scale0);
        _v1 = vmulq_f32(_v1, _scale1);
        vst1_s8(s8ptr, float2int8(_v0, _v1));
        ptr0 += 4;
        ptr1 += 4;
        s8ptr += 8;
    }
}

static void quantize_pack4to1_bf16s(const unsigned short* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to1_bf16s %d   %d", scale_data_size, elemcount);

    float scale = scale_data[0];
    float32x4_t _scale = vdupq_n_f32(scale);
    if (scale_data_size > 1)
    {
        _scale = vld1q_f32((const float*)scale_data);
    }

    int i = 0;
    for (; i + 7 < elemcount; i += 8)
    {
        uint16x8_t _v01 = vld1q_u16(ptr);
        uint16x8_t _v23 = vld1q_u16(ptr + 8);
        uint16x8_t _v45 = vld1q_u16(ptr + 16);
        uint16x8_t _v67 = vld1q_u16(ptr + 24);
        float32x4_t _v0 = bfloat2float(vget_low_u16(_v01));
        float32x4_t _v1 = bfloat2float(vget_high_u16(_v01));
        float32x4_t _v2 = bfloat2float(vget_low_u16(_v23));
        float32x4_t _v3 = bfloat2float(vget_high_u16(_v23));
        float32x4_t _v4 = bfloat2float(vget_low_u16(_v45));
        float32x4_t _v5 = bfloat2float(vget_high_u16(_v45));
        float32x4_t _v6 = bfloat2float(vget_low_u16(_v67));
        float32x4_t _v7 = bfloat2float(vget_high_u16(_v67));
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
        float32x4_t _v = bfloat2float(vld1_u16(ptr));
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
#endif // __ARM_NEON

int Quantize_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        int out_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            out_elempack = w * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outw = w * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const unsigned short* ptr = (const unsigned short*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;

            // assert scale_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            quantize_bf16s(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        int out_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            out_elempack = h * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outh = h * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ARM_NEON
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* ptr0 = bottom_blob.row<const unsigned short>(i * 2);
                const unsigned short* ptr1 = bottom_blob.row<const unsigned short>(i * 2 + 1);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8_bf16s(ptr0, ptr1, s8ptr, scale_data_i, w);
            }
        }
        if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(i);
                signed char* s8ptr0 = top_blob.row<signed char>(i * 4);
                signed char* s8ptr1 = top_blob.row<signed char>(i * 4 + 1);
                signed char* s8ptr2 = top_blob.row<signed char>(i * 4 + 2);
                signed char* s8ptr3 = top_blob.row<signed char>(i * 4 + 3);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_pack4to1_bf16s(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_i, w);
            }
        }
#endif // __ARM_NEON
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(i);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_bf16s(ptr, s8ptr, scale_data_i, w, elempack);
            }
        }
    }

    if (dims == 3)
    {
        int out_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            out_elempack = channels * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outc = channels * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ARM_NEON
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q * 2);
                const unsigned short* ptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8_bf16s(ptr0, ptr1, s8ptr, scale_data_q, w * h);
            }
        }
        if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                signed char* s8ptr0 = top_blob.channel(q * 4);
                signed char* s8ptr1 = top_blob.channel(q * 4 + 1);
                signed char* s8ptr2 = top_blob.channel(q * 4 + 2);
                signed char* s8ptr3 = top_blob.channel(q * 4 + 3);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_pack4to1_bf16s(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_q, w * h);
            }
        }
#endif // __ARM_NEON
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_bf16s(ptr, s8ptr, scale_data_q, w * h, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
