// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "quantize_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

Quantize_loongarch::Quantize_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif
}

static void quantize(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize %d   %d %d", scale_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __loongarch_sx
    __m128 _scale = (__m128)__lsx_vreplfr2vr_s(scale);
#if __loongarch_asx
    __m256 _scale_avx = (__m256)__lasx_xvreplfr2vr_s(scale);
#endif // __loongarch_asx
    if (scale_data_size > 1)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            _scale_avx = (__m256)__lasx_xvld((const float*)scale_data, 0);
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            _scale = (__m128)__lsx_vld((const float*)scale_data, 0);
#if __loongarch_asx
            _scale_avx = combine4x2_ps(_scale, _scale);
#endif // __loongarch_asx
        }
    }
#endif // __loongarch_sx

    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; i + 15 < size; i += 16)
    {
        __m256 _v0 = (__m256)__lasx_xvld(ptr, 0);
        __m256 _v1 = (__m256)__lasx_xvld(ptr + 8, 0);
        _v0 = __lasx_xvfmul_s(_v0, _scale_avx);
        _v1 = __lasx_xvfmul_s(_v1, _scale_avx);
        __m256i _v0i = float2int8(_v0);
        __m256i _v1i = float2int8(_v1);
        __m128i _lo = __lasx_extract_lo128(_v0i);
        __m128i _hi = __lasx_extract_lo128(_v1i);
        __m128i _result = __lsx_vilvl_d(_hi, _lo);
        __lsx_vst(_result, s8ptr, 0);
        ptr += 16;
        s8ptr += 16;
    }
#endif // __loongarch_asx
    for (; i + 7 < size; i += 8)
    {
        __builtin_prefetch(ptr + 32);
#if __loongarch_asx
        __m256 _v = (__m256)__lasx_xvld(ptr, 0);
        _v = __lasx_xvfmul_s(_v, _scale_avx);
        __m256i _vi = float2int8(_v);
        *(int64_t*)s8ptr = __lsx_vpickve2gr_d(__lasx_extract_lo128(_vi), 0);
#else  // __loongarch_asx
        __m128 _v0 = (__m128)__lsx_vld(ptr, 0);
        __m128 _v1 = (__m128)__lsx_vld(ptr + 4, 0);
        _v0 = __lsx_vfmul_s(_v0, _scale);
        _v1 = __lsx_vfmul_s(_v1, _scale);
        *((int64_t*)s8ptr) = float2int8(_v0, _v1);
#endif // __loongarch_asx
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        __m128 _v = (__m128)__lsx_vld(ptr, 0);
        _v = __lsx_vfmul_s(_v, _scale);
        v16i8 v = (v16i8)float2int8(_v);
        s8ptr[0] = v[0];
        s8ptr[1] = v[1];
        s8ptr[2] = v[2];
        s8ptr[3] = v[3];
        ptr += 4;
        s8ptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        float v = *ptr * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

#if __loongarch_sx
static void quantize_pack4to8(const float* ptr0, const float* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to8 %d   %d", scale_data_size, elemcount);

    float scale = scale_data[0];
    __m128 _scale0 = (__m128)__lsx_vreplfr2vr_s(scale);
    __m128 _scale1 = _scale0;
    if (scale_data_size > 1)
    {
        _scale0 = (__m128)__lsx_vld((const float*)scale_data, 0);
        _scale1 = (__m128)__lsx_vld((const float*)scale_data + 4, 0);
    }

    int i = 0;
    for (; i < elemcount; i++)
    {
        __m128 _v0 = (__m128)__lsx_vld(ptr0, 0);
        __m128 _v1 = (__m128)__lsx_vld(ptr1, 0);
        _v0 = __lsx_vfmul_s(_v0, _scale0);
        _v1 = __lsx_vfmul_s(_v1, _scale1);
        *((int64_t*)s8ptr) = float2int8(_v0, _v1);
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
    __m128 _scale = (__m128)__lsx_vreplfr2vr_s(scale);
    if (scale_data_size > 1)
    {
        _scale = (__m128)__lsx_vld((const float*)scale_data, 0);
    }

    int i = 0;
    for (; i < elemcount; i++)
    {
        __m128 _v = (__m128)__lsx_vld(ptr, 0);
        _v = __lsx_vfmul_s(_v, _scale);
        v16i8 v = (v16i8)float2int8(_v);
        s8ptr0[0] = v[0];
        s8ptr1[0] = v[1];
        s8ptr2[0] = v[2];
        s8ptr3[0] = v[3];
        ptr += 4;
        s8ptr0 += 1;
        s8ptr1 += 1;
        s8ptr2 += 1;
        s8ptr3 += 1;
    }
}
#endif // __loongarch_sx

int Quantize_loongarch::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        int out_elempack = 1;
#if __loongarch_sx
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
#if __loongarch_sx
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

#if __loongarch_sx
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
#endif // __loongarch_sx
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
#if __loongarch_sx
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

#if __loongarch_sx
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
#endif // __loongarch_sx
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
static void quantize_bf16(const unsigned short* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    float scale = scale_data[0];
#if __loongarch_sx
    __m128 _scale = (__m128)__lsx_vreplfr2vr_s(scale);
#if __loongarch_asx
    __m256 _scale_avx = (__m256)__lasx_xvreplfr2vr_s(scale);
#endif // __loongarch_asx
    if (scale_data_size > 1)
    {
#if __loongarch_asx
        if (elempack == 8)
        {
            _scale_avx = (__m256)__lasx_xvld((const float*)scale_data, 0);
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            _scale = (__m128)__lsx_vld((const float*)scale_data, 0);
#if __loongarch_asx
            _scale_avx = combine4x2_ps(_scale, _scale);
#endif // __loongarch_asx
        }
    }
#endif // __loongarch_sx

    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; i + 15 < size; i += 16)
    {
        __m256 _v0 = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
        __m256 _v1 = bfloat2float_lasx((__m128i)__lsx_vld(ptr + 8, 0));
        _v0 = __lasx_xvfmul_s(_v0, _scale_avx);
        _v1 = __lasx_xvfmul_s(_v1, _scale_avx);
        __m256i _v0i = float2int8(_v0);
        __m256i _v1i = float2int8(_v1);
        // extract low 32 bits from each, combine into 128-bit (16 bytes)
        __m128i _lo = __lasx_extract_lo128(_v0i);
        __m128i _hi = __lasx_extract_lo128(_v1i);
        __m128i _result = __lsx_vilvl_d(_hi, _lo);
        __lsx_vst(_result, s8ptr, 0);
        ptr += 16;
        s8ptr += 16;
    }
#endif // __loongarch_asx
    for (; i + 7 < size; i += 8)
    {
#if __loongarch_asx
        __m256 _v = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
        _v = __lasx_xvfmul_s(_v, _scale_avx);
        __m256i _vi = float2int8(_v);
        *(int64_t*)s8ptr = __lsx_vpickve2gr_d(__lasx_extract_lo128(_vi), 0);
#else  // __loongarch_asx
        __m128 _v0 = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
        __m128 _v1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr + 4, 0));
        _v0 = __lsx_vfmul_s(_v0, _scale);
        _v1 = __lsx_vfmul_s(_v1, _scale);
        *(int64_t*)s8ptr = float2int8(_v0, _v1);
#endif // __loongarch_asx
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        __m128 _v = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
        _v = __lsx_vfmul_s(_v, _scale);
        __m128i v = float2int8(_v);
        int32_t vi = __lsx_vpickve2gr_w(v, 0);
        s8ptr[0] = (vi >> 0) & 0xff;
        s8ptr[1] = (vi >> 8) & 0xff;
        s8ptr[2] = (vi >> 16) & 0xff;
        s8ptr[3] = (vi >> 24) & 0xff;
        ptr += 4;
        s8ptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        float v = bfloat16_to_float32(*ptr) * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

#if __loongarch_sx
#if !__loongarch_asx
static void quantize_bf16_pack4to8(const unsigned short* ptr0, const unsigned short* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    float scale = scale_data[0];
    __m128 _scale0 = (__m128)__lsx_vreplfr2vr_s(scale);
    __m128 _scale1 = _scale0;
    if (scale_data_size > 1)
    {
        _scale0 = (__m128)__lsx_vld((const float*)scale_data, 0);
        _scale1 = (__m128)__lsx_vld((const float*)scale_data + 4, 0);
    }

    int i = 0;
    for (; i + 1 < elemcount; i += 2)
    {
        __m128 _v0 = bfloat2float_lsx((__m128i)__lsx_vld(ptr0, 0));
        __m128 _v1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr1, 0));
        __m128 _v2 = bfloat2float_lsx((__m128i)__lsx_vld(ptr0 + 4, 0));
        __m128 _v3 = bfloat2float_lsx((__m128i)__lsx_vld(ptr1 + 4, 0));
        _v0 = __lsx_vfmul_s(_v0, _scale0);
        _v1 = __lsx_vfmul_s(_v1, _scale1);
        _v2 = __lsx_vfmul_s(_v2, _scale0);
        _v3 = __lsx_vfmul_s(_v3, _scale1);
        int64_t v01 = float2int8(_v0, _v1);
        int64_t v23 = float2int8(_v2, _v3);
        __m128i _result = __lsx_vinsgr2vr_d(__lsx_vreplgr2vr_d(0), v01, 0);
        _result = __lsx_vinsgr2vr_d(_result, v23, 1);
        __lsx_vst(_result, s8ptr, 0);
        ptr0 += 8;
        ptr1 += 8;
        s8ptr += 16;
    }
    for (; i < elemcount; i++)
    {
        __m128 _v0 = bfloat2float_lsx((__m128i)__lsx_vld(ptr0, 0));
        __m128 _v1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr1, 0));
        _v0 = __lsx_vfmul_s(_v0, _scale0);
        _v1 = __lsx_vfmul_s(_v1, _scale1);
        *(int64_t*)s8ptr = float2int8(_v0, _v1);
        ptr0 += 4;
        ptr1 += 4;
        s8ptr += 8;
    }
}
#endif // !__loongarch_asx

static void quantize_bf16_pack4to1(const unsigned short* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    float scale = scale_data[0];
    __m128 _scale = (__m128)__lsx_vreplfr2vr_s(scale);
    if (scale_data_size > 1)
    {
        _scale = (__m128)__lsx_vld((const float*)scale_data, 0);
    }

    int i = 0;
    for (; i + 7 < elemcount; i += 8)
    {
        __m128 _v0 = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
        __m128 _v1 = bfloat2float_lsx((__m128i)__lsx_vld(ptr + 4, 0));
        __m128 _v2 = bfloat2float_lsx((__m128i)__lsx_vld(ptr + 8, 0));
        __m128 _v3 = bfloat2float_lsx((__m128i)__lsx_vld(ptr + 12, 0));
        __m128 _v4 = bfloat2float_lsx((__m128i)__lsx_vld(ptr + 16, 0));
        __m128 _v5 = bfloat2float_lsx((__m128i)__lsx_vld(ptr + 20, 0));
        __m128 _v6 = bfloat2float_lsx((__m128i)__lsx_vld(ptr + 24, 0));
        __m128 _v7 = bfloat2float_lsx((__m128i)__lsx_vld(ptr + 28, 0));
        _v0 = __lsx_vfmul_s(_v0, _scale);
        _v1 = __lsx_vfmul_s(_v1, _scale);
        _v2 = __lsx_vfmul_s(_v2, _scale);
        _v3 = __lsx_vfmul_s(_v3, _scale);
        _v4 = __lsx_vfmul_s(_v4, _scale);
        _v5 = __lsx_vfmul_s(_v5, _scale);
        _v6 = __lsx_vfmul_s(_v6, _scale);
        _v7 = __lsx_vfmul_s(_v7, _scale);
        int64_t lo0426 = float2int8(_v0, _v4);
        int64_t hi0426 = float2int8(_v2, _v6);
        __m128i v0426 = __lsx_vinsgr2vr_d(__lsx_vreplgr2vr_d(0), lo0426, 0);
        v0426 = __lsx_vinsgr2vr_d(v0426, hi0426, 1);
        int64_t lo1537 = float2int8(_v1, _v5);
        int64_t hi1537 = float2int8(_v3, _v7);
        __m128i v1537 = __lsx_vinsgr2vr_d(__lsx_vreplgr2vr_d(0), lo1537, 0);
        v1537 = __lsx_vinsgr2vr_d(v1537, hi1537, 1);
        __m128i v0145 = __lsx_vilvl_b(v1537, v0426);
        __m128i v2367 = __lsx_vilvh_b(v1537, v0426);
        __m128i v0123 = __lsx_vilvl_h(v2367, v0145);
        __m128i v4567 = __lsx_vilvh_h(v2367, v0145);
        __m128i v01 = __lsx_vilvl_w(v4567, v0123);
        __m128i v23 = __lsx_vilvh_w(v4567, v0123);
        *(int64_t*)s8ptr0 = __lsx_vpickve2gr_d(v01, 0);
        *(int64_t*)s8ptr1 = __lsx_vpickve2gr_d(v01, 1);
        *(int64_t*)s8ptr2 = __lsx_vpickve2gr_d(v23, 0);
        *(int64_t*)s8ptr3 = __lsx_vpickve2gr_d(v23, 1);
        ptr += 32;
        s8ptr0 += 8;
        s8ptr1 += 8;
        s8ptr2 += 8;
        s8ptr3 += 8;
    }
    for (; i < elemcount; i++)
    {
        __m128 _v = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
        _v = __lsx_vfmul_s(_v, _scale);
        int64_t v = float2int8(_v, _v);
        s8ptr0[0] = (v >> 32) & 0xff;
        s8ptr1[0] = (v >> 40) & 0xff;
        s8ptr2[0] = (v >> 48) & 0xff;
        s8ptr3[0] = (v >> 56) & 0xff;
        ptr += 4;
        s8ptr0 += 1;
        s8ptr1 += 1;
        s8ptr2 += 1;
        s8ptr3 += 1;
    }
}
#endif // __loongarch_sx

static int quantize_forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_data, int scale_data_size, const Option& opt)
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        int out_elempack = 1;
#if __loongarch_sx
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

            quantize_bf16(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        int out_elempack = 1;
#if __loongarch_sx
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

#if __loongarch_sx
#if !__loongarch_asx
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* ptr0 = bottom_blob.row<const unsigned short>(i * 2);
                const unsigned short* ptr1 = bottom_blob.row<const unsigned short>(i * 2 + 1);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * out_elempack, out_elempack) : scale_data;

                quantize_bf16_pack4to8(ptr0, ptr1, s8ptr, scale_data_i, w);
            }
        }
#endif // !__loongarch_asx
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

                quantize_bf16_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_i, w);
            }
        }
#endif // __loongarch_sx
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(i);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_bf16(ptr, s8ptr, scale_data_i, w, elempack);
            }
        }
    }

    if (dims == 3)
    {
        int out_elempack = 1;
#if __loongarch_sx
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

#if __loongarch_sx
#if !__loongarch_asx
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q * 2);
                const unsigned short* ptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;

                quantize_bf16_pack4to8(ptr0, ptr1, s8ptr, scale_data_q, w * h);
            }
        }
#endif // !__loongarch_asx
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

                quantize_bf16_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_q, w * h);
            }
        }
#endif // __loongarch_sx
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_bf16(ptr, s8ptr, scale_data_q, w * h, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
