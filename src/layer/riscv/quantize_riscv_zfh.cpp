// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "quantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector
#include "riscv_activation.h"
#include "riscv_usability.h"

#include "cpu.h"

namespace ncnn {
#if NCNN_ZFH
static void quantize_fp16s(const __fp16* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;
    float scale = scale_data[0];

    int i = 0;
#if __riscv_vector
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();
    vfloat32m8_t _scale;
    if (scale_data_size == 1)
    {
        _scale = __riscv_vfmv_v_f_f32m8(scale, __riscv_vsetvlmax_e32m8());
    }
    else if (elempack == vlm1)
    {
        vfloat32m1_t _s = __riscv_vle32_v_f32m1(scale_data, vlm1);
        _scale = __riscv_vcreate_v_f32m1_f32m8(_s, _s, _s, _s, _s, _s, _s, _s);
    }

    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m4(n);
        vfloat32m8_t _v0 = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
        _v0 = __riscv_vfmul_vv_f32m8(_v0, _scale, vl);
        __riscv_vse8_v_i8m2(s8ptr, float2int8(_v0, vl), vl);

        ptr += vl;
        s8ptr += vl;
        n -= vl;
    }

    i += (size - n);
#endif
    for (; i < size; i++)
    {
        float v = (float)(*ptr) * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

#if __riscv_vector
static void quantize_packnto2n_fp16s(const __fp16* ptr0, const __fp16* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const size_t vlm1 = __riscv_vsetvlmax_e16m1();
    const size_t vlm2 = __riscv_vsetvlmax_e16m2();
    const size_t vlm4 = __riscv_vsetvlmax_e16m4();

    float scale = scale_data[0];
    vfloat32m4_t _scale0 = __riscv_vfmv_v_f_f32m4(scale, __riscv_vsetvlmax_e32m4());
    if (scale_data.w > 1)
    {
        _scale0 = __riscv_vle32_v_f32m4(scale_data, __riscv_vsetvlmax_e32m4());
    }
    vfloat32m8_t _scale = __riscv_vcreate_v_f32m4_f32m8(_scale0, _scale0);

    int i = 0;
    for (; i + 1 < elemcount; i += 2)
    {
        vfloat16m1_t _v0 = __riscv_vle16_v_f16m1(ptr0, vlm1);
        vfloat16m1_t _v1 = __riscv_vle16_v_f16m1(ptr1, vlm1);
        vfloat16m1_t _v2 = __riscv_vle16_v_f16m1(ptr0 + vlm1, vlm1);
        vfloat16m1_t _v3 = __riscv_vle16_v_f16m1(ptr1 + vlm1, vlm1);
        vfloat16m4_t _v4 = __riscv_vcreate_v_f16m1_f16m4(_v0, _v1, _v2, _v3);
        vfloat32m8_t _v = __riscv_vfwcvt_f_f_v_f32m8(_v4, vlm4);
        _v = __riscv_vfmul_vv_f32m8(_v, _scale, vlm4);
        __riscv_vse8_v_i8m2(s8ptr, float2int8(_v, vlm4), vlm4);

        ptr0 += vlm1 * 2;
        ptr1 += vlm1 * 2;
        s8ptr += vlm4;
    }

    for (; i < elemcount; i++)
    {
        vfloat16m1_t _v0 = __riscv_vle16_v_f16m1(ptr0, vlm1);
        vfloat16m1_t _v1 = __riscv_vle16_v_f16m1(ptr1, vlm1);
        vfloat32m4_t _v = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vcreate_v_f16m1_f16m2(_v0, _v1), vlm2);
        _v = __riscv_vfmul_vv_f32m4(_v, _scale0, vlm2);
        __riscv_vse8_v_i8m1(s8ptr, float2int8(_v, vlm2), vlm2);

        ptr0 += vlm1;
        ptr1 += vlm1;
        s8ptr += vlm2;
    }
}

static void quantize_packnto1_fp16s(const __fp16* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int stride)
{
    const size_t vlm4 = __riscv_vsetvlmax_e16m4();
    const size_t vlm1 = __riscv_vsetvlmax_e16m1();

    float scale = scale_data[0];
    vfloat32m8_t _scale = __riscv_vfmv_v_f_f32m8(scale, __riscv_vsetvlmax_e32m8());
    if (scale_data.w > 1)
    {
        vfloat32m1_t _s = __riscv_vle32_v_f32m1(scale_data, __riscv_vsetvlmax_e32m1());
        _scale = __riscv_vcreate_v_f32m1_f32m8(_s, _s, _s, _s, _s, _s, _s, _s);
    }

    signed char tmp[vlm4];
    int n = elemcount * vlm1;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m4(n);
        vfloat32m8_t v32 = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);
        v32 = __riscv_vfmul_vv_f32m8(v32, _scale, vl);
        __riscv_vse8_v_i8m2(tmp, float2int8(v32, vl), vl);
        for (size_t j = 0; j < (vl / vlm1); j++)
        {
            for (int i = 0; i < vlm1; i++)
            {
                s8ptr[i * stride] = tmp[j * vlm1 + i];
            }
            s8ptr++;
        }

        ptr += vl;
        n -= vl;
    }
}
#endif // __riscv_vector

int Quantize_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const int pack2n = csrr_vlenb();
#endif

    if (dims == 1)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = w * elempack % pack2n == 0 ? pack2n : 1;
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

            const __fp16* ptr = (const __fp16*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;
            const int size = std::min(w - i, wp) * elempack;

            quantize_fp16s(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = h * elempack % pack2n == 0 ? pack2n : 1;
        }
#endif
        const int outh = h * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        if (elempack == packn && out_elempack == pack2n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const __fp16* ptr0 = bottom_blob.row<const __fp16>(i * 2);
                const __fp16* ptr1 = bottom_blob.row<const __fp16>(i * 2 + 1);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * out_elempack, out_elempack) : scale_data;

                quantize_packnto2n_fp16s(ptr0, ptr1, s8ptr, scale_data_i, w);
            }
        }
        if (elempack == packn && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(i);
                signed char* s8ptr = top_blob.row<signed char>(i * packn);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_packnto1_fp16s(ptr, s8ptr, scale_data_i, w, w);
            }
        }
#endif
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(i);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_fp16s(ptr, s8ptr, scale_data_i, w, elempack);
            }
        }
    }

    if (dims == 3)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = channels * elempack % pack2n == 0 ? pack2n : 1;
        }
#endif
        const int outc = channels * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
#if __riscv_vector
        if (elempack == packn && out_elempack == pack2n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const __fp16* ptr0 = bottom_blob.channel(q * 2);
                const __fp16* ptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;

                quantize_packnto2n_fp16s(ptr0, ptr1, s8ptr, scale_data_q, w * h);
            }
        }
        if (elempack == packn && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q * packn);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_packnto1_fp16s(ptr, s8ptr, scale_data_q, w * h, top_blob.cstep);
            }
        }
#endif
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_fp16s(ptr, s8ptr, scale_data_q, w * h, elempack);
            }
        }
    }

    return 0;
}

static void quantize_fp16sa(const __fp16* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;
    __fp16 scale = (__fp16)scale_data[0];

    int i = 0;
#if __riscv_zvfh
    const size_t vlm1 = __riscv_vsetvlmax_e16m1();
    vfloat16m8_t _scale;
    if (scale_data_size == 1)
    {
        _scale = __riscv_vfmv_v_f_f16m8(scale, __riscv_vsetvlmax_e16m8());
    }
    else if (elempack == vlm1)
    {
        vfloat32m1_t _s32 = __riscv_vle32_v_f32m1(scale_data, __riscv_vsetvlmax_e32m1());
        vfloat16m1_t _s16 = __riscv_vfncvt_f_f_w_f16m1(__riscv_vcreate_v_f32m1_f32m2(_s32, _s32), vlm1);
        _scale = __riscv_vcreate_v_f16m1_f16m8(_s16, _s16, _s16, _s16, _s16, _s16, _s16, _s16);
    }

    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m8(n);
        vfloat16m8_t _v0 = __riscv_vle16_v_f16m8(ptr, vl);
        _v0 = __riscv_vfmul_vv_f16m8(_v0, _scale, vl);
        __riscv_vse8_v_i8m4(s8ptr, float2int8(_v0, vl), vl);

        ptr += vl;
        s8ptr += vl;
        n -= vl;
    }

    i += (size - n);
#endif // __riscv_zvfh
    for (; i < size; i++)
    {
        __fp16 v = *ptr * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

#if __riscv_zvfh
static void quantize_packnto2n_fp16sa(const __fp16* ptr0, const __fp16* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const size_t vlm1 = __riscv_vsetvlmax_e16m1();
    const size_t vlm2 = __riscv_vsetvlmax_e16m2();
    const size_t vlm4 = __riscv_vsetvlmax_e16m4();
    const size_t vlm8 = __riscv_vsetvlmax_e16m8();

    __fp16 scale = (__fp16)scale_data[0];
    vfloat16m2_t _scale0 = __riscv_vfmv_v_f_f16m2(scale, __riscv_vsetvlmax_e16m2());
    if (scale_data.w > 1)
    {
        vfloat32m4_t _s32 = __riscv_vle32_v_f32m4(scale_data, __riscv_vsetvlmax_e32m4());
        _scale0 = __riscv_vfncvt_f_f_w_f16m2(_s32, vlm2);
    }
    vfloat16m4_t _scale1 = __riscv_vcreate_v_f16m2_f16m4(_scale0, _scale0);
    vfloat16m8_t _scale2 = __riscv_vcreate_v_f16m4_f16m8(_scale1, _scale1);

    int i = 0;
    for (; i + 3 < elemcount; i += 4)
    {
        vfloat16m1_t _v0 = __riscv_vle16_v_f16m1(ptr0, vlm1);
        vfloat16m1_t _v1 = __riscv_vle16_v_f16m1(ptr1, vlm1);
        vfloat16m1_t _v2 = __riscv_vle16_v_f16m1(ptr0 + vlm1, vlm1);
        vfloat16m1_t _v3 = __riscv_vle16_v_f16m1(ptr1 + vlm1, vlm1);
        vfloat16m1_t _v4 = __riscv_vle16_v_f16m1(ptr0 + vlm1 * 2, vlm1);
        vfloat16m1_t _v5 = __riscv_vle16_v_f16m1(ptr1 + vlm1 * 2, vlm1);
        vfloat16m1_t _v6 = __riscv_vle16_v_f16m1(ptr0 + vlm1 * 3, vlm1);
        vfloat16m1_t _v7 = __riscv_vle16_v_f16m1(ptr1 + vlm1 * 3, vlm1);
        vfloat16m8_t _v = __riscv_vcreate_v_f16m1_f16m8(_v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7);
        _v = __riscv_vfmul_vv_f16m8(_v, _scale2, vlm8);
        __riscv_vse8_v_i8m4(s8ptr, float2int8(_v, vlm8), vlm8);

        ptr0 += vlm1 * 4;
        ptr1 += vlm1 * 4;
        s8ptr += vlm8;
    }

    for (; i + 1 < elemcount; i += 2)
    {
        vfloat16m1_t _v0 = __riscv_vle16_v_f16m1(ptr0, vlm1);
        vfloat16m1_t _v1 = __riscv_vle16_v_f16m1(ptr1, vlm1);
        vfloat16m1_t _v2 = __riscv_vle16_v_f16m1(ptr0 + vlm1, vlm1);
        vfloat16m1_t _v3 = __riscv_vle16_v_f16m1(ptr1 + vlm1, vlm1);
        vfloat16m4_t _v = __riscv_vcreate_v_f16m1_f16m4(_v0, _v1, _v2, _v3);
        _v = __riscv_vfmul_vv_f16m4(_v, _scale1, vlm4);
        __riscv_vse8_v_i8m2(s8ptr, float2int8(_v, vlm4), vlm4);

        ptr0 += vlm1 * 2;
        ptr1 += vlm1 * 2;
        s8ptr += vlm4;
    }

    for (; i < elemcount; i++)
    {
        vfloat16m1_t _v0 = __riscv_vle16_v_f16m1(ptr0, vlm1);
        vfloat16m1_t _v1 = __riscv_vle16_v_f16m1(ptr1, vlm1);
        vfloat16m2_t _v = __riscv_vcreate_v_f16m1_f16m2(_v0, _v1);
        _v = __riscv_vfmul_vv_f16m2(_v, _scale0, vlm2);
        __riscv_vse8_v_i8m1(s8ptr, float2int8(_v, vlm2), vlm2);

        ptr0 += vlm1;
        ptr1 += vlm1;
        s8ptr += vlm2;
    }
}

static void quantize_packnto1_fp16sa(const __fp16* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int stride)
{
    const size_t vlm8 = __riscv_vsetvlmax_e16m8();
    const size_t vlm1 = __riscv_vsetvlmax_e16m1();

    __fp16 scale = (__fp16)scale_data[0];
    vfloat16m8_t _scale = __riscv_vfmv_v_f_f16m8(scale, __riscv_vsetvlmax_e16m8());
    if (scale_data.w > 1)
    {
        vfloat32m1_t _s32 = __riscv_vle32_v_f32m1(scale_data, __riscv_vsetvlmax_e32m1());
        vfloat16m1_t _s16 = __riscv_vfncvt_f_f_w_f16m1(__riscv_vcreate_v_f32m1_f32m2(_s32, _s32), vlm1);
        _scale = __riscv_vcreate_v_f16m1_f16m8(_s16, _s16, _s16, _s16, _s16, _s16, _s16, _s16);
    }

    signed char tmp[vlm8];
    int n = elemcount * vlm1;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m8(n);
        vfloat16m8_t v = __riscv_vle16_v_f16m8(ptr, vl);
        v = __riscv_vfmul_vv_f16m8(v, _scale, vl);
        __riscv_vse8_v_i8m4(tmp, float2int8(v, vl), vl);
        for (size_t j = 0; j < (vl / vlm1); j++)
        {
            for (int i = 0; i < vlm1; i++)
            {
                s8ptr[i * stride] = tmp[j * vlm1 + i];
            }
            s8ptr++;
        }

        ptr += vl;
        n -= vl;
    }
}
#endif // __riscv_zvfh

int Quantize_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    const int pack2n = csrr_vlenb();
#endif

    if (dims == 1)
    {
        int out_elempack = 1;
#if __riscv_zvfh
        if (opt.use_packing_layout)
        {
            out_elempack = w * elempack % pack2n == 0 ? pack2n : 1;
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

            const __fp16* ptr = (const __fp16*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;
            const int size = std::min(w - i, wp) * elempack;

            quantize_fp16sa(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        int out_elempack = 1;
#if __riscv_zvfh
        if (opt.use_packing_layout)
        {
            out_elempack = h * elempack % pack2n == 0 ? pack2n : 1;
        }
#endif
        const int outh = h * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
#if __riscv_zvfh
        if (elempack == packn && out_elempack == pack2n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const __fp16* ptr0 = bottom_blob.row<const __fp16>(i * 2);
                const __fp16* ptr1 = bottom_blob.row<const __fp16>(i * 2 + 1);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * out_elempack, out_elempack) : scale_data;

                quantize_packnto2n_fp16sa(ptr0, ptr1, s8ptr, scale_data_i, w);
            }
        }

        if (elempack == packn && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(i);
                signed char* s8ptr = top_blob.row<signed char>(i * packn);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_packnto1_fp16sa(ptr, s8ptr, scale_data_i, w, w);
            }
        }
#endif
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(i);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_fp16sa(ptr, s8ptr, scale_data_i, w, elempack);
            }
        }
    }

    if (dims == 3)
    {
        int out_elempack = 1;
#if __riscv_zvfh
        if (opt.use_packing_layout)
        {
            out_elempack = channels * elempack % pack2n == 0 ? pack2n : 1;
        }
#endif
        const int outc = channels * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_zvfh
        if (elempack == packn && out_elempack == pack2n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const __fp16* ptr0 = bottom_blob.channel(q * 2);
                const __fp16* ptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;

                quantize_packnto2n_fp16sa(ptr0, ptr1, s8ptr, scale_data_q, w * h);
            }
        }

        if (elempack == packn && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q * packn);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_packnto1_fp16sa(ptr, s8ptr, scale_data_q, w * h, top_blob.cstep);
            }
        }
#endif
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_fp16sa(ptr, s8ptr, scale_data_q, w * h, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_ZFH
} // namespace ncnn