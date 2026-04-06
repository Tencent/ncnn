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

Quantize_riscv::Quantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector

#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
}

static void quantize(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int size = elemcount * elempack;
    float scale = scale_data[0];

    int i = 0;
#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t _v0 = __riscv_vle32_v_f32m8(ptr, vl);
        _v0 = __riscv_vfmul_vf_f32m8(_v0, scale, vl);
        __riscv_vse8_v_i8m2(s8ptr, float2int8(_v0, vl), vl);

        ptr += vl;
        s8ptr += vl;
        n -= vl;
    }

    i += (size - n);
#endif
    for (; i < size; i++)
    {
        *s8ptr = float2int8(*ptr * scale);
        ptr++;
        s8ptr++;
    }
}

#if __riscv_vector
static void quantize_packnto4n(const float* ptr0, const float* ptr1, const float* ptr2, const float* ptr3, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();
    const size_t vlm4 = __riscv_vsetvlmax_e32m4();
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();

    float scale = scale_data[0];
    vfloat32m4_t _scale0 = __riscv_vfmv_v_f_f32m4(scale, vlm4);
    if (scale_data.w > 1)
    {
        _scale0 = __riscv_vle32_v_f32m4(scale_data, vlm4);
    }
    vfloat32m8_t _scale = __riscv_vcreate_v_f32m4_f32m8(_scale0, _scale0);

    int i = 0;
    for (; i + 1 < elemcount; i += 2)
    {
        vfloat32m1_t _v0 = __riscv_vle32_v_f32m1(ptr0, vlm1);
        vfloat32m1_t _v1 = __riscv_vle32_v_f32m1(ptr1, vlm1);
        vfloat32m1_t _v2 = __riscv_vle32_v_f32m1(ptr2, vlm1);
        vfloat32m1_t _v3 = __riscv_vle32_v_f32m1(ptr3, vlm1);
        vfloat32m1_t _v4 = __riscv_vle32_v_f32m1(ptr0 + vlm1, vlm1);
        vfloat32m1_t _v5 = __riscv_vle32_v_f32m1(ptr1 + vlm1, vlm1);
        vfloat32m1_t _v6 = __riscv_vle32_v_f32m1(ptr2 + vlm1, vlm1);
        vfloat32m1_t _v7 = __riscv_vle32_v_f32m1(ptr3 + vlm1, vlm1);
        vfloat32m8_t _v = __riscv_vcreate_v_f32m1_f32m8(_v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7);
        _v = __riscv_vfmul_vv_f32m8(_v, _scale, vlm8);
        __riscv_vse8_v_i8m2(s8ptr, float2int8(_v, vlm8), vlm8);
        ptr0 += vlm1 * 2;
        ptr1 += vlm1 * 2;
        ptr2 += vlm1 * 2;
        ptr3 += vlm1 * 2;
        s8ptr += vlm8;
    }

    for (; i < elemcount; i++)
    {
        vfloat32m1_t _v0 = __riscv_vle32_v_f32m1(ptr0, vlm1);
        vfloat32m1_t _v1 = __riscv_vle32_v_f32m1(ptr1, vlm1);
        vfloat32m1_t _v2 = __riscv_vle32_v_f32m1(ptr2, vlm1);
        vfloat32m1_t _v3 = __riscv_vle32_v_f32m1(ptr3, vlm1);
        vfloat32m4_t _v = __riscv_vcreate_v_f32m1_f32m4(_v0, _v1, _v2, _v3);
        _v = __riscv_vfmul_vv_f32m4(_v, _scale0, vlm4);
        __riscv_vse8_v_i8m1(s8ptr, float2int8(_v, vlm4), vlm4);
        ptr0 += vlm1;
        ptr1 += vlm1;
        ptr2 += vlm1;
        ptr3 += vlm1;
        s8ptr += vlm4;
    }
}

static void quantize_packnto1(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int stride)
{
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();

    float scale = scale_data[0];
    vfloat32m8_t _scale = __riscv_vfmv_v_f_f32m8(scale, vlm8);
    if (scale_data.w > 1)
    {
        vfloat32m1_t _s = __riscv_vle32_v_f32m1(scale_data, vlm1);
        _scale = __riscv_vcreate_v_f32m1_f32m8(_s, _s, _s, _s, _s, _s, _s, _s);
    }

    signed char tmp[vlm8];
    int n = elemcount * vlm1;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v32 = __riscv_vle32_v_f32m8(ptr, vl);
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

int Quantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage && bottom_blob.elembits() == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const int packn_s8 = csrr_vlenb();
#endif
    if (dims == 1)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = w * elempack % packn_s8 == 0 ? packn_s8 : 1;
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
            const int size = std::min(w - i, wp) * elempack;

            quantize(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = h * elempack % packn_s8 == 0 ? packn_s8 : 1;
        }
#endif // __riscv_vector
        const int outh = h * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
#if __riscv_vector
        if (elempack == packn && out_elempack == packn_s8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* ptr0 = bottom_blob.row(i * 4);
                const float* ptr1 = bottom_blob.row(i * 4 + 1);
                const float* ptr2 = bottom_blob.row(i * 4 + 2);
                const float* ptr3 = bottom_blob.row(i * 4 + 3);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * out_elempack, out_elempack) : scale_data;
                quantize_packnto4n(ptr0, ptr1, ptr2, ptr3, s8ptr, scale_data_i, w);
            }
        }

        if (elempack == packn && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr = top_blob.row<signed char>(i * packn);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
                quantize_packnto1(ptr, s8ptr, scale_data_i, w, w);
            }
        }
#endif // __riscv_vector
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
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = channels * elempack % packn_s8 == 0 ? packn_s8 : 1;
        }
#endif
        const int outc = channels * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        if (elempack == packn && out_elempack == packn_s8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* ptr0 = bottom_blob.channel(q * 4);
                const float* ptr1 = bottom_blob.channel(q * 4 + 1);
                const float* ptr2 = bottom_blob.channel(q * 4 + 2);
                const float* ptr3 = bottom_blob.channel(q * 4 + 3);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;

                quantize_packnto4n(ptr0, ptr1, ptr2, ptr3, s8ptr, scale_data_q, w * h);
            }
        }
        if (elempack == packn && out_elempack == 1)
        {
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q * packn);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_packnto1(ptr, s8ptr, scale_data_q, w * h, top_blob.cstep);
            }
        }
#endif // __riscv_vector
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
} // namespace ncnn