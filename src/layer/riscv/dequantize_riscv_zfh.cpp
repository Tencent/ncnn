// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dequantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {
#if NCNN_ZFH
#if __riscv_vector
static void dequantize_packnton_f16_fp16s(const int* ptr0, const int* ptr1, __fp16* f16ptr, const Mat& scale_data, const Mat& bias_data, int elemcount)
{
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();
    const size_t vlm2 = __riscv_vsetvlmax_e32m2();
    const size_t vlm4 = __riscv_vsetvlmax_e32m4();
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();

    float scale = scale_data[0];
    vfloat32m2_t _scale0 = __riscv_vfmv_v_f_f32m2(scale, vlm2);
    if (scale_data.w > 1)
    {
        _scale0 = __riscv_vle32_v_f32m2(scale_data, vlm2);
    }
    vfloat32m4_t _scale1 = __riscv_vcreate_v_f32m2_f32m4(_scale0, _scale0);
    vfloat32m8_t _scale2 = __riscv_vcreate_v_f32m4_f32m8(_scale1, _scale1);

    int i = 0;
    if (bias_data.w == 0)
    {
        for (; i + 3 < elemcount; i += 4)
        {
            vint32m1_t _v0 = __riscv_vle32_v_i32m1(ptr0, vlm1);
            vint32m1_t _v1 = __riscv_vle32_v_i32m1(ptr1, vlm1);
            vint32m1_t _v2 = __riscv_vle32_v_i32m1(ptr0 + vlm1, vlm1);
            vint32m1_t _v3 = __riscv_vle32_v_i32m1(ptr1 + vlm1, vlm1);
            vint32m1_t _v4 = __riscv_vle32_v_i32m1(ptr0 + vlm1 * 2, vlm1);
            vint32m1_t _v5 = __riscv_vle32_v_i32m1(ptr1 + vlm1 * 2, vlm1);
            vint32m1_t _v6 = __riscv_vle32_v_i32m1(ptr0 + vlm1 * 3, vlm1);
            vint32m1_t _v7 = __riscv_vle32_v_i32m1(ptr1 + vlm1 * 3, vlm1);
            vint32m8_t _v = __riscv_vcreate_v_i32m1_i32m8(_v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7);
            vfloat32m8_t _vf = __riscv_vfcvt_f_x_v_f32m8(_v, vlm8);
            _vf = __riscv_vfmul_vv_f32m8(_vf, _scale2, vlm8);
            __riscv_vse16_v_f16m4(f16ptr, __riscv_vfncvt_f_f_w_f16m4(_vf, vlm8), vlm8);

            ptr0 += vlm1 * 4;
            ptr1 += vlm1 * 4;
            f16ptr += vlm8;
        }

        for (; i + 1 < elemcount; i += 2)
        {
            vint32m1_t _v0 = __riscv_vle32_v_i32m1(ptr0, vlm1);
            vint32m1_t _v1 = __riscv_vle32_v_i32m1(ptr1, vlm1);
            vint32m1_t _v2 = __riscv_vle32_v_i32m1(ptr0 + vlm1, vlm1);
            vint32m1_t _v3 = __riscv_vle32_v_i32m1(ptr1 + vlm1, vlm1);
            vint32m4_t _v = __riscv_vcreate_v_i32m1_i32m4(_v0, _v1, _v2, _v3);
            vfloat32m4_t _vf = __riscv_vfcvt_f_x_v_f32m4(_v, vlm4);
            _vf = __riscv_vfmul_vv_f32m4(_vf, _scale1, vlm4);
            __riscv_vse16_v_f16m2(f16ptr, __riscv_vfncvt_f_f_w_f16m2(_vf, vlm4), vlm4);

            ptr0 += vlm1 * 2;
            ptr1 += vlm1 * 2;
            f16ptr += vlm4;
        }

        for (; i < elemcount; i++)
        {
            vint32m1_t _v0 = __riscv_vle32_v_i32m1(ptr0, vlm1);
            vint32m1_t _v1 = __riscv_vle32_v_i32m1(ptr1, vlm1);
            vint32m2_t _v = __riscv_vcreate_v_i32m1_i32m2(_v0, _v1);
            vfloat32m2_t _vf = __riscv_vfcvt_f_x_v_f32m2(_v, vlm2);
            _vf = __riscv_vfmul_vv_f32m2(_vf, _scale0, vlm2);
            __riscv_vse16_v_f16m1(f16ptr, __riscv_vfncvt_f_f_w_f16m1(_vf, vlm2), vlm2);

            ptr0 += vlm1;
            ptr1 += vlm1;
            f16ptr += vlm2;
        }
    }
    else
    {
        float bias = bias_data[0];
        vfloat32m2_t _bias0 = __riscv_vfmv_v_f_f32m2(bias, vlm2);
        if (bias_data.w > 1)
        {
            _bias0 = __riscv_vle32_v_f32m2(bias_data, vlm2);
        }
        vfloat32m4_t _bias1 = __riscv_vcreate_v_f32m2_f32m4(_bias0, _bias0);
        vfloat32m8_t _bias2 = __riscv_vcreate_v_f32m4_f32m8(_bias1, _bias1);

        for (; i + 3 < elemcount; i += 4)
        {
            vint32m1_t _v0 = __riscv_vle32_v_i32m1(ptr0, vlm1);
            vint32m1_t _v1 = __riscv_vle32_v_i32m1(ptr1, vlm1);
            vint32m1_t _v2 = __riscv_vle32_v_i32m1(ptr0 + vlm1, vlm1);
            vint32m1_t _v3 = __riscv_vle32_v_i32m1(ptr1 + vlm1, vlm1);
            vint32m1_t _v4 = __riscv_vle32_v_i32m1(ptr0 + vlm1 * 2, vlm1);
            vint32m1_t _v5 = __riscv_vle32_v_i32m1(ptr1 + vlm1 * 2, vlm1);
            vint32m1_t _v6 = __riscv_vle32_v_i32m1(ptr0 + vlm1 * 3, vlm1);
            vint32m1_t _v7 = __riscv_vle32_v_i32m1(ptr1 + vlm1 * 3, vlm1);
            vint32m8_t _v = __riscv_vcreate_v_i32m1_i32m8(_v0, _v1, _v2, _v3, _v4, _v5, _v6, _v7);
            vfloat32m8_t _vf = __riscv_vfcvt_f_x_v_f32m8(_v, vlm8);
            _vf = __riscv_vfmacc_vv_f32m8(_bias2, _vf, _scale2, vlm8);
            __riscv_vse16_v_f16m4(f16ptr, __riscv_vfncvt_f_f_w_f16m4(_vf, vlm8), vlm8);

            ptr0 += vlm1 * 4;
            ptr1 += vlm1 * 4;
            f16ptr += vlm8;
        }

        for (; i + 1 < elemcount; i += 2)
        {
            vint32m1_t _v0 = __riscv_vle32_v_i32m1(ptr0, vlm1);
            vint32m1_t _v1 = __riscv_vle32_v_i32m1(ptr1, vlm1);
            vint32m1_t _v2 = __riscv_vle32_v_i32m1(ptr0 + vlm1, vlm1);
            vint32m1_t _v3 = __riscv_vle32_v_i32m1(ptr1 + vlm1, vlm1);
            vint32m4_t _v = __riscv_vcreate_v_i32m1_i32m4(_v0, _v1, _v2, _v3);
            vfloat32m4_t _vf = __riscv_vfcvt_f_x_v_f32m4(_v, vlm4);
            _vf = __riscv_vfmacc_vv_f32m4(_bias1, _vf, _scale1, vlm4);
            __riscv_vse16_v_f16m2(f16ptr, __riscv_vfncvt_f_f_w_f16m2(_vf, vlm4), vlm4);

            ptr0 += vlm1 * 2;
            ptr1 += vlm1 * 2;
            f16ptr += vlm4;
        }

        for (; i < elemcount; i++)
        {
            vint32m1_t _v0 = __riscv_vle32_v_i32m1(ptr0, vlm1);
            vint32m1_t _v1 = __riscv_vle32_v_i32m1(ptr1, vlm1);
            vint32m2_t _v = __riscv_vcreate_v_i32m1_i32m2(_v0, _v1);
            vfloat32m2_t _vf = __riscv_vfcvt_f_x_v_f32m2(_v, vlm2);
            _vf = __riscv_vfmacc_vv_f32m2(_bias0, _vf, _scale0, vlm2);
            __riscv_vse16_v_f16m1(f16ptr, __riscv_vfncvt_f_f_w_f16m1(_vf, vlm2), vlm2);

            ptr0 += vlm1;
            ptr1 += vlm1;
            f16ptr += vlm2;
        }
    }
}

static void dequantize_packnto1_fp16s(const int* intptr, __fp16* f16ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int stride)
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

    __fp16 tmp[vlm8];
    int n = elemcount * vlm1;
    if (bias_data.w == 0)
    {
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vint32m8_t _v = __riscv_vle32_v_i32m8(intptr, vl);
            vfloat32m8_t _vf = __riscv_vfcvt_f_x_v_f32m8(_v, vl);
            _vf = __riscv_vfmul_vv_f32m8(_vf, _scale, vl);
            __riscv_vse16_v_f16m4(tmp, __riscv_vfncvt_f_f_w_f16m4(_vf, vl), vl);
            for (size_t j = 0; j < (vl / vlm1); j++)
            {
                for (int i = 0; i < vlm1; i++)
                {
                    f16ptr[i * stride] = tmp[j * vlm1 + i];
                }
                f16ptr++;
            }

            intptr += vl;
            n -= vl;
        }
    }
    else
    {
        float bias = bias_data[0];
        vfloat32m8_t _bias = __riscv_vfmv_v_f_f32m8(bias, vlm8);
        if (bias_data.w > 1)
        {
            vfloat32m1_t _b = __riscv_vle32_v_f32m1(bias_data, vlm1);
            _bias = __riscv_vcreate_v_f32m1_f32m8(_b, _b, _b, _b, _b, _b, _b, _b);
        }

        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vint32m8_t _v = __riscv_vle32_v_i32m8(intptr, vl);
            vfloat32m8_t _vf = __riscv_vfcvt_f_x_v_f32m8(_v, vl);
            _vf = __riscv_vfmacc_vv_f32m8(_bias, _vf, _scale, vl);
            __riscv_vse16_v_f16m4(tmp, __riscv_vfncvt_f_f_w_f16m4(_vf, vl), vl);

            for (size_t j = 0; j < (vl / vlm1); j++)
            {
                for (int i = 0; i < vlm1; i++)
                {
                    f16ptr[i * stride] = tmp[j * vlm1 + i];
                }
                f16ptr++;
            }

            intptr += vl;
            n -= vl;
        }
    }
}
#endif // __riscv_vector

static void dequantize_fp16s(const int* intptr, __fp16* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int size = elemcount * elempack;
    float scale = scale_data[0];

#if __riscv_vector
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();
    vfloat32m8_t _scale = __riscv_vfmv_v_f_f32m8(scale, __riscv_vsetvlmax_e32m8());
    if (scale_data.w > 1 && (size_t)elempack == vlm1)
    {
        vfloat32m1_t _s = __riscv_vle32_v_f32m1(scale_data, vlm1);
        _scale = __riscv_vcreate_v_f32m1_f32m8(_s, _s, _s, _s, _s, _s, _s, _s);
    }
#endif // __riscv_vector

    if (bias_data.w == 0)
    {
#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(__riscv_vle32_v_i32m8(intptr, vl), vl);
            _v = __riscv_vfmul_vv_f32m8(_v, _scale, vl);
            __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_v, vl), vl);

            intptr += vl;
            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            *ptr = (__fp16)((float)*intptr * scale);
            intptr++;
            ptr++;
        }
#endif // __riscv_vector
    }
    else
    {
        float bias = bias_data[0];
#if __riscv_vector
        vfloat32m8_t _bias = __riscv_vfmv_v_f_f32m8(bias, __riscv_vsetvlmax_e32m8());
        if (bias_data.w > 1 && (size_t)elempack == vlm1)
        {
            vfloat32m1_t _b = __riscv_vle32_v_f32m1(bias_data, vlm1);
            _bias = __riscv_vcreate_v_f32m1_f32m8(_b, _b, _b, _b, _b, _b, _b, _b);
        }

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(__riscv_vle32_v_i32m8(intptr, vl), vl);
            _v = __riscv_vfmacc_vv_f32m8(_bias, _v, _scale, vl);
            __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_v, vl), vl);

            intptr += vl;
            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            *ptr = (__fp16)((float)*intptr * scale + bias);
            intptr++;
            ptr++;
        }
#endif // __riscv_vector
    }
}

int Dequantize_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const int packn_f16 = csrr_vlenb() / 2;
#endif // __riscv_vector
    if (dims == 1)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = w * elempack % packn_f16 == 0 ? packn_f16 : 1;
        }
#endif
        const int outw = w * elempack / out_elempack;
        size_t out_elemsize = out_elempack * 2u;

        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;
            const int* intptr = (const int*)bottom_blob + i * elempack;
            __fp16* ptr = (__fp16*)top_blob + i * out_elempack;
            const int size = std::min(w - i, wp) * elempack;

            dequantize_fp16s(intptr, ptr, scale_data, bias_data, size, 1);
        }
    }

    if (dims == 2)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = h * elempack % packn_f16 == 0 ? packn_f16 : 1;
        }
#endif
        const int outh = h * elempack / out_elempack;
        size_t out_elemsize = out_elempack * 2u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
#if __riscv_vector
        if (elempack == packn && out_elempack == packn_f16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const int* ptr0 = bottom_blob.row<const int>(i * 2);
                const int* ptr1 = bottom_blob.row<const int>(i * 2 + 1);
                __fp16* f16ptr = top_blob.row<__fp16>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * out_elempack, out_elempack) : scale_data;
                const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * out_elempack, out_elempack) : bias_data;

                dequantize_packnton_f16_fp16s(ptr0, ptr1, f16ptr, scale_data_i, bias_data_i, w);
            }
        }

        if (elempack == packn && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                __fp16* f16ptr = top_blob.row<__fp16>(i * packn);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
                const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;

                dequantize_packnto1_fp16s(intptr, f16ptr, scale_data_i, bias_data_i, w, w);
            }
        }
#endif // __riscv_vector
        if (elempack == 1 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                __fp16* ptr = top_blob.row<__fp16>(i);
                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
                const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;

                dequantize_fp16s(intptr, ptr, scale_data_i, bias_data_i, w, elempack);
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = channels * elempack % packn_f16 == 0 ? packn_f16 : 1;
        }
#endif
        const int outc = channels * elempack / out_elempack;
        size_t out_elemsize = out_elempack * 2u;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
#if __riscv_vector
        if (elempack == packn && out_elempack == packn_f16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const int* ptr0 = bottom_blob.channel(q * 2);
                const int* ptr1 = bottom_blob.channel(q * 2 + 1);
                __fp16* f16ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;
                const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * out_elempack, out_elempack) : bias_data;

                dequantize_packnton_f16_fp16s(ptr0, ptr1, f16ptr, scale_data_q, bias_data_q, w * h * d);
            }
        }

        if (elempack == packn && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                __fp16* f16ptr = top_blob.channel(q * packn);
                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
                const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;

                dequantize_packnto1_fp16s(intptr, f16ptr, scale_data_q, bias_data_q, w * h * d, top_blob.cstep);
            }
        }
#endif // __riscv_vector
        if (elempack == 1 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                __fp16* ptr = top_blob.channel(q);
                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
                const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;

                dequantize_fp16s(intptr, ptr, scale_data_q, bias_data_q, w * h * d, elempack);
            }
        }
    }

    return 0;
}
#endif
} // namespace ncnn
