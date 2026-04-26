// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "requantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

Requantize_riscv::Requantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static void requantize_leakyrelu(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, float slope, int elemcount, int elempack)
{
    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    // int8(relu(v * scale_in) * scale_out)
    // int8_relu(v * (scale_in * scale_out))
    // int8(relu(v * scale_in + bias) * scale_out)
    // int8_relu(v * (scale_in * scale_out) + (bias * scale_out))

    // int8(leakyrelu(v * scale_in, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out), slope)
    // int8(leakyrelu(v * scale_in + bias, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out) + (bias * scale_out), slope)

#if __riscv_vector
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();
    const size_t vlmax = __riscv_vsetvlmax_e32m8();

    vfloat32m8_t _scale = __riscv_vfmv_v_f_f32m8(scale_in_data[0], vlmax);
    if (scale_in_data_size > 1)
    {
        // if (elempack == vlm1)
        {
            vfloat32m1_t _s = __riscv_vle32_v_f32m1((const float*)scale_in_data, vlm1);
            _scale = __riscv_vcreate_v_f32m1_f32m8(_s, _s, _s, _s, _s, _s, _s, _s);
        }
    }

    vfloat32m8_t _bias = __riscv_vfmv_v_f_f32m8(0.f, vlmax);
    if (bias_data_size == 1)
    {
        _bias = __riscv_vfmv_v_f_f32m8(bias_data[0], vlmax);
    }
    else if (bias_data_size > 1)
    {
        // if (elempack == vlm1)
        {
            vfloat32m1_t _b = __riscv_vle32_v_f32m1((const float*)bias_data, vlm1);
            _bias = __riscv_vcreate_v_f32m1_f32m8(_b, _b, _b, _b, _b, _b, _b, _b);
        }
    }

    if (scale_out_data_size > 1)
    {
        // if (elempack == vlm1)
        {
            vfloat32m1_t _s = __riscv_vle32_v_f32m1((const float*)scale_out_data, vlm1);
            vfloat32m8_t _s2 = __riscv_vcreate_v_f32m1_f32m8(_s, _s, _s, _s, _s, _s, _s, _s);
            _scale = __riscv_vfmul_vv_f32m8(_scale, _s2, vlmax);
            _bias = __riscv_vfmul_vv_f32m8(_bias, _s2, vlmax);
        }
    }
    else
    {
        _scale = __riscv_vfmul_vf_f32m8(_scale, scale_out_data[0], vlmax);
        _bias = __riscv_vfmul_vf_f32m8(_bias, scale_out_data[0], vlmax);
    }

    int n = size;
    if (slope > 0.f) // Leaky ReLU
    {
        if (bias_data_size == 0)
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vl);
                vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vl);
                _v = __riscv_vfmul_vv_f32m8(_v, _scale, vl);
                __riscv_vse8_v_i8m2(ptr, float2int8leakyrelu(_v, slope, vl), vl);

                intptr += vl;
                ptr += vl;
                n -= vl;
            }
        }
        else
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vl);
                vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vl);
                _v = __riscv_vfmadd_vv_f32m8(_v, _scale, _bias, vl);
                __riscv_vse8_v_i8m2(ptr, float2int8leakyrelu(_v, slope, vl), vl);

                intptr += vl;
                ptr += vl;
                n -= vl;
            }
        }
    }
    else
    {
        if (bias_data_size == 0)
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vl);
                vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vl);
                _v = __riscv_vfmul_vv_f32m8(_v, _scale, vl);
                __riscv_vse8_v_i8m2(ptr, float2int8relu(_v, vl), vl);

                intptr += vl;
                ptr += vl;
                n -= vl;
            }
        }
        else
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vl);
                vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vl);
                _v = __riscv_vfmadd_vv_f32m8(_v, _scale, _bias, vl);
                __riscv_vse8_v_i8m2(ptr, float2int8relu(_v, vl), vl);

                intptr += vl;
                ptr += vl;
                n -= vl;
            }
        }
    }
#else  // __riscv_vector
    float scale = scale_in_data[0] * scale_out_data[0];
    if (slope > 0.f)
    {
        if (bias_data_size == 0)
        {
            for (int i = 0; i < size; i++)
            {
                float v = *intptr * scale;
                *ptr = (v < 0) ? float2int8(v * slope) : float2int8(v);
                intptr++;
                ptr++;
            }
        }
        else
        {
            float bias = bias_data[0] * scale_out_data[0];
            for (int i = 0; i < size; i++)
            {
                float v = *intptr * scale + bias;
                *ptr = (v < 0) ? float2int8(v * slope) : float2int8(v);
                intptr++;
                ptr++;
            }
        }
    }
    else
    {
        if (bias_data_size == 0)
        {
            for (int i = 0; i < size; i++)
            {
                float v = *intptr * scale;
                *ptr = (v < 0) ? 0 : float2int8(v);
                intptr++;
                ptr++;
            }
        }
        else
        {
            float bias = bias_data[0] * scale_out_data[0];
            for (int i = 0; i < size; i++)
            {
                float v = *intptr * scale + bias;
                *ptr = (v < 0) ? 0 : float2int8(v);
                intptr++;
                ptr++;
            }
        }
    }
#endif // __riscv_vector
}

static void requantize(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack)
{
    if ((activation_type == 1) || (activation_type == 2))
    {
        const float slope = activation_params[0];
        requantize_leakyrelu(intptr, ptr, scale_in_data, bias_data, scale_out_data, slope, elemcount, elempack);
        return;
    }

    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    const float scale_in = scale_in_data[0];
    const float scale_out = scale_out_data[0];
    const float bias = bias_data_size == 0 ? 0.f : bias_data[0];

#if __riscv_vector
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();
    const size_t vlmax = __riscv_vsetvlmax_e32m8();

    vfloat32m8_t _scale_in = __riscv_vfmv_v_f_f32m8(scale_in, vlmax);
    if (scale_in_data_size > 1)
    {
        // if (elempack == vlm1)
        {
            vfloat32m1_t _s = __riscv_vle32_v_f32m1((const float*)scale_in_data, vlm1);
            _scale_in = __riscv_vcreate_v_f32m1_f32m8(_s, _s, _s, _s, _s, _s, _s, _s);
        }
    }

    vfloat32m8_t _scale_out = __riscv_vfmv_v_f_f32m8(scale_out, vlmax);
    if (scale_out_data_size > 1)
    {
        // if (elempack == vlm1)
        {
            vfloat32m1_t _s = __riscv_vle32_v_f32m1((const float*)scale_out_data, vlm1);
            _scale_out = __riscv_vcreate_v_f32m1_f32m8(_s, _s, _s, _s, _s, _s, _s, _s);
        }
    }

    int n = size;
    if (bias_data_size == 0)
    {
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vl);
            vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vl);
            _v = __riscv_vfmul_vv_f32m8(_v, _scale_in, vl);
            _v = activation_ps(_v, activation_type, activation_params, vl);
            _v = __riscv_vfmul_vv_f32m8(_v, _scale_out, vl);
            __riscv_vse8_v_i8m2(ptr, float2int8(_v, vl), vl);

            intptr += vl;
            ptr += vl;
            n -= vl;
        }
    }
    else // if (bias_data_size >= 1)
    {
        vfloat32m8_t _bias = __riscv_vfmv_v_f_f32m8(bias, vlmax);
        if (bias_data_size > 1)
        {
            // if (elempack == vlm1)
            {
                vfloat32m1_t _b = __riscv_vle32_v_f32m1((const float*)bias_data, vlm1);
                _bias = __riscv_vcreate_v_f32m1_f32m8(_b, _b, _b, _b, _b, _b, _b, _b);
            }
        }

        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vl);
            vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vl);
            _v = __riscv_vfmadd_vv_f32m8(_v, _scale_in, _bias, vl); // add bias
            _v = activation_ps(_v, activation_type, activation_params, vl);
            _v = __riscv_vfmul_vv_f32m8(_v, _scale_out, vl);
            __riscv_vse8_v_i8m2(ptr, float2int8(_v, vl), vl);

            intptr += vl;
            ptr += vl;
            n -= vl;
        }
    }
#else  // __riscv_vector
    if (bias_data_size == 0)
    {
        for (int i = 0; i < size; i++)
        {
            float v = (float)(*intptr) * scale_in;
            v = activation_ss(v, activation_type, activation_params);
            *ptr = float2int8(v * scale_out);
            intptr++;
            ptr++;
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            float v = (float)(*intptr) * scale_in + bias;
            v = activation_ss(v, activation_type, activation_params);
            *ptr = float2int8(v * scale_out);
            intptr++;
            ptr++;
        }
    }
#endif // __riscv_vector
}

int Requantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const size_t out_elemsize = elempack * 1u;

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const int* intptr = (const int*)bottom_blob + i * elempack;
            signed char* ptr = (signed char*)top_blob + i * elempack;
            const int size = std::min(w - i, wp) * elempack;

            requantize(intptr, ptr, scale_in_data, bias_data, scale_out_data, activation_type, activation_params, size, 1);
        }
    }

    if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const int* intptr = bottom_blob.row<const int>(i);
            signed char* ptr = top_blob.row<signed char>(i);

            const Mat scale_in_data_i = scale_in_data_size > 1 ? scale_in_data.range(i * elempack, elempack) : scale_in_data;
            const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;
            const Mat scale_out_data_i = scale_out_data_size > 1 ? scale_out_data.range(i * elempack, elempack) : scale_out_data;

            requantize(intptr, ptr, scale_in_data_i, bias_data_i, scale_out_data_i, activation_type, activation_params, w, elempack);
        }
    }

    if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            signed char* ptr = top_blob.channel(q);

            const Mat scale_in_data_q = scale_in_data_size > 1 ? scale_in_data.range(q * elempack, elempack) : scale_in_data;
            const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;
            const Mat scale_out_data_q = scale_out_data_size > 1 ? scale_out_data.range(q * elempack, elempack) : scale_out_data;

            requantize(intptr, ptr, scale_in_data_q, bias_data_q, scale_out_data_q, activation_type, activation_params, w * h, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
