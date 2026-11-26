// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "requantize_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include "fused_activation.h"

#include <algorithm>

namespace ncnn {

Requantize_powerpc::Requantize_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void requantize_relu(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int elemcount, int elempack)
{
    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    // int8(relu(v * scale_in + bias) * scale_out)
    // int8_relu(v * (scale_in * scale_out) + (bias * scale_out))

    const float scale_in0 = scale_in_data[0];
    const float scale_out0 = scale_out_data[0];

    int i = 0;
#if __ALTIVEC__
    const bool vec_ok = (scale_in_data_size == 1 || elempack == 4 || elempack == 8)
                        && (bias_data_size == 0 || bias_data_size == 1 || elempack == 4 || elempack == 8)
                        && (scale_out_data_size == 1 || elempack == 4 || elempack == 8);

    v4f32 _scale_in0 = vec_splats(scale_in0);
    v4f32 _scale_in1 = vec_splats(scale_in0);
    if (scale_in_data_size > 1 && elempack == 4)
    {
        _scale_in0 = __ncnn_altivec_load(scale_in_data);
        _scale_in1 = _scale_in0;
    }
    if (scale_in_data_size > 1 && elempack == 8)
    {
        _scale_in0 = __ncnn_altivec_load(scale_in_data);
        _scale_in1 = __ncnn_altivec_load((const float*)scale_in_data + 4);
    }

    v4f32 _scale_out0 = vec_splats(scale_out0);
    v4f32 _scale_out1 = vec_splats(scale_out0);
    if (scale_out_data_size > 1 && elempack == 4)
    {
        _scale_out0 = __ncnn_altivec_load(scale_out_data);
        _scale_out1 = _scale_out0;
    }
    if (scale_out_data_size > 1 && elempack == 8)
    {
        _scale_out0 = __ncnn_altivec_load(scale_out_data);
        _scale_out1 = __ncnn_altivec_load((const float*)scale_out_data + 4);
    }

    v4f32 _scale0 = __ncnn_altivec_mul(_scale_in0, _scale_out0);
    v4f32 _scale1 = __ncnn_altivec_mul(_scale_in1, _scale_out1);

    if (vec_ok && bias_data_size == 0)
    {
        for (; i + 7 < size; i += 8)
        {
            v4f32 _v0 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            v4f32 _v1 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr + 4));
            _v0 = __ncnn_altivec_mul(_v0, _scale0);
            _v1 = __ncnn_altivec_mul(_v1, _scale1);
            _v0 = vec_max(_v0, vec_splats(0.f));
            _v1 = vec_max(_v1, vec_splats(0.f));
            v16i8 _b0 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v0), 0, 127);
            v16i8 _b1 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v1), 0, 127);
            memcpy(ptr, &_b0, 4);
            memcpy(ptr + 4, &_b1, 4);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            _v = __ncnn_altivec_mul(_v, _scale0);
            _v = vec_max(_v, vec_splats(0.f));
            v16i8 _b = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v), 0, 127);
            memcpy(ptr, &_b, 4);
            intptr += 4;
            ptr += 4;
        }
    }
    else if (vec_ok)
    {
        const float bias0 = bias_data[0];
        v4f32 _bias0 = vec_splats(bias0);
        v4f32 _bias1 = vec_splats(bias0);
        if (bias_data_size > 1 && elempack == 4)
        {
            _bias0 = __ncnn_altivec_load(bias_data);
            _bias1 = _bias0;
        }
        if (bias_data_size > 1 && elempack == 8)
        {
            _bias0 = __ncnn_altivec_load(bias_data);
            _bias1 = __ncnn_altivec_load((const float*)bias_data + 4);
        }

        _bias0 = __ncnn_altivec_mul(_bias0, _scale_out0);
        _bias1 = __ncnn_altivec_mul(_bias1, _scale_out1);

        for (; i + 7 < size; i += 8)
        {
            v4f32 _v0 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            v4f32 _v1 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr + 4));
            _v0 = __ncnn_altivec_fmadd(_bias0, _v0, _scale0);
            _v1 = __ncnn_altivec_fmadd(_bias1, _v1, _scale1);
            _v0 = vec_max(_v0, vec_splats(0.f));
            _v1 = vec_max(_v1, vec_splats(0.f));
            v16i8 _b0 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v0), 0, 127);
            v16i8 _b1 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v1), 0, 127);
            memcpy(ptr, &_b0, 4);
            memcpy(ptr + 4, &_b1, 4);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            _v = __ncnn_altivec_fmadd(_bias0, _v, _scale0);
            _v = vec_max(_v, vec_splats(0.f));
            v16i8 _b = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v), 0, 127);
            memcpy(ptr, &_b, 4);
            intptr += 4;
            ptr += 4;
        }
    }
#endif // __ALTIVEC__

    const float scale = scale_in0 * scale_out0;
    const float bias = bias_data_size == 0 ? 0.f : bias_data[0] * scale_out0;

    for (; i < size; i++)
    {
        const float scale_i = scale_in_data_size == 1 ? scale : (scale_in_data[i % elempack] * (scale_out_data_size == 1 ? scale_out0 : scale_out_data[i % elempack]));
        const float bias_i = bias_data_size == 0 ? 0.f : (bias_data_size == 1 ? bias : bias_data[i % elempack] * (scale_out_data_size == 1 ? scale_out0 : scale_out_data[i % elempack]));

        float v = *intptr * scale_i + bias_i;
        if (v < 0) v = 0;
        *ptr = float2int8(v);
        intptr++;
        ptr++;
    }
}

static void requantize_leakyrelu(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, float slope, int elemcount, int elempack)
{
    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    // int8(leakyrelu(v * scale_in + bias, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out) + (bias * scale_out), slope)

    const float scale_in0 = scale_in_data[0];
    const float scale_out0 = scale_out_data[0];

    int i = 0;
#if __ALTIVEC__
    const bool vec_ok = (scale_in_data_size == 1 || elempack == 4 || elempack == 8)
                        && (bias_data_size == 0 || bias_data_size == 1 || elempack == 4 || elempack == 8)
                        && (scale_out_data_size == 1 || elempack == 4 || elempack == 8);

    v4f32 _scale_in0 = vec_splats(scale_in0);
    v4f32 _scale_in1 = vec_splats(scale_in0);
    if (scale_in_data_size > 1 && elempack == 4)
    {
        _scale_in0 = __ncnn_altivec_load(scale_in_data);
        _scale_in1 = _scale_in0;
    }
    if (scale_in_data_size > 1 && elempack == 8)
    {
        _scale_in0 = __ncnn_altivec_load(scale_in_data);
        _scale_in1 = __ncnn_altivec_load((const float*)scale_in_data + 4);
    }

    v4f32 _scale_out0 = vec_splats(scale_out0);
    v4f32 _scale_out1 = vec_splats(scale_out0);
    if (scale_out_data_size > 1 && elempack == 4)
    {
        _scale_out0 = __ncnn_altivec_load(scale_out_data);
        _scale_out1 = _scale_out0;
    }
    if (scale_out_data_size > 1 && elempack == 8)
    {
        _scale_out0 = __ncnn_altivec_load(scale_out_data);
        _scale_out1 = __ncnn_altivec_load((const float*)scale_out_data + 4);
    }

    v4f32 _scale0 = __ncnn_altivec_mul(_scale_in0, _scale_out0);
    v4f32 _scale1 = __ncnn_altivec_mul(_scale_in1, _scale_out1);
    v4f32 _slope = vec_splats(slope);

    if (vec_ok && bias_data_size == 0)
    {
        for (; i + 7 < size; i += 8)
        {
            v4f32 _v0 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            v4f32 _v1 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr + 4));
            _v0 = __ncnn_altivec_mul(_v0, _scale0);
            _v1 = __ncnn_altivec_mul(_v1, _scale1);
            _v0 = vec_max(_v0, __ncnn_altivec_mul(_v0, _slope));
            _v1 = vec_max(_v1, __ncnn_altivec_mul(_v1, _slope));
            v16i8 _b0 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v0), -127, 127);
            v16i8 _b1 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v1), -127, 127);
            memcpy(ptr, &_b0, 4);
            memcpy(ptr + 4, &_b1, 4);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            _v = __ncnn_altivec_mul(_v, _scale0);
            _v = vec_max(_v, __ncnn_altivec_mul(_v, _slope));
            v16i8 _b = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v), -127, 127);
            memcpy(ptr, &_b, 4);
            intptr += 4;
            ptr += 4;
        }
    }
    else if (vec_ok)
    {
        const float bias0 = bias_data[0];
        v4f32 _bias0 = vec_splats(bias0);
        v4f32 _bias1 = vec_splats(bias0);
        if (bias_data_size > 1 && elempack == 4)
        {
            _bias0 = __ncnn_altivec_load(bias_data);
            _bias1 = _bias0;
        }
        if (bias_data_size > 1 && elempack == 8)
        {
            _bias0 = __ncnn_altivec_load(bias_data);
            _bias1 = __ncnn_altivec_load((const float*)bias_data + 4);
        }

        _bias0 = __ncnn_altivec_mul(_bias0, _scale_out0);
        _bias1 = __ncnn_altivec_mul(_bias1, _scale_out1);

        for (; i + 7 < size; i += 8)
        {
            v4f32 _v0 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            v4f32 _v1 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr + 4));
            _v0 = __ncnn_altivec_fmadd(_bias0, _v0, _scale0);
            _v1 = __ncnn_altivec_fmadd(_bias1, _v1, _scale1);
            _v0 = vec_max(_v0, __ncnn_altivec_mul(_v0, _slope));
            _v1 = vec_max(_v1, __ncnn_altivec_mul(_v1, _slope));
            v16i8 _b0 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v0), -127, 127);
            v16i8 _b1 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v1), -127, 127);
            memcpy(ptr, &_b0, 4);
            memcpy(ptr + 4, &_b1, 4);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            _v = __ncnn_altivec_fmadd(_bias0, _v, _scale0);
            _v = vec_max(_v, __ncnn_altivec_mul(_v, _slope));
            v16i8 _b = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v), -127, 127);
            memcpy(ptr, &_b, 4);
            intptr += 4;
            ptr += 4;
        }
    }
#endif // __ALTIVEC__

    const float scale = scale_in0 * scale_out0;
    const float bias = bias_data_size == 0 ? 0.f : bias_data[0] * scale_out0;

    for (; i < size; i++)
    {
        const float scale_i = scale_in_data_size == 1 ? scale : (scale_in_data[i % elempack] * (scale_out_data_size == 1 ? scale_out0 : scale_out_data[i % elempack]));
        const float bias_i = bias_data_size == 0 ? 0.f : (bias_data_size == 1 ? bias : bias_data[i % elempack] * (scale_out_data_size == 1 ? scale_out0 : scale_out_data[i % elempack]));

        float v = *intptr * scale_i + bias_i;
        if (v < 0) v *= slope;
        *ptr = float2int8(v);
        intptr++;
        ptr++;
    }
}

static void requantize(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack)
{
    if (activation_type == 1)
    {
        requantize_relu(intptr, ptr, scale_in_data, bias_data, scale_out_data, elemcount, elempack);
        return;
    }

    if (activation_type == 2 && activation_params[0] > 0.f)
    {
        const float slope = activation_params[0];
        requantize_leakyrelu(intptr, ptr, scale_in_data, bias_data, scale_out_data, slope, elemcount, elempack);
        return;
    }

    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    const float scale_in0 = scale_in_data[0];
    const float scale_out0 = scale_out_data[0];

    int i = 0;
#if __ALTIVEC__
    // pure requantize without activation can be fully vectorized
    const bool vec_ok = activation_type == 0
                        && (scale_in_data_size == 1 || elempack == 4 || elempack == 8)
                        && (bias_data_size == 0 || bias_data_size == 1 || elempack == 4 || elempack == 8)
                        && (scale_out_data_size == 1 || elempack == 4 || elempack == 8);

    v4f32 _scale_in0 = vec_splats(scale_in0);
    v4f32 _scale_in1 = vec_splats(scale_in0);
    if (scale_in_data_size > 1 && elempack == 4)
    {
        _scale_in0 = __ncnn_altivec_load(scale_in_data);
        _scale_in1 = _scale_in0;
    }
    if (scale_in_data_size > 1 && elempack == 8)
    {
        _scale_in0 = __ncnn_altivec_load(scale_in_data);
        _scale_in1 = __ncnn_altivec_load((const float*)scale_in_data + 4);
    }

    v4f32 _scale_out0 = vec_splats(scale_out0);
    v4f32 _scale_out1 = vec_splats(scale_out0);
    if (scale_out_data_size > 1 && elempack == 4)
    {
        _scale_out0 = __ncnn_altivec_load(scale_out_data);
        _scale_out1 = _scale_out0;
    }
    if (scale_out_data_size > 1 && elempack == 8)
    {
        _scale_out0 = __ncnn_altivec_load(scale_out_data);
        _scale_out1 = __ncnn_altivec_load((const float*)scale_out_data + 4);
    }

    if (vec_ok && bias_data_size == 0)
    {
        for (; i + 7 < size; i += 8)
        {
            v4f32 _v0 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            v4f32 _v1 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr + 4));
            _v0 = __ncnn_altivec_mul(_v0, _scale_in0);
            _v1 = __ncnn_altivec_mul(_v1, _scale_in1);
            _v0 = __ncnn_altivec_mul(_v0, _scale_out0);
            _v1 = __ncnn_altivec_mul(_v1, _scale_out1);
            v16i8 _b0 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v0), -127, 127);
            v16i8 _b1 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v1), -127, 127);
            memcpy(ptr, &_b0, 4);
            memcpy(ptr + 4, &_b1, 4);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            _v = __ncnn_altivec_mul(_v, _scale_in0);
            _v = __ncnn_altivec_mul(_v, _scale_out0);
            v16i8 _b = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v), -127, 127);
            memcpy(ptr, &_b, 4);
            intptr += 4;
            ptr += 4;
        }
    }
    else if (vec_ok)
    {
        const float bias0 = bias_data[0];
        v4f32 _bias0 = vec_splats(bias0);
        v4f32 _bias1 = vec_splats(bias0);
        if (bias_data_size > 1 && elempack == 4)
        {
            _bias0 = __ncnn_altivec_load(bias_data);
            _bias1 = _bias0;
        }
        if (bias_data_size > 1 && elempack == 8)
        {
            _bias0 = __ncnn_altivec_load(bias_data);
            _bias1 = __ncnn_altivec_load((const float*)bias_data + 4);
        }

        for (; i + 7 < size; i += 8)
        {
            v4f32 _v0 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            v4f32 _v1 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr + 4));
            _v0 = __ncnn_altivec_fmadd(_bias0, _v0, _scale_in0);
            _v1 = __ncnn_altivec_fmadd(_bias1, _v1, _scale_in1);
            _v0 = __ncnn_altivec_mul(_v0, _scale_out0);
            _v1 = __ncnn_altivec_mul(_v1, _scale_out1);
            v16i8 _b0 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v0), -127, 127);
            v16i8 _b1 = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v1), -127, 127);
            memcpy(ptr, &_b0, 4);
            memcpy(ptr + 4, &_b1, 4);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            _v = __ncnn_altivec_fmadd(_bias0, _v, _scale_in0);
            _v = __ncnn_altivec_mul(_v, _scale_out0);
            v16i8 _b = __ncnn_altivec_i32_to_i8(__ncnn_altivec_round_away_i32(_v), -127, 127);
            memcpy(ptr, &_b, 4);
            intptr += 4;
            ptr += 4;
        }
    }
#endif // __ALTIVEC__

    for (; i < size; i++)
    {
        const float scale_in = scale_in_data_size == 1 ? scale_in0 : scale_in_data[i % elempack];
        const float bias = bias_data_size == 0 ? 0.f : bias_data_size == 1 ? bias_data[0] : bias_data[i % elempack];
        const float scale_out = scale_out_data_size == 1 ? scale_out0 : scale_out_data[i % elempack];

        float v = *intptr * scale_in + bias;
        v = activation_ss(v, activation_type, activation_params);
        *ptr = float2int8(v * scale_out);
        intptr++;
        ptr++;
    }
}

int Requantize_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
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

            // assert scale_in_data_size == 1
            // assert bias_data_size == 0 || bias_data_size == 1
            // assert scale_out_data_size == 1

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

    if (dims == 3 || dims == 4)
    {
        if (dims == 3)
            top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
        else
            top_blob.create(w, h, d, channels, out_elemsize, elempack, opt.blob_allocator);
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

            requantize(intptr, ptr, scale_in_data_q, bias_data_q, scale_out_data_q, activation_type, activation_params, w * h * d, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
