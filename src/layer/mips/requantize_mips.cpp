// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "requantize_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_activation.h"
#include "mips_usability.h"

namespace ncnn {

Requantize_mips::Requantize_mips()
{
#if __mips_msa
    support_packing = true;
#endif
}

static void requantize_relu(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int elemcount, int elempack)
{
    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("requantize_relu %d %d %d   %d %d", scale_in_data_size, bias_data_size, scale_out_data_size, elemcount, elempack);

    // int8(relu(v * scale_in) * scale_out)
    // int8_relu(v * (scale_in * scale_out))

    // int8(relu(v * scale_in + bias) * scale_out)
    // int8_relu(v * (scale_in * scale_out) + (bias * scale_out))

    float scale_in = scale_in_data[0];
#if __mips_msa
    v4f32 _scale_in0 = (v4f32)__msa_fill_w_f32(scale_in);
    v4f32 _scale_in1 = _scale_in0;
    if (scale_in_data_size > 1)
    {
        if (elempack == 8)
        {
            _scale_in0 = (v4f32)__msa_ld_w((const float*)scale_in_data, 0);
            _scale_in1 = (v4f32)__msa_ld_w((const float*)scale_in_data + 4, 0);
        }
    }
#endif // __mips_msa

    float scale_out = scale_out_data[0];
#if __mips_msa
    v4f32 _scale_out0 = (v4f32)__msa_fill_w_f32(scale_out);
    v4f32 _scale_out1 = _scale_out0;
    if (scale_out_data_size > 1)
    {
        if (elempack == 8)
        {
            _scale_out0 = (v4f32)__msa_ld_w((const float*)scale_out_data, 0);
            _scale_out1 = (v4f32)__msa_ld_w((const float*)scale_out_data + 4, 0);
        }
    }
#endif // __mips_msa

    float scale = scale_in * scale_out;
#if __mips_msa
    v4f32 _scale0 = __msa_fmul_w(_scale_in0, _scale_out0);
    v4f32 _scale1 = __msa_fmul_w(_scale_in1, _scale_out1);
#endif // __mips_msa

    if (bias_data_size == 0)
    {
        int i = 0;
#if __mips_msa
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(intptr + 32);
            v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
            _v0 = __msa_fmul_w(_v0, _scale0);
            _v1 = __msa_fmul_w(_v1, _scale1);
            *((int64_t*)ptr) = float2int8relu(_v0, _v1);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            _v = __msa_fmul_w(_v, _scale0);
            v16i8 v = float2int8relu(_v);
            ptr[0] = v[0];
            ptr[1] = v[1];
            ptr[2] = v[2];
            ptr[3] = v[3];
            intptr += 4;
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = *intptr * scale;
            if (v < 0) v = 0;
            *ptr = float2int8(v);
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __mips_msa
        v4f32 _bias0 = (v4f32)__msa_fill_w_f32(bias);
        v4f32 _bias1 = _bias0;
        if (bias_data_size > 1)
        {
            if (elempack == 8)
            {
                _bias0 = (v4f32)__msa_ld_w((const float*)bias_data, 0);
                _bias1 = (v4f32)__msa_ld_w((const float*)bias_data + 4, 0);
            }
        }
#endif // __mips_msa

        bias = bias * scale_out;
#if __mips_msa
        _bias0 = __msa_fmul_w(_bias0, _scale_out0);
        _bias1 = __msa_fmul_w(_bias1, _scale_out1);
#endif // __mips_msa

        int i = 0;
#if __mips_msa
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(intptr + 32);
            v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
            _v0 = __msa_fmadd_w(_bias0, _v0, _scale0);
            _v1 = __msa_fmadd_w(_bias1, _v1, _scale1);
            *((int64_t*)ptr) = float2int8relu(_v0, _v1);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            _v = __msa_fmadd_w(_bias0, _v, _scale0);
            v16i8 v = float2int8relu(_v);
            ptr[0] = v[0];
            ptr[1] = v[1];
            ptr[2] = v[2];
            ptr[3] = v[3];
            intptr += 4;
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = *intptr * scale + bias;
            if (v < 0) v = 0;
            *ptr = float2int8(v);
            intptr++;
            ptr++;
        }
    }
}

static void requantize_leakyrelu(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, float slope, int elemcount, int elempack)
{
    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("requantize_leakyrelu %d %d %d   %d %d", scale_in_data_size, bias_data_size, scale_out_data_size, elemcount, elempack);

    // int8(leakyrelu(v * scale_in, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out), slope)

    // int8(leakyrelu(v * scale_in + bias, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out) + (bias * scale_out), slope)

    float scale_in = scale_in_data[0];
#if __mips_msa
    v4f32 _scale_in0 = (v4f32)__msa_fill_w_f32(scale_in);
    v4f32 _scale_in1 = _scale_in0;
    if (scale_in_data_size > 1)
    {
        if (elempack == 8)
        {
            _scale_in0 = (v4f32)__msa_ld_w((const float*)scale_in_data, 0);
            _scale_in1 = (v4f32)__msa_ld_w((const float*)scale_in_data + 4, 0);
        }
    }
#endif // __mips_msa

    float scale_out = scale_out_data[0];
#if __mips_msa
    v4f32 _scale_out0 = (v4f32)__msa_fill_w_f32(scale_out);
    v4f32 _scale_out1 = _scale_out0;
    if (scale_out_data_size > 1)
    {
        if (elempack == 8)
        {
            _scale_out0 = (v4f32)__msa_ld_w((const float*)scale_out_data, 0);
            _scale_out1 = (v4f32)__msa_ld_w((const float*)scale_out_data + 4, 0);
        }
    }
#endif // __mips_msa

    float scale = scale_in * scale_out;
#if __mips_msa
    v4f32 _scale0 = __msa_fmul_w(_scale_in0, _scale_out0);
    v4f32 _scale1 = __msa_fmul_w(_scale_in1, _scale_out1);
    v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);
#endif // __mips_msa

    if (bias_data_size == 0)
    {
        int i = 0;
#if __mips_msa
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(intptr + 32);
            v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
            _v0 = __msa_fmul_w(_v0, _scale0);
            _v1 = __msa_fmul_w(_v1, _scale1);
            *((int64_t*)ptr) = float2int8leakyrelu(_v0, _v1, _slope);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            _v = __msa_fmul_w(_v, _scale0);
            v16i8 v = float2int8leakyrelu(_v, _slope);
            ptr[0] = v[0];
            ptr[1] = v[1];
            ptr[2] = v[2];
            ptr[3] = v[3];
            intptr += 4;
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = *intptr * scale;
            if (v < 0) v *= slope;
            *ptr = float2int8(v);
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __mips_msa
        v4f32 _bias0 = (v4f32)__msa_fill_w_f32(bias);
        v4f32 _bias1 = _bias0;
        if (bias_data_size > 1)
        {
            if (elempack == 8)
            {
                _bias0 = (v4f32)__msa_ld_w((const float*)bias_data, 0);
                _bias1 = (v4f32)__msa_ld_w((const float*)bias_data + 4, 0);
            }
        }
#endif // __mips_msa

        bias = bias * scale_out;
#if __mips_msa
        _bias0 = __msa_fmul_w(_bias0, _scale_out0);
        _bias1 = __msa_fmul_w(_bias1, _scale_out1);
#endif // __mips_msa

        int i = 0;
#if __mips_msa
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(intptr + 32);
            v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
            _v0 = __msa_fmadd_w(_bias0, _v0, _scale0);
            _v1 = __msa_fmadd_w(_bias1, _v1, _scale1);
            *((int64_t*)ptr) = float2int8leakyrelu(_v0, _v1, _slope);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            _v = __msa_fmadd_w(_bias0, _v, _scale0);
            v16i8 v = float2int8leakyrelu(_v, _slope);
            ptr[0] = v[0];
            ptr[1] = v[1];
            ptr[2] = v[2];
            ptr[3] = v[3];
            intptr += 4;
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = *intptr * scale + bias;
            if (v < 0) v *= slope;
            *ptr = float2int8(v);
            intptr++;
            ptr++;
        }
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

    // NCNN_LOGE("requantize %d %d %d   %d %d", scale_in_data_size, bias_data_size, scale_out_data_size, elemcount, elempack);

    float scale_in = scale_in_data[0];
#if __mips_msa
    v4f32 _scale_in0 = (v4f32)__msa_fill_w_f32(scale_in);
    v4f32 _scale_in1 = _scale_in0;
    if (scale_in_data_size > 1)
    {
        if (elempack == 8)
        {
            _scale_in0 = (v4f32)__msa_ld_w((const float*)scale_in_data, 0);
            _scale_in1 = (v4f32)__msa_ld_w((const float*)scale_in_data + 4, 0);
        }
    }
#endif // __mips_msa

    float scale_out = scale_out_data[0];
#if __mips_msa
    v4f32 _scale_out0 = (v4f32)__msa_fill_w_f32(scale_out);
    v4f32 _scale_out1 = _scale_out0;
    if (scale_out_data_size > 1)
    {
        if (elempack == 8)
        {
            _scale_out0 = (v4f32)__msa_ld_w((const float*)scale_out_data, 0);
            _scale_out1 = (v4f32)__msa_ld_w((const float*)scale_out_data + 4, 0);
        }
    }
#endif // __mips_msa

    if (bias_data_size == 0)
    {
        int i = 0;
#if __mips_msa
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(intptr + 32);
            v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
            _v0 = __msa_fmul_w(_v0, _scale_in0);
            _v1 = __msa_fmul_w(_v1, _scale_in1);
            _v0 = activation_ps(_v0, activation_type, activation_params);
            _v1 = activation_ps(_v1, activation_type, activation_params);
            _v0 = __msa_fmul_w(_v0, _scale_out0);
            _v1 = __msa_fmul_w(_v1, _scale_out1);
            *((int64_t*)ptr) = float2int8(_v0, _v1);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            _v = __msa_fmul_w(_v, _scale_in0);
            _v = activation_ps(_v, activation_type, activation_params);
            _v = __msa_fmul_w(_v, _scale_out0);
            v16i8 v = float2int8(_v);
            ptr[0] = v[0];
            ptr[1] = v[1];
            ptr[2] = v[2];
            ptr[3] = v[3];
            intptr += 4;
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = *intptr * scale_in;
            v = activation_ss(v, activation_type, activation_params);
            *ptr = float2int8(v * scale_out);
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __mips_msa
        v4f32 _bias0 = (v4f32)__msa_fill_w_f32(bias);
        v4f32 _bias1 = _bias0;
        if (bias_data_size > 1)
        {
            if (elempack == 8)
            {
                _bias0 = (v4f32)__msa_ld_w((const float*)bias_data, 0);
                _bias1 = (v4f32)__msa_ld_w((const float*)bias_data + 4, 0);
            }
        }
#endif // __mips_msa

        int i = 0;
#if __mips_msa
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(intptr + 32);
            v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
            _v0 = __msa_fmadd_w(_bias0, _v0, _scale_in0);
            _v1 = __msa_fmadd_w(_bias1, _v1, _scale_in1);
            _v0 = activation_ps(_v0, activation_type, activation_params);
            _v1 = activation_ps(_v1, activation_type, activation_params);
            _v0 = __msa_fmul_w(_v0, _scale_out0);
            _v1 = __msa_fmul_w(_v1, _scale_out1);
            *((int64_t*)ptr) = float2int8(_v0, _v1);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
            _v = __msa_fmadd_w(_bias0, _v, _scale_in0);
            _v = activation_ps(_v, activation_type, activation_params);
            _v = __msa_fmul_w(_v, _scale_out0);
            v16i8 v = float2int8(_v);
            ptr[0] = v[0];
            ptr[1] = v[1];
            ptr[2] = v[2];
            ptr[3] = v[3];
            intptr += 4;
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = *intptr * scale_in + bias;
            v = activation_ss(v, activation_type, activation_params);
            *ptr = float2int8(v * scale_out);
            intptr++;
            ptr++;
        }
    }
}

int Requantize_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
