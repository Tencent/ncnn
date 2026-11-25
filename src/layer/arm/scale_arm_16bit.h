// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<typename Conv>
static void scale_forward_inplace_16bit_pack(unsigned short* ptr, const float* scale, const float* bias, int size)
{
    int i = 0;
#if __ARM_NEON
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _p0 = Conv::load4(ptr);
        float32x4_t _p1 = Conv::load4(ptr + 4);
        float32x4_t _s0 = vld1q_f32(scale + i);
        float32x4_t _s1 = vld1q_f32(scale + i + 4);
        if (bias)
        {
            _p0 = vmlaq_f32(vld1q_f32(bias + i), _p0, _s0);
            _p1 = vmlaq_f32(vld1q_f32(bias + i + 4), _p1, _s1);
        }
        else
        {
            _p0 = vmulq_f32(_p0, _s0);
            _p1 = vmulq_f32(_p1, _s1);
        }
        Conv::store4(ptr, _p0);
        Conv::store4(ptr + 4, _p1);
        ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = Conv::load4(ptr);
        float32x4_t _s = vld1q_f32(scale + i);
        if (bias)
            _p = vmlaq_f32(vld1q_f32(bias + i), _p, _s);
        else
            _p = vmulq_f32(_p, _s);
        Conv::store4(ptr, _p);
        ptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        float v = Conv::load1(ptr) * scale[i];
        if (bias)
            v += bias[i];
        Conv::store1(ptr, v);
        ptr++;
    }
}

template<typename Conv>
static void scale_forward_inplace_16bit_channel(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
    int i = 0;
#if __ARM_NEON
    if (elempack >= 4)
    {
        for (; i + elempack - 1 < size; i += elempack)
        {
            for (int k = 0; k < elempack; k += 4)
            {
                float32x4_t _p = Conv::load4(ptr + i + k);
                float32x4_t _s = vld1q_f32(scale + k);
                if (bias)
                    _p = vmlaq_f32(vld1q_f32(bias + k), _p, _s);
                else
                    _p = vmulq_f32(_p, _s);
                Conv::store4(ptr + i + k, _p);
            }
        }
    }
    else if (elempack == 1)
    {
        float32x4_t _s = vdupq_n_f32(scale[0]);
        float32x4_t _b = bias ? vdupq_n_f32(bias[0]) : vdupq_n_f32(0.f);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = Conv::load4(ptr + i);
            if (bias)
                _p = vmlaq_f32(_b, _p, _s);
            else
                _p = vmulq_f32(_p, _s);
            Conv::store4(ptr + i, _p);
        }
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        float v = Conv::load1(ptr + i) * scale[i % elempack];
        if (bias)
            v += bias[i % elempack];
        Conv::store1(ptr + i, v);
    }
}

template<typename Conv>
static int scale_forward_inplace_16bit(Mat& bottom_top_blob, const Mat& scale_blob, const float* bias_data, bool bias_term, const Option& opt)
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

    // scale_blob may be 16-bit when supplied as the second input
    Mat scale_fp32;
    const float* scale = 0;
    if (scale_blob.elembits() == 16)
    {
        const int scale_data_size = scale_blob.w * scale_blob.elempack;
        scale_fp32.create(scale_data_size, 4u, 1, opt.workspace_allocator);
        if (scale_fp32.empty())
            return -100;

        const unsigned short* src = scale_blob;
        float* dst = scale_fp32;
        for (int i = 0; i < scale_data_size; i++)
        {
            dst[i] = Conv::load1(src + i);
        }
        scale = scale_fp32;
    }
    else
    {
        scale = scale_blob;
    }

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;
        const int size = w * elempack;
        scale_forward_inplace_16bit_pack<Conv>(ptr, scale, bias_term ? bias_data : 0, size);
    }

    if (dims == 2)
    {
        const int size = w * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            scale_forward_inplace_16bit_channel<Conv>(ptr, scale + i * elempack, bias_term ? bias_data + i * elempack : 0, size, elempack);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            scale_forward_inplace_16bit_channel<Conv>(ptr, scale + q * elempack, bias_term ? bias_data + q * elempack : 0, size, elempack);
        }
    }

    return 0;
}
