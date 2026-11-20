// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Scale_powerpc::Scale_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Scale_powerpc::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blobs, opt);
#endif

    const Mat& scale_blob = bottom_top_blobs[1];

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    const float* scale = scale_blob;
    const float* bias = bias_data;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        const int size = w * elempack;

#if __ALTIVEC__
        if (bias_term)
        {
            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr + i);
                v4f32 _s = __ncnn_altivec_load(scale + i);
                v4f32 _bias = __ncnn_altivec_load(bias + i);
                _p = vec_madd(_p, _s, _bias);
                __ncnn_altivec_store(ptr + i, _p);
            }
            for (; i < size; i++)
            {
                ptr[i] = ptr[i] * scale[i] + bias[i];
            }
        }
        else
        {
            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr + i);
                v4f32 _s = __ncnn_altivec_load(scale + i);
                _p = __ncnn_altivec_mul(_p, _s);
                __ncnn_altivec_store(ptr + i, _p);
            }
            for (; i < size; i++)
            {
                ptr[i] *= scale[i];
            }
        }
#else
        if (bias_term)
        {
            for (int i = 0; i < size; i++)
            {
                ptr[i] = ptr[i] * scale[i] + bias[i];
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                ptr[i] *= scale[i];
            }
        }
#endif // __ALTIVEC__

        return 0;
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            float s = scale[i];
            float b = bias_term ? bias[i] : 0.f;

#if __ALTIVEC__
            v4f32 _s = (elempack == 4) ? __ncnn_altivec_load(scale + i * 4) : vec_splats(s);
            v4f32 _b = (elempack == 4) ? __ncnn_altivec_load(bias + i * 4) : vec_splats(b);

            int j = 0;
            if (bias_term)
            {
                for (; j + 3 < size; j += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    _p = vec_madd(_p, _s, _b);
                    __ncnn_altivec_store(ptr, _p);
                    ptr += 4;
                }
            }
            else
            {
                for (; j + 3 < size; j += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    _p = __ncnn_altivec_mul(_p, _s);
                    __ncnn_altivec_store(ptr, _p);
                    ptr += 4;
                }
            }
            for (; j < size; j++)
            {
                *ptr = *ptr * s + b;
                ptr++;
            }
#else
            if (bias_term)
            {
                for (int j = 0; j < size; j++)
                {
                    ptr[j] = ptr[j] * s + b;
                }
            }
            else
            {
                for (int j = 0; j < size; j++)
                {
                    ptr[j] *= s;
                }
            }
#endif // __ALTIVEC__
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale[q];
            float b = bias_term ? bias[q] : 0.f;

#if __ALTIVEC__
            v4f32 _s = (elempack == 4) ? __ncnn_altivec_load(scale + q * 4) : vec_splats(s);
            v4f32 _b = (elempack == 4) ? __ncnn_altivec_load(bias + q * 4) : vec_splats(b);

            int i = 0;
            if (bias_term)
            {
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    _p = vec_madd(_p, _s, _b);
                    __ncnn_altivec_store(ptr, _p);
                    ptr += 4;
                }
            }
            else
            {
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    _p = __ncnn_altivec_mul(_p, _s);
                    __ncnn_altivec_store(ptr, _p);
                    ptr += 4;
                }
            }
            for (; i < size; i++)
            {
                *ptr = *ptr * s + b;
                ptr++;
            }
#else
            if (bias_term)
            {
                for (int i = 0; i < size; i++)
                {
                    *ptr = *ptr * s + b;
                    ptr++;
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    *ptr *= s;
                    ptr++;
                }
            }
#endif // __ALTIVEC__
        }

        return 0;
    }

    return 0;
}

#if NCNN_BF16
int Scale_powerpc::forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    // scale blob may be bf16 as well, convert to fp32 for the per-channel params
    const float* scale = 0;
    Mat scale_fp32;
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
            dst[i] = bfloat16_to_float32(src[i]);
        }
        scale = scale_fp32;
    }
    else
    {
        scale = scale_blob;
    }
    const float* bias = bias_data;

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;
        const int size = w * elempack;

        if (bias_term)
        {
            for (int i = 0; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                ptr[i] = float32_to_bfloat16(v * scale[i] + bias[i]);
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                ptr[i] = float32_to_bfloat16(v * scale[i]);
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* sptr = scale + i * elempack;
            const float* bptr = bias + i * elempack;

            float s = sptr[0];
            float b = bias_term ? bptr[0] : 0.f;

#if __ALTIVEC__
            v4f32 _s = (elempack == 4) ? __ncnn_altivec_load(sptr) : vec_splats(s);
            v4f32 _b = (elempack == 4) ? __ncnn_altivec_load(bptr) : vec_splats(b);

            int j = 0;
            if (bias_term)
            {
                for (; j + 3 < size; j += 4)
                {
                    v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
                    _p = vec_madd(_p, _s, _b);
                    __ncnn_altivec_f32_to_bf16(ptr, _p);
                    ptr += 4;
                }
            }
            else
            {
                for (; j + 3 < size; j += 4)
                {
                    v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
                    _p = __ncnn_altivec_mul(_p, _s);
                    __ncnn_altivec_f32_to_bf16(ptr, _p);
                    ptr += 4;
                }
            }
            for (; j < size; j++)
            {
                float v = bfloat16_to_float32(*ptr);
                *ptr = float32_to_bfloat16(v * s + b);
                ptr++;
            }
#else
            if (bias_term)
            {
                for (int j = 0; j < size; j++)
                {
                    float v = bfloat16_to_float32(ptr[j]);
                    ptr[j] = float32_to_bfloat16(v * s + b);
                }
            }
            else
            {
                for (int j = 0; j < size; j++)
                {
                    float v = bfloat16_to_float32(ptr[j]);
                    ptr[j] = float32_to_bfloat16(v * s);
                }
            }
#endif // __ALTIVEC__
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* sptr = scale + q * elempack;
            const float* bptr = bias + q * elempack;

            float s = sptr[0];
            float b = bias_term ? bptr[0] : 0.f;

#if __ALTIVEC__
            v4f32 _s = (elempack == 4) ? __ncnn_altivec_load(sptr) : vec_splats(s);
            v4f32 _b = (elempack == 4) ? __ncnn_altivec_load(bptr) : vec_splats(b);

            int i = 0;
            if (bias_term)
            {
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
                    _p = vec_madd(_p, _s, _b);
                    __ncnn_altivec_f32_to_bf16(ptr, _p);
                    ptr += 4;
                }
            }
            else
            {
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
                    _p = __ncnn_altivec_mul(_p, _s);
                    __ncnn_altivec_f32_to_bf16(ptr, _p);
                    ptr += 4;
                }
            }
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                *ptr = float32_to_bfloat16(v * s + b);
                ptr++;
            }
#else
            if (bias_term)
            {
                for (int i = 0; i < size; i++)
                {
                    float v = bfloat16_to_float32(ptr[i]);
                    ptr[i] = float32_to_bfloat16(v * s + b);
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    float v = bfloat16_to_float32(ptr[i]);
                    ptr[i] = float32_to_bfloat16(v * s);
                }
            }
#endif // __ALTIVEC__
        }

        return 0;
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
