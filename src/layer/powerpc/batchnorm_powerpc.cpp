// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "batchnorm_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

BatchNorm_powerpc::BatchNorm_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int BatchNorm_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    if (dims == 1)
    {
        int w = bottom_top_blob.w * elempack;

        float* ptr = bottom_top_blob;

#if __ALTIVEC__
        int i = 0;
        for (; i + 3 < w; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr + i);
            v4f32 _a = __ncnn_altivec_load((const float*)a_data + i);
            v4f32 _b = __ncnn_altivec_load((const float*)b_data + i);
            _p = vec_madd(_p, _b, _a);
            __ncnn_altivec_store(ptr + i, _p);
        }
        for (; i < w; i++)
        {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];
        }
#else
        for (int i = 0; i < w; i++)
        {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];
        }
#endif // __ALTIVEC__
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w * elempack;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float a = a_data[i];
            float b = b_data[i];

            int j = 0;
#if __ALTIVEC__
            v4f32 _a = (elempack == 4) ? __ncnn_altivec_load((const float*)a_data + i * 4) : vec_splats(a);
            v4f32 _b = (elempack == 4) ? __ncnn_altivec_load((const float*)b_data + i * 4) : vec_splats(b);
            for (; j + 3 < w; j += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                _p = vec_madd(_p, _b, _a);
                __ncnn_altivec_store(ptr, _p);
                ptr += 4;
            }
#endif // __ALTIVEC__
            for (; j < w; j++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;
        int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];

            int i = 0;
#if __ALTIVEC__
            v4f32 _a = (elempack == 4) ? __ncnn_altivec_load((const float*)a_data + q * 4) : vec_splats(a);
            v4f32 _b = (elempack == 4) ? __ncnn_altivec_load((const float*)b_data + q * 4) : vec_splats(b);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                _p = vec_madd(_p, _b, _a);
                __ncnn_altivec_store(ptr, _p);
                ptr += 4;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int BatchNorm_powerpc::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        int w = bottom_top_blob.w * elempack;

        unsigned short* ptr = bottom_top_blob;
        const float* aptr = (const float*)a_data;
        const float* bptr = (const float*)b_data;

#if __ALTIVEC__
        int i = 0;
        for (; i + 3 < w; i += 4)
        {
            v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr + i);
            v4f32 _a = __ncnn_altivec_load(aptr + i);
            v4f32 _b = __ncnn_altivec_load(bptr + i);
            _p = vec_madd(_p, _b, _a);
            __ncnn_altivec_f32_to_bf16(ptr + i, _p);
        }
        for (; i < w; i++)
        {
            float v = bfloat16_to_float32(ptr[i]);
            ptr[i] = float32_to_bfloat16(bptr[i] * v + aptr[i]);
        }
#else
        for (int i = 0; i < w; i++)
        {
            float v = bfloat16_to_float32(ptr[i]);
            ptr[i] = float32_to_bfloat16(bptr[i] * v + aptr[i]);
        }
#endif // __ALTIVEC__
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w * elempack;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            float a = a_data[i];
            float b = b_data[i];

            int j = 0;
#if __ALTIVEC__
            v4f32 _a = (elempack == 4) ? __ncnn_altivec_load((const float*)a_data + i * 4) : vec_splats(a);
            v4f32 _b = (elempack == 4) ? __ncnn_altivec_load((const float*)b_data + i * 4) : vec_splats(b);
            for (; j + 3 < w; j += 4)
            {
                v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
                _p = vec_madd(_p, _b, _a);
                __ncnn_altivec_f32_to_bf16(ptr, _p);
                ptr += 4;
            }
#endif // __ALTIVEC__
            for (; j < w; j++)
            {
                float v = bfloat16_to_float32(*ptr);
                *ptr = float32_to_bfloat16(b * v + a);
                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;
        int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];

            int i = 0;
#if __ALTIVEC__
            v4f32 _a = (elempack == 4) ? __ncnn_altivec_load((const float*)a_data + q * 4) : vec_splats(a);
            v4f32 _b = (elempack == 4) ? __ncnn_altivec_load((const float*)b_data + q * 4) : vec_splats(b);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
                _p = vec_madd(_p, _b, _a);
                __ncnn_altivec_f32_to_bf16(ptr, _p);
                ptr += 4;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                *ptr = float32_to_bfloat16(b * v + a);
                ptr++;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
