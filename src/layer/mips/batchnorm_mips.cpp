// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "batchnorm_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

#if NCNN_BF16
static void batchnorm_bf16s_msa(unsigned short* ptr, const float* a, const float* b, int size, int elempack)
{
#if __mips_msa
    v4f32 _a = (elempack == 4) ? (v4f32)__msa_ld_w(a, 0) : (v4f32)__msa_fill_w_f32(a[0]);
    v4f32 _b = (elempack == 4) ? (v4f32)__msa_ld_w(b, 0) : (v4f32)__msa_fill_w_f32(b[0]);
#endif
    float sa = a[0];
    float sb = b[0];

    int i = 0;
#if __mips_msa
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        _p = __ncnn_msa_fmadd_w(_a, _p, _b);
        float2bfloat_msa_store(_p, ptr);
        ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(sb * bfloat16_to_float32(*ptr) + sa);
        ptr++;
    }
}

static void batchnorm_bf16s_per_element_msa(unsigned short* ptr, const float* a, const float* b, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __mips_msa
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        v4f32 _p = bfloat2float_msa(ptr + i);
        v4f32 _a0 = (v4f32)__msa_ld_w(a + i, 0);
        v4f32 _b0 = (v4f32)__msa_ld_w(b + i, 0);
        _p = __ncnn_msa_fmadd_w(_a0, _p, _b0);
        float2bfloat_msa_store(_p, ptr + i);
    }
    remain_size_start += nn_size * 4;
#endif // __mips_msa
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(b[i] * bfloat16_to_float32(ptr[i]) + a[i]);
    }
}
#endif

BatchNorm_mips::BatchNorm_mips()
{
#if __mips_msa
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
#endif // __mips_msa
}

int BatchNorm_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        int w = bottom_top_blob.w * elempack;

#if __mips_msa
        int nn_w = w / 4;
        int remain_w_start = nn_w * 4;
#else
        int remain_w_start = 0;
#endif // __mips_msa

        float* ptr = bottom_top_blob;

#if __mips_msa
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn_w; i++)
        {
            float* ptr0 = ptr + i * 4;

            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            v4f32 _a = (v4f32)__msa_ld_w((const float*)a_data + i * 4, 0);
            v4f32 _b = (v4f32)__msa_ld_w((const float*)b_data + i * 4, 0);
            _p = __ncnn_msa_fmadd_w(_a, _p, _b);
            __msa_st_w((v4i32)_p, ptr0, 0);
        }
#endif // __mips_msa

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_w_start; i < w; i++)
        {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];
        }
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
#if __mips_msa
            v4f32 _a = elempack == 4 ? (v4f32)__msa_ld_w((const float*)a_data + i * 4, 0) : (v4f32)__msa_fill_w_f32(a);
            v4f32 _b = elempack == 4 ? (v4f32)__msa_ld_w((const float*)b_data + i * 4, 0) : (v4f32)__msa_fill_w_f32(b);
            for (; j + 3 < w; j += 4)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                _p = __ncnn_msa_fmadd_w(_a, _p, _b);
                __msa_st_w((v4i32)_p, ptr, 0);

                ptr += 4;
            }
#endif // __mips_msa
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
#if __mips_msa
            v4f32 _a = elempack == 4 ? (v4f32)__msa_ld_w((const float*)a_data + q * 4, 0) : (v4f32)__msa_fill_w_f32(a);
            v4f32 _b = elempack == 4 ? (v4f32)__msa_ld_w((const float*)b_data + q * 4, 0) : (v4f32)__msa_fill_w_f32(b);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                _p = __ncnn_msa_fmadd_w(_a, _p, _b);
                __msa_st_w((v4i32)_p, ptr, 0);

                ptr += 4;
            }
#endif // __mips_msa
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
int BatchNorm_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;
        const float* aptr = a_data;
        const float* bptr = b_data;

        const int size = w * elempack;

        batchnorm_bf16s_per_element_msa(ptr, aptr, bptr, size, opt.num_threads);
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* aptr = (const float*)a_data + i * elempack;
            const float* bptr = (const float*)b_data + i * elempack;

            batchnorm_bf16s_msa(ptr, aptr, bptr, size, elempack);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* aptr = (const float*)a_data + q * elempack;
            const float* bptr = (const float*)b_data + q * elempack;

            batchnorm_bf16s_msa(ptr, aptr, bptr, size, elempack);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
