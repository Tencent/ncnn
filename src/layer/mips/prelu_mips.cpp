// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

PReLU_mips::PReLU_mips()
{
#if __mips_msa
    support_packing = true;
    support_any_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int PReLU_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

#if __mips_msa
        int nn_w = w / 4;
        int remain_w_start = nn_w * 4;
#else
        int remain_w_start = 0;
#endif // __mips_msa

        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;

#if __mips_msa
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < nn_w; i++)
            {
                float* ptr0 = ptr + i * 4;

                v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
                v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _slope = (v4f32)__msa_ld_w(slope + i * 4, 0);
                v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                v4f32 _ps = __msa_fmul_w(_p, _slope);
                _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                __msa_st_w((v4i32)_p, ptr0, 0);
            }
#endif // __mips_msa

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_w_start; i < w; i++)
            {
                float v = ptr[i];
                if (v < 0.f)
                    ptr[i] = v * slope[i];
            }
        }
        else
        {
            const float slope = slope_data[0];

#if __mips_msa
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < nn_w; i++)
            {
                float* ptr0 = ptr + i * 4;

                v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
                v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);
                v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                v4f32 _ps = __msa_fmul_w(_p, _slope);
                _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                __msa_st_w((v4i32)_p, ptr0, 0);
            }
#endif // __mips_msa

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_w_start; i < w; i++)
            {
                float v = ptr[i];
                if (v < 0.f)
                    ptr[i] = v * slope;
            }
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

            const float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            int j = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            v4f32 _slope = (elempack == 4 && num_slope > 1) ? (v4f32)__msa_ld_w((const float*)slope_data + i * 4, 0) : (v4f32)__msa_fill_w_f32(slope);

            for (; j + 3 < w; j += 4)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                v4f32 _ps = __msa_fmul_w(_p, _slope);
                _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                __msa_st_w((v4i32)_p, ptr, 0);

                ptr += 4;
            }
#endif // __mips_msa
            for (; j < w; j++)
            {
                float v = *ptr;
                if (v < 0.f)
                    *ptr = v * slope;

                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d * elempack;

        const float* slope_data_ptr = slope_data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            v4f32 _slope = (elempack == 4 && num_slope > 1) ? (v4f32)__msa_ld_w((const float*)slope_data + q * 4, 0) : (v4f32)__msa_fill_w_f32(slope);

            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                v4f32 _ps = __msa_fmul_w(_p, _slope);
                _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                __msa_st_w((v4i32)_p, ptr, 0);

                ptr += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;

                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int PReLU_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        int w = bottom_top_blob.w * elempack;

#if __mips_msa
        int nn_w8 = w / 8;
        int nn_w = (w - nn_w8 * 8) / 4;
        int remain_w_start = nn_w8 * 8 + nn_w * 4;
#else
        int remain_w_start = 0;
#endif // __mips_msa

        unsigned short* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;

#if __mips_msa
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < nn_w8; i++)
            {
                unsigned short* ptr0 = ptr + i * 8;

                v8i16 _zero_bf16 = __msa_fill_h(0);
                v8i16 _p01 = __msa_ld_h(ptr0, 0);
                v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
                v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
                v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _slope0 = (v4f32)__msa_ld_w((const float*)slope + i * 8, 0);
                v4f32 _slope1 = (v4f32)__msa_ld_w((const float*)slope + i * 8 + 4, 0);
                v4i32_w _lemask0 = __msa_fcle_w(_p0, _zero);
                v4i32_w _lemask1 = __msa_fcle_w(_p1, _zero);
                v4f32 _ps0 = __msa_fmul_w(_p0, _slope0);
                v4f32 _ps1 = __msa_fmul_w(_p1, _slope1);
                _p0 = (v4f32)__msa_bsel_v((v16u8)_lemask0, (v16u8)_p0, (v16u8)_ps0);
                _p1 = (v4f32)__msa_bsel_v((v16u8)_lemask1, (v16u8)_p1, (v16u8)_ps1);
                __msa_st_w(float2bfloat_msa(_p0, _p1), ptr0, 0);
            }
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < nn_w; i++)
            {
                unsigned short* ptr0 = ptr + nn_w8 * 8 + i * 4;

                v4f32 _p = bfloat2float_msa(ptr0);
                v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _slope = (v4f32)__msa_ld_w((const float*)slope + nn_w8 * 8 + i * 4, 0);
                v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                v4f32 _ps = __msa_fmul_w(_p, _slope);
                _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                *(int64_t*)ptr0 = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
            }
#endif // __mips_msa

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_w_start; i < w; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                if (v < 0.f)
                    ptr[i] = float32_to_bfloat16(v * slope[i]);
                else
                    ptr[i] = float32_to_bfloat16(v);
            }
        }
        else
        {
            const float slope = slope_data[0];

#if __mips_msa
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < nn_w8; i++)
            {
                unsigned short* ptr0 = ptr + i * 8;

                v8i16 _zero_bf16 = __msa_fill_h(0);
                v8i16 _p01 = __msa_ld_h(ptr0, 0);
                v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
                v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
                v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);
                v4i32_w _lemask0 = __msa_fcle_w(_p0, _zero);
                v4i32_w _lemask1 = __msa_fcle_w(_p1, _zero);
                v4f32 _ps0 = __msa_fmul_w(_p0, _slope);
                v4f32 _ps1 = __msa_fmul_w(_p1, _slope);
                _p0 = (v4f32)__msa_bsel_v((v16u8)_lemask0, (v16u8)_p0, (v16u8)_ps0);
                _p1 = (v4f32)__msa_bsel_v((v16u8)_lemask1, (v16u8)_p1, (v16u8)_ps1);
                __msa_st_w(float2bfloat_msa(_p0, _p1), ptr0, 0);
            }
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < nn_w; i++)
            {
                unsigned short* ptr0 = ptr + nn_w8 * 8 + i * 4;

                v4f32 _p = bfloat2float_msa(ptr0);
                v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);
                v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                v4f32 _ps = __msa_fmul_w(_p, _slope);
                _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                *(int64_t*)ptr0 = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
            }
#endif // __mips_msa

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_w_start; i < w; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                if (v < 0.f)
                    ptr[i] = float32_to_bfloat16(v * slope);
                else
                    ptr[i] = float32_to_bfloat16(v);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w * elempack;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            const float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            int j = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            v4f32 _slope0 = (elempack == 4 && num_slope > 1) ? (v4f32)__msa_ld_w((const float*)slope_data + i * 4, 0) : (v4f32)__msa_fill_w_f32(slope);
            v4f32 _slope1 = _slope0;
            if (elempack == 8 && num_slope > 1)
            {
                _slope0 = (v4f32)__msa_ld_w((const float*)slope_data + i * 8, 0);
                _slope1 = (v4f32)__msa_ld_w((const float*)slope_data + i * 8 + 4, 0);
            }

            v8i16 _zero_bf16 = __msa_fill_h(0);
            for (; j + 7 < w; j += 8)
            {
                v8i16 _p01 = __msa_ld_h(ptr, 0);
                v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
                v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
                v4i32_w _lemask0 = __msa_fcle_w(_p0, _zero);
                v4i32_w _lemask1 = __msa_fcle_w(_p1, _zero);
                v4f32 _ps0 = __msa_fmul_w(_p0, _slope0);
                v4f32 _ps1 = __msa_fmul_w(_p1, _slope1);
                _p0 = (v4f32)__msa_bsel_v((v16u8)_lemask0, (v16u8)_p0, (v16u8)_ps0);
                _p1 = (v4f32)__msa_bsel_v((v16u8)_lemask1, (v16u8)_p1, (v16u8)_ps1);
                __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
                ptr += 8;
            }
            for (; j + 3 < w; j += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr);
                v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                v4f32 _ps = __msa_fmul_w(_p, _slope0);
                _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                *(int64_t*)ptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
                ptr += 4;
            }
#endif // __mips_msa
            for (; j < w; j++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f)
                    *ptr = float32_to_bfloat16(v * slope);
                else
                    *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d * elempack;

        const float* slope_data_ptr = slope_data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            v4f32 _slope0 = (elempack == 4 && num_slope > 1) ? (v4f32)__msa_ld_w((const float*)slope_data + q * 4, 0) : (v4f32)__msa_fill_w_f32(slope);
            v4f32 _slope1 = _slope0;
            if (elempack == 8 && num_slope > 1)
            {
                _slope0 = (v4f32)__msa_ld_w((const float*)slope_data + q * 8, 0);
                _slope1 = (v4f32)__msa_ld_w((const float*)slope_data + q * 8 + 4, 0);
            }

            v8i16 _zero_bf16 = __msa_fill_h(0);
            for (; i + 7 < size; i += 8)
            {
                v8i16 _p01 = __msa_ld_h(ptr, 0);
                v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
                v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
                v4i32_w _lemask0 = __msa_fcle_w(_p0, _zero);
                v4i32_w _lemask1 = __msa_fcle_w(_p1, _zero);
                v4f32 _ps0 = __msa_fmul_w(_p0, _slope0);
                v4f32 _ps1 = __msa_fmul_w(_p1, _slope1);
                _p0 = (v4f32)__msa_bsel_v((v16u8)_lemask0, (v16u8)_p0, (v16u8)_ps0);
                _p1 = (v4f32)__msa_bsel_v((v16u8)_lemask1, (v16u8)_p1, (v16u8)_ps1);
                __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr);
                v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                v4f32 _ps = __msa_fmul_w(_p, _slope0);
                _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                *(int64_t*)ptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
                ptr += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f)
                    *ptr = float32_to_bfloat16(v * slope);
                else
                    *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
