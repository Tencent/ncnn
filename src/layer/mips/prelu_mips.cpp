// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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
#endif // __mips_msa
}

int PReLU_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __mips_msa
    if (elempack == 4)
    {
        v4f32 _zero = (v4f32)__msa_fill_w(0);

        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            if (num_slope > 1)
            {
                const float* slope = slope_data;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _slope = (v4f32)__msa_ld_w(slope + i * 4, 0);
                    v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                    v4f32 _ps = __msa_fmul_w(_p, _slope);
                    _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                    __msa_st_w((v4i32)_p, ptr, 0);
                }
            }
            else
            {
                v4f32 _slope = (v4f32)__msa_fill_w_f32(slope_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                    v4f32 _ps = __msa_fmul_w(_p, _slope);
                    _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                    __msa_st_w((v4i32)_p, ptr, 0);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                v4f32 _slope = num_slope > 1 ? (v4f32)__msa_ld_w((const float*)slope_data + i * 4, 0) : (v4f32)__msa_fill_w_f32(slope_data[0]);

                for (int j = 0; j < w; j++)
                {
                    __builtin_prefetch(ptr + 32);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                    v4f32 _ps = __msa_fmul_w(_p, _slope);
                    _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                    __msa_st_w((v4i32)_p, ptr, 0);

                    ptr += 4;
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                v4f32 _slope = num_slope > 1 ? (v4f32)__msa_ld_w((const float*)slope_data + q * 4, 0) : (v4f32)__msa_fill_w_f32(slope_data[0]);

                for (int i = 0; i < size; i++)
                {
                    __builtin_prefetch(ptr + 32);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                    v4f32 _ps = __msa_fmul_w(_p, _slope);
                    _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                    __msa_st_w((v4i32)_p, ptr, 0);

                    ptr += 4;
                }
            }
        }

        return 0;
    }
#endif // __mips_msa

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float v = ptr[i];
                if (v < 0.f)
                    ptr[i] = v * slope[i];
            }
        }
        else
        {
            const float slope = slope_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float v = ptr[i];
                if (v < 0.f)
                    ptr[i] = v * slope;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            const float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            int j = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);

            for (; j + 3 < w; j += 4)
            {
                __builtin_prefetch(ptr + 32);
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

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        const float* slope_data_ptr = slope_data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);

            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 32);
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

} // namespace ncnn
