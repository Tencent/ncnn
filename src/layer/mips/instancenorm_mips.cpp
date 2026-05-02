// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "instancenorm_mips.h"

#if __mips_msa
#include <msa.h>
#include "mips_usability.h"
#endif // __mips_msa

namespace ncnn {

InstanceNorm_mips::InstanceNorm_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int InstanceNorm_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

#if __mips_msa
    int elempack = bottom_top_blob.elempack;
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            v4f32 _sum = (v4f32)__msa_fill_w(0);
            const float* ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
                _sum = __msa_fadd_w(_sum, _p);
                ptr0 += 4;
            }

            float sum_data[4];
            __msa_st_w((v4i32)_sum, sum_data, 0);

            float mean_data[4];
            for (int i = 0; i < 4; i++)
            {
                mean_data[i] = sum_data[i] / size;
            }
            v4f32 _mean = (v4f32)__msa_ld_w(mean_data, 0);

            v4f32 _sqsum = (v4f32)__msa_fill_w(0);
            ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
                _p = __msa_fsub_w(_p, _mean);
                _sqsum = __ncnn_msa_fmadd_w(_sqsum, _p, _p);
                ptr0 += 4;
            }

            float sqsum_data[4];
            __msa_st_w((v4i32)_sqsum, sqsum_data, 0);

            float a_data[4];
            float b_data[4];
            if (affine)
            {
                const float* gamma_ptr = (const float*)gamma_data + q * 4;
                const float* beta_ptr = (const float*)beta_data + q * 4;

                for (int i = 0; i < 4; i++)
                {
                    float a = gamma_ptr[i] / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a + beta_ptr[i];
                }
            }
            else
            {
                for (int i = 0; i < 4; i++)
                {
                    float a = 1.f / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a;
                }
            }

            v4f32 _a = (v4f32)__msa_ld_w(a_data, 0);
            v4f32 _b = (v4f32)__msa_ld_w(b_data, 0);

            for (int i = 0; i < size; i++)
            {
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                _p = __ncnn_msa_fmadd_w(_b, _p, _a);
                __msa_st_w((v4i32)_p, ptr, 0);
                ptr += 4;
            }
        }

        return 0;
    }
#endif // __mips_msa

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        const float* ptr0 = ptr;

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;

        int i = 0;
#if __mips_msa
        v4f32 _sum = (v4f32)__msa_fill_w(0);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _sum = __msa_fadd_w(_sum, _p);
            ptr0 += 4;
        }
        sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa
        for (; i < size; i++)
        {
            sum += *ptr0++;
        }

        float mean = sum / size;

        ptr0 = ptr;
        i = 0;
#if __mips_msa
        v4f32 _sqsum = (v4f32)__msa_fill_w(0);
        v4f32 _mean = __msa_fill_w_f32(mean);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr0, 0);
            _p = __msa_fsub_w(_p, _mean);
            _sqsum = __ncnn_msa_fmadd_w(_sqsum, _p, _p);
            ptr0 += 4;
        }
        sqsum += __msa_reduce_fadd_w(_sqsum);
#endif // __mips_msa
        for (; i < size; i++)
        {
            float tmp = *ptr0++ - mean;
            sqsum += tmp * tmp;
        }

        float var = sqsum / size;

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = gamma / sqrtf(var + eps);
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / sqrtf(var + eps);
            b = -mean * a;
        }

        i = 0;
#if __mips_msa
        v4f32 _a = __msa_fill_w_f32(a);
        v4f32 _b = __msa_fill_w_f32(b);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = __ncnn_msa_fmadd_w(_b, _p, _a);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = *ptr * a + b;
            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int InstanceNorm_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

#if __mips_msa
    int elempack = bottom_top_blob.elempack;
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            // compute mean
            v4f32 _sum = (v4f32)__msa_fill_w(0);
            const unsigned short* ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                v4f32 _p = bfloat2float_msa(ptr0);
                _sum = __msa_fadd_w(_sum, _p);
                ptr0 += 4;
            }

            float sum_data[4];
            __msa_st_w((v4i32)_sum, sum_data, 0);

            float mean_data[4];
            for (int i = 0; i < 4; i++)
            {
                mean_data[i] = sum_data[i] / size;
            }
            v4f32 _mean = (v4f32)__msa_ld_w(mean_data, 0);

            // compute variance
            v4f32 _sqsum = (v4f32)__msa_fill_w(0);
            ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                v4f32 _p = bfloat2float_msa(ptr0);
                _p = __msa_fsub_w(_p, _mean);
                _sqsum = __ncnn_msa_fmadd_w(_sqsum, _p, _p);
                ptr0 += 4;
            }

            float sqsum_data[4];
            __msa_st_w((v4i32)_sqsum, sqsum_data, 0);

            float a_data[4];
            float b_data[4];
            if (affine)
            {
                const float* gamma_ptr = (const float*)gamma_data + q * 4;
                const float* beta_ptr = (const float*)beta_data + q * 4;

                for (int i = 0; i < 4; i++)
                {
                    float a = gamma_ptr[i] / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a + beta_ptr[i];
                }
            }
            else
            {
                for (int i = 0; i < 4; i++)
                {
                    float a = 1.f / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a;
                }
            }

            v4f32 _a = (v4f32)__msa_ld_w(a_data, 0);
            v4f32 _b = (v4f32)__msa_ld_w(b_data, 0);

            for (int i = 0; i < size; i++)
            {
                v4f32 _p = bfloat2float_msa(ptr);
                _p = __ncnn_msa_fmadd_w(_b, _p, _a);
                __msa_storel_d(float2bfloat_msa(_p), ptr);
                ptr += 4;
            }
        }

        return 0;
    }
#endif // __mips_msa

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        // compute mean
        float mean = 0.f;
        {
            const unsigned short* ptr0 = ptr;
            int i = 0;
#if __mips_msa
            v4f32 _sum = (v4f32)__msa_fill_w(0);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr0 + i);
                _sum = __msa_fadd_w(_sum, _p);
            }
            mean += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa
            for (; i < size; i++)
            {
                mean += bfloat16_to_float32(ptr0[i]);
            }
            mean /= size;
        }

        // compute var
        float var = 0.f;
        {
            const unsigned short* ptr0 = ptr;
            int i = 0;
#if __mips_msa
            v4f32 _mean = __msa_fill_w_f32(mean);
            v4f32 _sqsum = (v4f32)__msa_fill_w(0);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr0 + i);
                _p = __msa_fsub_w(_p, _mean);
                _sqsum = __ncnn_msa_fmadd_w(_sqsum, _p, _p);
            }
            var += __msa_reduce_fadd_w(_sqsum);
#endif // __mips_msa
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr0[i]) - mean;
                var += v * v;
            }
            var /= size;
        }

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];
            a = gamma / sqrtf(var + eps);
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / sqrtf(var + eps);
            b = -mean * a;
        }

        // apply
        {
            int i = 0;
#if __mips_msa
            v4f32 _a = __msa_fill_w_f32(a);
            v4f32 _b = __msa_fill_w_f32(b);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = bfloat2float_msa(ptr + i);
                _p = __ncnn_msa_fmadd_w(_b, _p, _a);
                __msa_storel_d(float2bfloat_msa(_p), ptr + i);
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * a + b);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
