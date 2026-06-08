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

    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
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
            sum += ptr0[0];
            ptr0++;
        }

        float mean = sum / size;
        float tmp = 0.f;

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
            tmp = ptr0[0] - mean;
            sqsum += tmp * tmp;
            ptr0++;
        }

        float var = sqsum / size;
        // the var maybe minus due to accuracy

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = gamma / (sqrtf(var + eps));
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / (sqrtf(var + eps));
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
            ptr[0] = ptr[0] * a + b;
            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int InstanceNorm_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);
        const unsigned short* ptr0 = ptr;

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;

        int i = 0;
#if __mips_msa
        v4f32 _sum = (v4f32)__msa_fill_w(0);
        v8i16 _zero_bf16 = __msa_fill_h(0);
        for (; i + 7 < size; i += 8)
        {
            v8i16 _p01 = __msa_ld_h(ptr0, 0);
            v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
            v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
            _sum = __msa_fadd_w(_sum, _p0);
            _sum = __msa_fadd_w(_sum, _p1);
            ptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr0);
            _sum = __msa_fadd_w(_sum, _p);
            ptr0 += 4;
        }
        sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa
        for (; i < size; i++)
        {
            sum += bfloat16_to_float32(ptr0[0]);
            ptr0++;
        }

        float mean = sum / size;
        float tmp = 0.f;

        ptr0 = ptr;
        i = 0;
#if __mips_msa
        v4f32 _sqsum = (v4f32)__msa_fill_w(0);
        v4f32 _mean = __msa_fill_w_f32(mean);
        for (; i + 7 < size; i += 8)
        {
            v8i16 _p01 = __msa_ld_h(ptr0, 0);
            v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
            v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
            _p0 = __msa_fsub_w(_p0, _mean);
            _p1 = __msa_fsub_w(_p1, _mean);
            _sqsum = __ncnn_msa_fmadd_w(_sqsum, _p0, _p0);
            _sqsum = __ncnn_msa_fmadd_w(_sqsum, _p1, _p1);
            ptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr0);
            _p = __msa_fsub_w(_p, _mean);
            _sqsum = __ncnn_msa_fmadd_w(_sqsum, _p, _p);
            ptr0 += 4;
        }
        sqsum += __msa_reduce_fadd_w(_sqsum);
#endif // __mips_msa
        for (; i < size; i++)
        {
            tmp = bfloat16_to_float32(ptr0[0]) - mean;
            sqsum += tmp * tmp;
            ptr0++;
        }

        float var = sqsum / size;
        // the var maybe minus due to accuracy

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = gamma / (sqrtf(var + eps));
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / (sqrtf(var + eps));
            b = -mean * a;
        }

        i = 0;
#if __mips_msa
        v4f32 _a = __msa_fill_w_f32(a);
        v4f32 _b = __msa_fill_w_f32(b);
        for (; i + 7 < size; i += 8)
        {
            v8i16 _p01 = __msa_ld_h(ptr, 0);
            v4f32 _p0 = (v4f32)__msa_ilvr_h(_p01, _zero_bf16);
            v4f32 _p1 = (v4f32)__msa_ilvl_h(_p01, _zero_bf16);
            _p0 = __ncnn_msa_fmadd_w(_b, _p0, _a);
            _p1 = __ncnn_msa_fmadd_w(_b, _p1, _a);
            __msa_st_w(float2bfloat_msa(_p0, _p1), ptr, 0);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            _p = __ncnn_msa_fmadd_w(_b, _p, _a);
            *(int64_t*)ptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_p), 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            ptr[0] = float32_to_bfloat16(bfloat16_to_float32(ptr[0]) * a + b);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
