// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "instancenorm_mips.h"

#if __mips_msa
#include <msa.h>
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

#if __mips_msa
    if (bottom_top_blob.elempack == 4)
    {
        const int channels = bottom_top_blob.c;
        const int size = bottom_top_blob.w * bottom_top_blob.h;

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
                _sqsum = __msa_fmadd_w(_sqsum, _p, _p);
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
                _p = __msa_fmadd_w(_b, _p, _a);
                __msa_st_w((v4i32)_p, ptr, 0);
                ptr += 4;
            }
        }

        return 0;
    }
#endif // __mips_msa

    return InstanceNorm::forward_inplace(bottom_top_blob, opt);
}

#if NCNN_BF16
int InstanceNorm_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    Option opt_cast = opt;
    opt_cast.blob_allocator = opt.workspace_allocator;

    Mat bottom_top_blob_fp32;
    cast_bfloat16_to_float32(bottom_top_blob, bottom_top_blob_fp32, opt_cast);
    if (bottom_top_blob_fp32.empty())
        return -100;

    int ret = forward_inplace(bottom_top_blob_fp32, opt);
    if (ret != 0)
        return ret;

    cast_float32_to_bfloat16(bottom_top_blob_fp32, bottom_top_blob, opt);
    if (bottom_top_blob.empty())
        return -100;

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
