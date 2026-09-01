// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "instancenorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__
#include "x86_usability.h"
#include "cpu.h"

namespace ncnn {

#include "instancenorm_fp32.h"

#if NCNN_BF16
#include "instancenorm_bf16s.h"
#endif

InstanceNorm_x86::InstanceNorm_x86()
{
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int InstanceNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    instancenorm_fp32(bottom_top_blob, eps, affine, gamma_data, beta_data, opt);

    return 0;
}

#if NCNN_BF16
int InstanceNorm_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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

        // compute mean and var
        float mean = 0.f;
        float var = 0.f;
        instancenorm_bf16s_compute_mean_var(ptr, size, mean, var);

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

        instancenorm_bf16s_sse(ptr, size, a, b);
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
