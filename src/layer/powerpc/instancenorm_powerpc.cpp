// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "instancenorm_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include <math.h>

namespace ncnn {

InstanceNorm_powerpc::InstanceNorm_powerpc()
{
}

int InstanceNorm_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (bottom_top_blob.elembits() != 32)
    {
        // int8 / bf16 fall back to the generic implementation
        return InstanceNorm::forward_inplace(bottom_top_blob, opt);
    }

    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int size = w * h * d;

#if __ALTIVEC__
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        const float* ptr0 = ptr;

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;

        int i = 0;
        v4f32 _sum = vec_splats(0.f);
        for (; i + 4 <= size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr0);
            _sum = vec_add(_sum, _p);
            ptr0 += 4;
        }
        sum += __ncnn_altivec_reduce_add(_sum);
        for (; i < size; i++)
        {
            sum += ptr0[0];
            ptr0++;
        }

        float mean = sum / size;
        float tmp = 0.f;

        ptr0 = ptr;
        i = 0;
        v4f32 _sqsum = vec_splats(0.f);
        v4f32 _mean = vec_splats(mean);
        for (; i + 4 <= size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr0);
            _p = vec_sub(_p, _mean);
            _sqsum = vec_madd(_p, _p, _sqsum);
            ptr0 += 4;
        }
        sqsum += __ncnn_altivec_reduce_add(_sqsum);
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
        v4f32 _a = vec_splats(a);
        v4f32 _b = vec_splats(b);
        for (; i + 4 <= size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            _p = vec_madd(_p, _a, _b);
            __ncnn_altivec_store(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * a + b;
            ptr++;
        }
    }
#else
    return InstanceNorm::forward_inplace(bottom_top_blob, opt);
#endif // __ALTIVEC__

    return 0;
}

} // namespace ncnn
