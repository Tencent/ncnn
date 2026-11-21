// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rmsnorm_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include <math.h>

namespace ncnn {

RMSNorm_powerpc::RMSNorm_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
static void rmsnorm_powerpc(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // compute rms
    v4f32 _sumsq = vec_splats(0.f);
    float sumsq = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
        for (; i + 4 <= size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

            v4f32 _p = __ncnn_altivec_load(ptr0);
            _sumsq = vec_madd(_p, _p, _sumsq);
            ptr0 += 4;
        }
        sumsq += __ncnn_altivec_reduce_add(_sumsq);
        for (; i < size; i++)
        {
            sumsq += ptr0[0] * ptr0[0];
            ptr0++;
        }
    }

    v4f32 _rms = vec_splats(0.f);
    float rms = 1.f;
    if (elempack == 4)
    {
        // per-lane mean, each lane belongs to one channel
        v4f32 _elemcount = vec_splats((float)elemcount);
        v4f32 _eps = vec_splats(eps);
        _rms = __ncnn_altivec_div2(_sumsq, _elemcount);
        _rms = vec_add(_rms, _eps);
        _rms = __ncnn_altivec_rsqrt2(_rms);
    }
    if (elempack == 1)
    {
        rms = 1.f / sqrtf(sumsq / elemcount + eps);
        _rms = vec_splats(rms);
    }
    if (gamma_ptr)
    {
        int i = 0;
        if (elempack == 4)
        {
            for (; i + 4 <= size; i += 4)
            {
                __builtin_prefetch(ptr + 16);

                v4f32 _p = __ncnn_altivec_load(ptr);
                v4f32 _gamma = vec_splats(gamma_ptr[0]);
                _p = __ncnn_altivec_mul(_p, _rms);
                _p = __ncnn_altivec_mul(_p, _gamma);
                __ncnn_altivec_store(ptr, _p);
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 4 <= size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __builtin_prefetch(gamma_ptr + 16);

                v4f32 _p = __ncnn_altivec_load(ptr);
                v4f32 _gamma = __ncnn_altivec_load(gamma_ptr);
                _p = __ncnn_altivec_mul(_p, _rms);
                _p = __ncnn_altivec_mul(_p, _gamma);
                __ncnn_altivec_store(ptr, _p);
                ptr += 4;
                gamma_ptr += 4;
            }
        }
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * rms) * gamma_ptr[0];
            ptr++;
            gamma_ptr++;
        }
    }
    else
    {
        int i = 0;
        for (; i + 4 <= size; i += 4)
        {
            __builtin_prefetch(ptr + 16);

            v4f32 _p = __ncnn_altivec_load(ptr);
            _p = __ncnn_altivec_mul(_p, _rms);
            __ncnn_altivec_store(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * rms;
            ptr++;
        }
    }
}
#endif // __ALTIVEC__

int RMSNorm_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (bottom_top_blob.elembits() != 32)
    {
        // int8 / bf16 fall back to the generic implementation
        const int elempack = bottom_top_blob.elempack;
        if (elempack != 1)
        {
            Mat bottom_top_blob_unpacked;
            Option opt_unpacked = opt;
            opt_unpacked.blob_allocator = opt.workspace_allocator;

            convert_packing(bottom_top_blob, bottom_top_blob_unpacked, 1, opt_unpacked);
            if (bottom_top_blob_unpacked.empty())
                return -100;

            RMSNorm::forward_inplace(bottom_top_blob_unpacked, opt_unpacked);

            if (bottom_top_blob_unpacked.c % 4 == 0 && opt.use_packing_layout)
            {
                convert_packing(bottom_top_blob_unpacked, bottom_top_blob, 4, opt);
                if (bottom_top_blob.empty())
                    return -100;
            }
        }
        else
        {
            return RMSNorm::forward_inplace(bottom_top_blob, opt);
        }

        return 0;
    }

#if __ALTIVEC__
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        rmsnorm_powerpc(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            rmsnorm_powerpc(ptr, gamma_data, eps, w, elempack);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    rmsnorm_powerpc(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm_powerpc(ptr, gamma_data, eps, w * h, elempack);
            }
        }
    }

    if (dims == 4)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).depth(z).row(i);
                        rmsnorm_powerpc(ptr, gamma_data, eps, w, elempack);
                    }
                }
            }
        }
        else if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    float* ptr = bottom_top_blob.channel(q).depth(z);
                    rmsnorm_powerpc(ptr, gamma_data, eps, w * h, elempack);
                }
            }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm_powerpc(ptr, gamma_data, eps, w * h * d, elempack);
            }
        }
    }

    return 0;
#else
    return RMSNorm::forward_inplace(bottom_top_blob, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
