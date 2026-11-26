// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include <math.h>

namespace ncnn {

LayerNorm_powerpc::LayerNorm_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
static void layernorm_powerpc(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // compute mean
    v4f32 _sum = vec_splats(0.f);
    float sum = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
        for (; i + 4 <= size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

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
    }

    v4f32 _mean = vec_splats(0.f);
    float mean = sum;
    if (elempack == 4)
    {
        // per-lane mean, each lane belongs to one channel
        v4f32 _elemcount = vec_splats((float)elemcount);
        _mean = __ncnn_altivec_div2(_sum, _elemcount);
        mean = sum;
    }
    if (elempack == 1)
    {
        mean = sum / elemcount;
        _mean = vec_splats(mean);
    }

    // compute variance
    v4f32 _varsum = vec_splats(0.f);
    float varsum = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
        for (; i + 4 <= size; i += 4)
        {
            __builtin_prefetch(ptr0 + 16);

            v4f32 _p = __ncnn_altivec_load(ptr0);
            _p = vec_sub(_p, _mean);
            _varsum = vec_madd(_p, _p, _varsum);
            ptr0 += 4;
        }
        varsum += __ncnn_altivec_reduce_add(_varsum);
        for (; i < size; i++)
        {
            float v = ptr0[0] - mean;
            varsum += v * v;
            ptr0++;
        }
    }

    v4f32 _var = vec_splats(0.f);
    float var = varsum;
    if (elempack == 4)
    {
        v4f32 _elemcount = vec_splats((float)elemcount);
        v4f32 _eps = vec_splats(eps);
        _var = __ncnn_altivec_div2(_varsum, _elemcount);
        _var = vec_add(_var, _eps);
        _var = __ncnn_altivec_rsqrt2(_var);
        _mean = __ncnn_altivec_mul(_mean, _var);
        _mean = vec_sub(vec_splats(0.f), _mean);
        var = varsum;
    }
    if (elempack == 1)
    {
        var = 1.f / sqrtf(varsum / elemcount + eps);
        mean = -mean * var;
        _var = vec_splats(var);
        _mean = vec_splats(mean);
    }

    if (gamma_ptr && beta_ptr)
    {
        int i = 0;
        if (elempack == 4)
        {
            for (; i + 4 <= size; i += 4)
            {
                __builtin_prefetch(ptr + 16);

                v4f32 _p = __ncnn_altivec_load(ptr);
                v4f32 _gamma = vec_splats(gamma_ptr[0]);
                v4f32 _beta = vec_splats(beta_ptr[0]);
                _p = vec_madd(_p, _var, _mean);
                _p = vec_madd(_p, _gamma, _beta);
                __ncnn_altivec_store(ptr, _p);
                ptr += 4;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 4 <= size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __builtin_prefetch(gamma_ptr + 16);
                __builtin_prefetch(beta_ptr + 16);

                v4f32 _p = __ncnn_altivec_load(ptr);
                v4f32 _gamma = __ncnn_altivec_load(gamma_ptr);
                v4f32 _beta = __ncnn_altivec_load(beta_ptr);
                _p = vec_madd(_p, _var, _mean);
                _p = vec_madd(_p, _gamma, _beta);
                __ncnn_altivec_store(ptr, _p);
                ptr += 4;
                gamma_ptr += 4;
                beta_ptr += 4;
            }
        }
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * var + mean) * gamma_ptr[0] + beta_ptr[0];
            ptr++;
            gamma_ptr++;
            beta_ptr++;
        }
    }
    else
    {
        int i = 0;
        for (; i + 4 <= size; i += 4)
        {
            __builtin_prefetch(ptr + 16);

            v4f32 _p = __ncnn_altivec_load(ptr);
            _p = vec_madd(_p, _var, _mean);
            __ncnn_altivec_store(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * var + mean;
            ptr++;
        }
    }
}
#endif // __ALTIVEC__

int LayerNorm_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

            LayerNorm::forward_inplace(bottom_top_blob_unpacked, opt_unpacked);

            if (bottom_top_blob_unpacked.c % 4 == 0 && opt.use_packing_layout)
            {
                convert_packing(bottom_top_blob_unpacked, bottom_top_blob, 4, opt);
                if (bottom_top_blob.empty())
                    return -100;
            }
        }
        else
        {
            return LayerNorm::forward_inplace(bottom_top_blob, opt);
        }

        return 0;
    }

#if __ALTIVEC__
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        layernorm_powerpc(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            layernorm_powerpc(ptr, gamma_data, beta_data, eps, w, elempack);
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
                    layernorm_powerpc(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm_powerpc(ptr, gamma_data, beta_data, eps, w * h, elempack);
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
                        layernorm_powerpc(ptr, gamma_data, beta_data, eps, w, elempack);
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
                    layernorm_powerpc(ptr, gamma_data, beta_data, eps, w * h, elempack);
                }
            }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm_powerpc(ptr, gamma_data, beta_data, eps, w * h * d, elempack);
            }
        }
    }

    return 0;
#else
    return LayerNorm::forward_inplace(bottom_top_blob, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
