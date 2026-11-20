// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include <math.h>

namespace ncnn {

Normalize_powerpc::Normalize_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

int Normalize_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

            Normalize::forward_inplace(bottom_top_blob_unpacked, opt_unpacked);

            if (bottom_top_blob_unpacked.c % 4 == 0 && opt.use_packing_layout)
            {
                convert_packing(bottom_top_blob_unpacked, bottom_top_blob, 4, opt);
                if (bottom_top_blob.empty())
                    return -100;
            }
        }
        else
        {
            return Normalize::forward_inplace(bottom_top_blob, opt);
        }

        return 0;
    }

    const int dims = bottom_top_blob.dims;
    const int elempack_in = bottom_top_blob.elempack;
    if (dims <= 2 && elempack_in != 1)
    {
        // dims 1 / 2 packing is not per-channel lanes, fall back to the generic implementation
        Mat bottom_top_blob_unpacked;
        Option opt_unpacked = opt;
        opt_unpacked.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_top_blob, bottom_top_blob_unpacked, 1, opt_unpacked);
        if (bottom_top_blob_unpacked.empty())
            return -100;

        Normalize::forward_inplace(bottom_top_blob_unpacked, opt_unpacked);

        if (bottom_top_blob_unpacked.c % 4 == 0 && opt.use_packing_layout)
        {
            convert_packing(bottom_top_blob_unpacked, bottom_top_blob, 4, opt);
            if (bottom_top_blob.empty())
                return -100;
        }
        else
        {
            bottom_top_blob = bottom_top_blob_unpacked;
        }

        return 0;
    }

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (across_spatial && across_channel)
    {
        // square
        Mat square_sum_blob;
        square_sum_blob.create(channels, elemsize, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);

            int i = 0;
            float ssum = 0.f;
#if __ALTIVEC__
            v4f32 _ssum = vec_splats(0.f);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                _ssum = vec_madd(_p, _p, _ssum);
                ptr += 4;
            }
            ssum += __ncnn_altivec_reduce_add(_ssum);
            ptr -= i;
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                ssum += ptr[i] * ptr[i];
            }

            square_sum_blob[q] = ssum;
        }

        float ssum = 0.f;
        for (int q = 0; q < channels; q++)
        {
            ssum += square_sum_blob[q];
        }

        float a;
        if (eps_mode == 0) // caffe/mxnet
        {
            a = 1.f / sqrtf(ssum + eps);
        }
        else if (eps_mode == 1) // pytorch
        {
            a = 1.f / std::max(sqrtf(ssum), eps);
        }
        else //if (eps_mode == 2) // tensorflow
        {
            a = 1.f / sqrt(std::max(ssum, eps));
        }

        if (channel_shared)
        {
            float scale = a * scale_data[0];
#if __ALTIVEC__
            v4f32 _scale;
            if (elempack == 4)
                _scale = vec_splats(scale);
#endif

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                int i = 0;
#if __ALTIVEC__
                if (elempack == 4)
                {
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = __ncnn_altivec_load(ptr);
                        _p = vec_madd(_p, _scale, vec_splats(0.f));
                        __ncnn_altivec_store(ptr, _p);
                        ptr += 4;
                    }
                    ptr -= i;
                }
#endif // __ALTIVEC__
                for (; i < size; i++)
                {
                    ptr[i] = ptr[i] * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float scale = a * scale_data[q];

                int i = 0;
#if __ALTIVEC__
                // one scale per channel, broadcast to lanes when packed
                v4f32 _scale;
                if (elempack == 4)
                {
                    float scale_lanes[4];
                    for (int k = 0; k < 4; k++)
                        scale_lanes[k] = a * scale_data[q * elempack + k];
                    _scale = __ncnn_altivec_load(scale_lanes);
                }
                if (elempack == 4)
                {
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = __ncnn_altivec_load(ptr);
                        _p = vec_madd(_p, _scale, vec_splats(0.f));
                        __ncnn_altivec_store(ptr, _p);
                        ptr += 4;
                    }
                    ptr -= i;
                }
#endif // __ALTIVEC__
                for (; i < size; i++)
                {
                    ptr[i] = ptr[i] * scale;
                }
            }
        }

        return 0;
    }

    if (across_spatial && !across_channel)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
            float ssum = 0.f;
#if __ALTIVEC__
            if (elempack == 4 && dims != 1)
            {
                // per-lane ssum, each lane belongs to one channel
                v4f32 _ssum = vec_splats(0.f);
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    _ssum = vec_madd(_p, _p, _ssum);
                    ptr += 4;
                }
                ptr -= i;

                // per-lane 1 / sqrt(ssum + eps)
                v4f32 _eps = vec_splats(eps);
                v4f32 _a;
                if (eps_mode == 0) // caffe/mxnet
                {
                    _a = __ncnn_altivec_rsqrt2(vec_add(_ssum, _eps));
                }
                else if (eps_mode == 1) // pytorch
                {
                    v4f32 _t = vec_max(__ncnn_altivec_sqrt2(_ssum), _eps);
                    _a = __ncnn_altivec_recip2(_t);
                }
                else //if (eps_mode == 2) // tensorflow
                {
                    _a = __ncnn_altivec_rsqrt2(vec_max(_ssum, _eps));
                }

                // one scale per channel
                v4f32 _scale;
                {
                    float scale_lanes[4];
                    for (int k = 0; k < 4; k++)
                    {
                        float a_k = vec_extract(_a, k);
                        scale_lanes[k] = a_k * (channel_shared ? scale_data[0] : scale_data[q * elempack + k]);
                    }
                    _scale = __ncnn_altivec_load(scale_lanes);
                }

                for (i = 0; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    _p = vec_madd(_p, _scale, vec_splats(0.f));
                    __ncnn_altivec_store(ptr, _p);
                    ptr += 4;
                }
                ptr -= i;

                for (; i < size; i++)
                {
                    // flat index i: lane k = i % elempack
                    int k = i % elempack;
                    float a_k = vec_extract(_a, k);
                    float scale = a_k * (channel_shared ? scale_data[0] : scale_data[q * elempack + k]);
                    ptr[i] = ptr[i] * scale;
                }

                continue;
            }
            ssum = 0.f;
            i = 0;
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                ssum += ptr[i] * ptr[i];
            }

            float a;
            if (eps_mode == 0) // caffe/mxnet
            {
                a = 1.f / sqrtf(ssum + eps);
            }
            else if (eps_mode == 1) // pytorch
            {
                a = 1.f / std::max(sqrtf(ssum), eps);
            }
            else //if (eps_mode == 2) // tensorflow
            {
                a = 1.f / sqrtf(std::max(ssum, eps));
            }

            float scale = a * (channel_shared ? scale_data[0] : scale_data[q]);

            for (i = 0; i < size; i++)
            {
                ptr[i] = ptr[i] * scale;
            }
        }

        return 0;
    }

    if (!across_spatial && across_channel)
    {
        // square sum, 1 / sqrt(ssum)
        Mat square_sum_blob;
        square_sum_blob.create(size, elemsize, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

#if __ALTIVEC__
        if (elempack == 4 && dims != 1)
        {
            // each pack of 4 flat elements is one position across 4 channels
            // ssum per position must accumulate over all packs and all lanes
            const int npos = size / elempack;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int pos = 0; pos < npos; pos++)
            {
                v4f32 _ssum = vec_splats(0.f);
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_top_blob.channel(q);
                    v4f32 _p = __ncnn_altivec_load(ptr + pos * 4);
                    _ssum = vec_madd(_p, _p, _ssum);
                }

                float ssum = __ncnn_altivec_reduce_add(_ssum);

                float a;
                if (eps_mode == 0) // caffe/mxnet
                {
                    a = 1.f / sqrtf(ssum + eps);
                }
                else if (eps_mode == 1) // pytorch
                {
                    a = 1.f / std::max((float)sqrt(ssum), eps);
                }
                else //if (eps_mode == 2) // tensorflow
                {
                    a = 1.f / sqrtf(std::max(ssum, eps));
                }

                if (channel_shared)
                {
                    __ncnn_altivec_store((float*)square_sum_blob + pos * 4, vec_splats(a * scale_data[0]));
                }
                else
                {
                    __ncnn_altivec_store((float*)square_sum_blob + pos * 4, vec_splats(a));
                }
            }
        }
        else
#endif // __ALTIVEC__
        {
            if (channel_shared)
            {
                float scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < size; i++)
                {
                    float ssum = 0.f;
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = bottom_top_blob.channel(q);
                        ssum += ptr[i] * ptr[i];
                    }

                    float a;
                    if (eps_mode == 0) // caffe/mxnet
                    {
                        a = 1.f / sqrtf(ssum + eps);
                    }
                    else if (eps_mode == 1) // pytorch
                    {
                        a = 1.f / std::max((float)sqrt(ssum), eps);
                    }
                    else //if (eps_mode == 2) // tensorflow
                    {
                        a = 1.f / sqrtf(std::max(ssum, eps));
                    }

                    square_sum_blob[i] = a * scale;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < size; i++)
                {
                    float ssum = 0.f;
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = bottom_top_blob.channel(q);
                        ssum += ptr[i] * ptr[i];
                    }

                    float a;
                    if (eps_mode == 0) // caffe/mxnet
                    {
                        a = 1.f / sqrtf(ssum + eps);
                    }
                    else if (eps_mode == 1) // pytorch
                    {
                        a = 1.f / std::max(sqrtf(ssum), eps);
                    }
                    else //if (eps_mode == 2) // tensorflow
                    {
                        a = 1.f / sqrtf(std::max(ssum, eps));
                    }

                    square_sum_blob[i] = a;
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
            const float* ssptr = square_sum_blob;
#if __ALTIVEC__
            if (elempack == 4 && dims != 1 && !channel_shared)
            {
                // one scale per channel, broadcast to lanes
                float scale_lanes[4];
                for (int k = 0; k < 4; k++)
                    scale_lanes[k] = scale_data[q * elempack + k];
                v4f32 _scale = __ncnn_altivec_load(scale_lanes);

                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    v4f32 _s = __ncnn_altivec_load(ssptr + i);
                    _p = vec_madd(_p, _s, vec_splats(0.f));
                    _p = vec_madd(_p, _scale, vec_splats(0.f));
                    __ncnn_altivec_store(ptr, _p);
                    ptr += 4;
                }
                ptr -= i;

                for (; i < size; i++)
                {
                    float scale = scale_data[q * elempack + i % elempack];
                    ptr[i] = ptr[i] * square_sum_blob[i] * scale;
                }

                continue;
            }
            float scale = channel_shared ? 1.f : scale_data[q];
#else
            float scale = channel_shared ? 1.f : scale_data[q];
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                ptr[i] = ptr[i] * square_sum_blob[i] * scale;
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
