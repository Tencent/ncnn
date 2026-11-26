// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mvn_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include <math.h>

namespace ncnn {

MVN_powerpc::MVN_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

int MVN_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elembits() != 32)
    {
        // int8 / bf16 fall back to the generic implementation
        const int elempack = bottom_blob.elempack;
        if (elempack != 1)
        {
            Mat bottom_blob_unpacked;
            Option opt_unpacked = opt;
            opt_unpacked.blob_allocator = opt.workspace_allocator;

            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_unpacked);
            if (bottom_blob_unpacked.empty())
                return -100;

            Mat top_blob_unpacked;
            MVN::forward(bottom_blob_unpacked, top_blob_unpacked, opt_unpacked);
            if (top_blob_unpacked.empty())
                return -100;

            if (top_blob_unpacked.c % 4 == 0 && opt.use_packing_layout)
            {
                convert_packing(top_blob_unpacked, top_blob, 4, opt);
                if (top_blob.empty())
                    return -100;
            }
            else
            {
                top_blob = top_blob_unpacked;
            }
        }
        else
        {
            return MVN::forward(bottom_blob, top_blob, opt);
        }

        return 0;
    }

    const int dims_in = bottom_blob.dims;
    const int elempack_in = bottom_blob.elempack;
    if (dims_in <= 2 && elempack_in != 1)
    {
        // dims 1 / 2 packing is not per-channel lanes, fall back to the generic implementation
        Mat bottom_blob_unpacked;
        Option opt_unpacked = opt;
        opt_unpacked.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_unpacked);
        if (bottom_blob_unpacked.empty())
            return -100;

        Mat top_blob_unpacked;
        MVN::forward(bottom_blob_unpacked, top_blob_unpacked, opt_unpacked);
        if (top_blob_unpacked.empty())
            return -100;

        if (top_blob_unpacked.c % 4 == 0 && opt.use_packing_layout)
        {
            convert_packing(top_blob_unpacked, top_blob, 4, opt);
            if (top_blob.empty())
                return -100;
        }
        else
        {
            top_blob = top_blob_unpacked;
        }

        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    if (bottom_blob.dims == 4)
        top_blob.create(w, h, d, channels, elemsize, elempack, opt.blob_allocator);
    else
        top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // prepare sum per channel
    Mat sum(channels, elemsize, opt.workspace_allocator);
    if (sum.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);

        int i = 0;
        float s = 0.f;
#if __ALTIVEC__
        v4f32 _s = vec_splats(0.f);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            _s = vec_add(_s, _p);
            ptr += 4;
        }
        s += __ncnn_altivec_reduce_add(_s);
        ptr -= i;
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            s += ptr[i];
        }

        sum[q] = s;
    }

    if (across_channels)
    {
        // compute mean across channels
        float mean = 0.f;
        for (int q = 0; q < channels; q++)
        {
            mean += sum[q];
        }
        mean = mean / (channels * size);

        // subtract mean
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __ALTIVEC__
            v4f32 _mean = vec_splats(mean);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                _p = vec_sub(_p, _mean);
                __ncnn_altivec_store(outptr, _p);
                ptr += 4;
                outptr += 4;
            }
            ptr -= i;
            outptr -= i;
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                outptr[i] = ptr[i] - mean;
            }
        }
    }
    else
    {
        // subtract mean
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __ALTIVEC__
            if (elempack == 4 && bottom_blob.dims != 1)
            {
                // per-lane mean, each lane belongs to one channel
                v4f32 _s_lane = vec_splats(0.f);
                for (int j = 0; j + 3 < size; j += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr + j);
                    _s_lane = vec_add(_s_lane, _p);
                }
                // each lane accumulated one channel, so divide by the spatial count
                v4f32 _mean = __ncnn_altivec_div2(_s_lane, vec_splats((float)(size / elempack)));
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    _p = vec_sub(_p, _mean);
                    __ncnn_altivec_store(outptr, _p);
                    ptr += 4;
                    outptr += 4;
                }
                ptr -= i;
                outptr -= i;

                for (; i < size; i++)
                {
                    float mean_lane = vec_extract(_mean, i % elempack);
                    outptr[i] = ptr[i] - mean_lane;
                }

                continue;
            }
#endif // __ALTIVEC__
            float mean = sum[q] / size;

            for (; i < size; i++)
            {
                outptr[i] = ptr[i] - mean;
            }
        }
    }

    if (normalize_variance)
    {
        // prepare squared sum per channel
        Mat sqsum(channels, elemsize, opt.workspace_allocator);
        if (sqsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = top_blob.channel(q);

            int i = 0;
            float s = 0.f;
#if __ALTIVEC__
            v4f32 _s = vec_splats(0.f);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                _s = vec_madd(_p, _p, _s);
                ptr += 4;
            }
            s += __ncnn_altivec_reduce_add(_s);
            ptr -= i;
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                s += ptr[i] * ptr[i];
            }

            sqsum[q] = s;
        }

        if (across_channels)
        {
            // compute squared mean across channels
            float sqmean = 0.f;
            for (int q = 0; q < channels; q++)
            {
                sqmean += sqsum[q];
            }
            sqmean = sqmean / (channels * size);

            // normalize variance
            float norm_var = sqrtf(sqmean) + eps;
            float norm_var_inv = 1.f / norm_var;

            // apply normalize_variance
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ALTIVEC__
                v4f32 _norm_var_inv = vec_splats(norm_var_inv);
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(outptr);
                    _p = vec_madd(_p, _norm_var_inv, vec_splats(0.f));
                    __ncnn_altivec_store(outptr, _p);
                    outptr += 4;
                }
                outptr -= i;
#endif // __ALTIVEC__
                for (; i < size; i++)
                {
                    outptr[i] = outptr[i] * norm_var_inv;
                }
            }
        }
        else
        {
            // apply normalize_variance
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ALTIVEC__
                if (elempack == 4 && bottom_blob.dims != 1)
                {
                    // per-lane variance, each lane belongs to one channel
                    v4f32 _sq_lane = vec_splats(0.f);
                    for (int j = 0; j + 3 < size; j += 4)
                    {
                        v4f32 _p = __ncnn_altivec_load(outptr + j);
                        _sq_lane = vec_madd(_p, _p, _sq_lane);
                    }
                    // each lane accumulated one channel, so divide by the spatial count
                    v4f32 _sqmean = __ncnn_altivec_div2(_sq_lane, vec_splats((float)(size / elempack)));
                    v4f32 _norm_var = vec_add(__ncnn_altivec_sqrt2(_sqmean), vec_splats(eps));
                    v4f32 _norm_var_inv = __ncnn_altivec_recip2(_norm_var);
                    for (; i + 3 < size; i += 4)
                    {
                        v4f32 _p = __ncnn_altivec_load(outptr);
                        _p = vec_madd(_p, _norm_var_inv, vec_splats(0.f));
                        __ncnn_altivec_store(outptr, _p);
                        outptr += 4;
                    }
                    outptr -= i;

                    for (; i < size; i++)
                    {
                        float norm_var_inv_lane = vec_extract(_norm_var_inv, i % elempack);
                        outptr[i] = outptr[i] * norm_var_inv_lane;
                    }

                    continue;
                }
#endif // __ALTIVEC__
                float sqmean = sqsum[q] / size;
                float norm_var = sqrt(sqmean) + eps;
                float norm_var_inv = 1.f / norm_var;

                for (; i < size; i++)
                {
                    outptr[i] = outptr[i] * norm_var_inv;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
