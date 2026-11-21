// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

PReLU_powerpc::PReLU_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

int PReLU_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

            int ret = PReLU::forward_inplace(bottom_top_blob_unpacked, opt_unpacked);
            if (ret != 0)
                return ret;

            if (bottom_top_blob_unpacked.c % 4 == 0 && opt.use_packing_layout)
            {
                convert_packing(bottom_top_blob_unpacked, bottom_top_blob, 4, opt);
                if (bottom_top_blob.empty())
                    return -100;
            }

            return 0;
        }

        return PReLU::forward_inplace(bottom_top_blob, opt);
    }

    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        int w = bottom_top_blob.w * elempack;

        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;

#if __ALTIVEC__
            const int nn_w = w / 4 * 4;
#else
            const int nn_w = 0;
#endif // __ALTIVEC__

#if __ALTIVEC__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < nn_w; i += 4)
            {
                float* ptr0 = ptr + i;

                v4f32 _p = __ncnn_altivec_load(ptr0);
                v4f32 _zero = vec_splats(0.f);
                v4f32 _slope = __ncnn_altivec_load(slope + i);
                v4i32 _lemask = (v4i32)vec_cmpgt(_zero, _p);
                v4f32 _ps = __ncnn_altivec_mul(_p, _slope);
                _p = __ncnn_altivec_sel(_lemask, _ps, _p);
                __ncnn_altivec_store(ptr0, _p);
            }
#endif // __ALTIVEC__

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = nn_w; i < w; i++)
            {
                float v = ptr[i];
                if (v < 0.f)
                    ptr[i] = v * slope[i];
            }
        }
        else
        {
            const float slope = slope_data[0];

#if __ALTIVEC__
            const int nn_w = w / 4 * 4;
#else
            const int nn_w = 0;
#endif // __ALTIVEC__

#if __ALTIVEC__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < nn_w; i += 4)
            {
                float* ptr0 = ptr + i;

                v4f32 _p = __ncnn_altivec_load(ptr0);
                v4f32 _zero = vec_splats(0.f);
                v4f32 _slope = vec_splats(slope);
                v4i32 _lemask = (v4i32)vec_cmpgt(_zero, _p);
                v4f32 _ps = __ncnn_altivec_mul(_p, _slope);
                _p = __ncnn_altivec_sel(_lemask, _ps, _p);
                __ncnn_altivec_store(ptr0, _p);
            }
#endif // __ALTIVEC__

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = nn_w; i < w; i++)
            {
                float v = ptr[i];
                if (v < 0.f)
                    ptr[i] = v * slope;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w * elempack;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            const float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            int j = 0;
#if __ALTIVEC__
            v4f32 _zero = vec_splats(0.f);
            v4f32 _slope = (elempack == 4 && num_slope > 1) ? __ncnn_altivec_load((const float*)slope_data + i * 4) : vec_splats(slope);

            for (; j + 3 < w; j += 4)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = __ncnn_altivec_load(ptr);
                v4i32 _lemask = (v4i32)vec_cmpgt(_zero, _p);
                v4f32 _ps = __ncnn_altivec_mul(_p, _slope);
                _p = __ncnn_altivec_sel(_lemask, _ps, _p);
                __ncnn_altivec_store(ptr, _p);

                ptr += 4;
            }
#endif // __ALTIVEC__
            for (; j < w; j++)
            {
                float v = *ptr;
                if (v < 0.f)
                {
                    const float s = (num_slope > 1 && elempack == 4) ? slope_data[i * 4 + j % 4] : slope;
                    *ptr = v * s;
                }

                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d * elempack;

        const float* slope_data_ptr = slope_data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

            int i = 0;
#if __ALTIVEC__
            v4f32 _zero = vec_splats(0.f);
            v4f32 _slope = (elempack == 4 && num_slope > 1) ? __ncnn_altivec_load((const float*)slope_data + q * 4) : vec_splats(slope);

            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = __ncnn_altivec_load(ptr);
                v4i32 _lemask = (v4i32)vec_cmpgt(_zero, _p);
                v4f32 _ps = __ncnn_altivec_mul(_p, _slope);
                _p = __ncnn_altivec_sel(_lemask, _ps, _p);
                __ncnn_altivec_store(ptr, _p);

                ptr += 4;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                if (*ptr < 0)
                {
                    const float s = (num_slope > 1 && elempack == 4) ? slope_data_ptr[q * 4 + i % 4] : slope;
                    *ptr *= s;
                }

                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
