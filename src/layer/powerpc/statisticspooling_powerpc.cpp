// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "statisticspooling_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

StatisticsPooling_powerpc::StatisticsPooling_powerpc()
{
    // base defaults kept: one_blob_only = true, support_inplace = false
    // support_packing stays false, only pack1 fp32 is vectorized
}

int StatisticsPooling_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __ALTIVEC__
    if (bottom_blob.elempack == 1 && bottom_blob.elemsize == 4u)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;
        int size = w * h * d;
        size_t elemsize = bottom_blob.elemsize;

        int out_channels = channels;
        if (include_stddev)
        {
            out_channels *= 2;
        }

        top_blob.create(out_channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);

            int i = 0;
            v4f32 _sum = vec_splats(0.f);
            for (; i + 4 <= size; i += 4)
            {
                _sum = vec_add(_sum, __ncnn_altivec_load(ptr + i));
            }
            float mean = __ncnn_altivec_reduce_add(_sum);
            for (; i < size; i++)
            {
                mean += ptr[i];
            }
            top_blob[q] = mean / size;
        }

        if (include_stddev)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = channels; q < out_channels; q++)
            {
                const float* ptr = bottom_blob.channel(q - channels);

                v4f32 _mean = vec_splats(top_blob[q - channels]);

                int i = 0;
                v4f32 _sum = vec_splats(0.f);
                for (; i + 4 <= size; i += 4)
                {
                    v4f32 _v = vec_sub(__ncnn_altivec_load(ptr + i), _mean);
                    _sum = vec_add(_sum, __ncnn_altivec_mul(_v, _v));
                }
                float std = __ncnn_altivec_reduce_add(_sum);
                for (; i < size; i++)
                {
                    float v = ptr[i] - top_blob[q - channels];
                    std += v * v;
                }
                top_blob[q] = sqrtf(std / size);
            }
        }

        return 0;
    }
#endif // __ALTIVEC__

    return StatisticsPooling::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
