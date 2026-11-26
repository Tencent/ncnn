// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "shufflechannel_powerpc.h"

#include <string.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

ShuffleChannel_powerpc::ShuffleChannel_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

int ShuffleChannel_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h * d;

    int logical_channels = channels * elempack;
    if (logical_channels % group != 0)
        return -100;

    int _group = reverse ? logical_channels / group : group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (elempack == 1)
        return ShuffleChannel::forward(bottom_blob, top_blob, opt);

#if __ALTIVEC__
    if (elempack == 4 && elemsize == 16u)
    {
        if (channels % _group == 0)
        {
            int channels_per_group = channels / _group;

            if (_group == 4)
            {
                top_blob.create_like(bottom_blob, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels_per_group; q++)
                {
                    const float* ptr0 = bottom_blob.channel(q);
                    const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                    const float* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                    const float* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                    float* outptr0 = top_blob.channel(q * 4);
                    float* outptr1 = top_blob.channel(q * 4 + 1);
                    float* outptr2 = top_blob.channel(q * 4 + 2);
                    float* outptr3 = top_blob.channel(q * 4 + 3);

                    for (int i = 0; i < size; i++)
                    {
                        v4f32 _p0 = __ncnn_altivec_load(ptr0);
                        v4f32 _p1 = __ncnn_altivec_load(ptr1);
                        v4f32 _p2 = __ncnn_altivec_load(ptr2);
                        v4f32 _p3 = __ncnn_altivec_load(ptr3);

                        transpose4x4_ps(_p0, _p1, _p2, _p3);

                        __ncnn_altivec_store(outptr0, _p0);
                        __ncnn_altivec_store(outptr1, _p1);
                        __ncnn_altivec_store(outptr2, _p2);
                        __ncnn_altivec_store(outptr3, _p3);

                        ptr0 += 4;
                        ptr1 += 4;
                        ptr2 += 4;
                        ptr3 += 4;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
                }

                return 0;
            }
        }
    }
#endif // __ALTIVEC__

    int channels_per_group = logical_channels / _group;
    size_t lane_size = elemsize / elempack;

    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    for (int i = 0; i < _group; i++)
    {
        for (int j = 0; j < channels_per_group; j++)
        {
            int src_c = channels_per_group * i + j;
            int dst_c = _group * j + i;

            int src_q = src_c / elempack;
            int src_lane = src_c % elempack;
            int dst_q = dst_c / elempack;
            int dst_lane = dst_c % elempack;

            const unsigned char* ptr = bottom_blob.channel(src_q);
            unsigned char* outptr = top_blob.channel(dst_q);

            ptr += src_lane * lane_size;
            outptr += dst_lane * lane_size;

            for (int k = 0; k < size; k++)
            {
                memcpy(outptr, ptr, lane_size);

                ptr += elemsize;
                outptr += elemsize;
            }
        }
    }

    return 0;
}

} // namespace ncnn
