// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed.h"

namespace ncnn {

RotaryEmbed::RotaryEmbed()
{
}

int RotaryEmbed::load_param(const ParamDict& pd)
{
    interleaved = pd.get(0, 0);

    return 0;
}

int RotaryEmbed::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // assert bottom_blobs.size() == 3

    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& sin_cache = bottom_blobs[1];
    const Mat& cos_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            if (interleaved)
            {
                const float* ptr = head.row(i);
                const float* sin_ptr = sin_cache.row(i);
                const float* cos_ptr = cos_cache.row(i);
                float* outptr = out_head.row(i);

                for (int j = 0; j < embed_dim / 2; j++)
                {
                    const float x1 = ptr[0];
                    const float x2 = ptr[1];
                    const float sin_val = *sin_ptr++;
                    const float cos_val = *cos_ptr++;
                    outptr[0] = x1 * cos_val - x2 * sin_val;
                    outptr[1] = x1 * sin_val + x2 * cos_val;
                    ptr += 2;
                    outptr += 2;
                }
            }
            else
            {
                const float* ptr1 = head.row(i);
                const float* ptr2 = ptr1 + embed_dim / 2;
                const float* sin_ptr = sin_cache.row(i);
                const float* cos_ptr = cos_cache.row(i);
                float* outptr1 = out_head.row(i);
                float* outptr2 = outptr1 + embed_dim / 2;

                for (int j = 0; j < embed_dim / 2; j++)
                {
                    const float x1 = *ptr1++;
                    const float x2 = *ptr2++;
                    const float sin_val = *sin_ptr++;
                    const float cos_val = *cos_ptr++;
                    *outptr1++ = x1 * cos_val - x2 * sin_val;
                    *outptr2++ = x1 * sin_val + x2 * cos_val;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
