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
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

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
                const float* cos_ptr = cos_cache.row(i);
                const float* sin_ptr = sin_cache.row(i);
                float* outptr = out_head.row(i);

                for (int j = 0; j < embed_dim / 2; j++)
                {
                    const float x0 = ptr[0];
                    const float x1 = ptr[1];
                    const float cos_val = *cos_ptr++;
                    const float sin_val = *sin_ptr++;
                    outptr[0] = x0 * cos_val - x1 * sin_val;
                    outptr[1] = x0 * sin_val + x1 * cos_val;
                    ptr += 2;
                    outptr += 2;
                }
            }
            else
            {
                const int half = embed_dim / 2;
                // A full-width (embed_dim) cache carries independent cos/sin for the two halves
                // (2D / vision rope where the halves differ); a half-width (embed_dim/2) cache is
                // the standard-rope form whose halves are identical, so the second half reuses the
                // first. This keeps existing half-width callers bit-identical.
                const int cw = cos_cache.w == embed_dim ? half : 0;
                const float* ptr0 = head.row(i);
                const float* ptr1 = ptr0 + half;
                const float* cos_ptr0 = cos_cache.row(i);
                const float* sin_ptr0 = sin_cache.row(i);
                const float* cos_ptr1 = cos_ptr0 + cw;
                const float* sin_ptr1 = sin_ptr0 + cw;
                float* outptr0 = out_head.row(i);
                float* outptr1 = outptr0 + half;

                for (int j = 0; j < half; j++)
                {
                    const float x0 = ptr0[j];
                    const float x1 = ptr1[j];
                    outptr0[j] = x0 * cos_ptr0[j] - x1 * sin_ptr0[j];
                    outptr1[j] = x0 * sin_ptr1[j] + x1 * cos_ptr1[j];
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
