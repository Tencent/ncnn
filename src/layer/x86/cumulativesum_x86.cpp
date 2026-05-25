// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_x86.h"

#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#include "cumulativesum_x86_helper.h"

CumulativeSum_x86::CumulativeSum_x86()
{
}

int CumulativeSum_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1)
    {
        cumulative_sum_prefix_sum_row(bottom_top_blob, bottom_top_blob.w);
        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        for (int i = 1; i < h; i++)
        {
            const float* prev_row = bottom_top_blob.row(i - 1);
            float* this_row = bottom_top_blob.row(i);
            cumulative_sum_add(prev_row, this_row, w);
        }
        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            cumulative_sum_prefix_sum_row(bottom_top_blob.row(i), w);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 0)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int c = bottom_top_blob.c;
        const int size = w * h;

        for (int q = 1; q < c; q++)
        {
            const float* prev = bottom_top_blob.channel(q - 1);
            float* cur = bottom_top_blob.channel(q);
            cumulative_sum_add(prev, cur, size);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 1)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int c = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            Mat this_channel = bottom_top_blob.channel(q);
            for (int i = 1; i < h; i++)
            {
                const float* prev_row = this_channel.row(i - 1);
                float* this_row = this_channel.row(i);
                cumulative_sum_add(prev_row, this_row, w);
            }
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 2)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int c = bottom_top_blob.c;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int idx = 0; idx < c * h; idx++)
        {
            const int q = idx / h;
            const int i = idx - q * h;
            cumulative_sum_prefix_sum_row(bottom_top_blob.channel(q).row(i), w);
        }
        return 0;
    }

    return -100;
}

} // namespace ncnn
