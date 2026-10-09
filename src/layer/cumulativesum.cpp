// Copyright 2023 Xiaomi Corp.   (author: Fangjun Kuang)
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum.h"

namespace ncnn {

CumulativeSum::CumulativeSum()
{
    one_blob_only = true;
    support_inplace = true;
}

int CumulativeSum::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

int CumulativeSum::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1)
    {   // ignore axis
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        for (int i = 1; i < w; ++i)
        {
            ptr[i] = ptr[i] + ptr[i - 1];
        }

        return 0;
    } // if (dims == 1)

    if (dims == 2 && positive_axis == 0)
    {
        // sum over rows
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        for (int i = 1; i < h; ++i)
        {
            const float* prev_row = bottom_top_blob.row(i - 1);
            float* this_row = bottom_top_blob.row(i);

            for (int k = 0; k < w; ++k)
            {
                this_row[k] = this_row[k] + prev_row[k];
            }
        }

        return 0;
    } // if (dims == 2 && positive_axis == 0)

    if (dims == 2 && positive_axis == 1)
    {
        // sum over columns
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; ++i)
        {
            float* ptr = bottom_top_blob.row(i);

            for (int k = 1; k < w; ++k)
            {
                ptr[k] = ptr[k] + ptr[k - 1];
            }
        }

        return 0;
    } // if (dims == 2 && positive_axis == 1)

    if (dims == 3 && positive_axis == 0)
    {
        // sum over channels
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int c = bottom_top_blob.c;

        int size = w * h;

        for (int i = 1; i < c; ++i)
        {
            const float* prev = bottom_top_blob.channel(i - 1);
            float* cur = bottom_top_blob.channel(i);

            for (int k = 0; k < size; ++k)
            {
                cur[k] = cur[k] + prev[k];
            }
        }

        return 0;
    } // if (dims == 3 && positive_axis == 0)

    if (dims == 3 && positive_axis == 1)
    {
        // sum over rows within each channel

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int c = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat this_channel = bottom_top_blob.channel(q);

            for (int i = 1; i < h; ++i)
            {
                const float* prev_row = this_channel.row(i - 1);
                float* this_row = this_channel.row(i);

                for (int k = 0; k < w; ++k)
                {
                    this_row[k] = this_row[k] + prev_row[k];
                }
            }
        }

        return 0;
    } // if (dims == 3 && positive_axis == 1)

    if (dims == 3 && positive_axis == 2)
    {
        // sum over columns within each channel

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int c = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat this_channel = bottom_top_blob.channel(q);

            for (int i = 0; i < h; ++i)
            {
                float* ptr = this_channel.row(i);
                for (int k = 1; k < w; ++k)
                {
                    ptr[k] = ptr[k] + ptr[k - 1];
                }
            }
        }

        return 0;
    } // if (dims == 3 && positive_axis == 2)

    if (dims == 4 && positive_axis == 0)
    {
        // sum over channels
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;

        int size = w * h * d;

        for (int i = 1; i < c; ++i)
        {
            const float* prev = bottom_top_blob.channel(i - 1);
            float* cur = bottom_top_blob.channel(i);

            for (int k = 0; k < size; ++k)
            {
                cur[k] = cur[k] + prev[k];
            }
        }

        return 0;
    } // if (dims == 4 && positive_axis == 0)

    if (dims == 4 && positive_axis == 1)
    {
        // sum over depth within each channel

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;

        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat this_channel = bottom_top_blob.channel(q);

            for (int z = 1; z < d; ++z)
            {
                const float* prev = this_channel.depth(z - 1);
                float* cur = this_channel.depth(z);

                for (int k = 0; k < size; ++k)
                {
                    cur[k] = cur[k] + prev[k];
                }
            }
        }

        return 0;
    } // if (dims == 4 && positive_axis == 1)

    if (dims == 4 && positive_axis == 2)
    {
        // sum over rows within each depth slice

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat this_channel = bottom_top_blob.channel(q);

            for (int z = 0; z < d; ++z)
            {
                Mat this_depth = this_channel.depth(z);

                for (int i = 1; i < h; ++i)
                {
                    const float* prev_row = this_depth.row(i - 1);
                    float* this_row = this_depth.row(i);

                    for (int k = 0; k < w; ++k)
                    {
                        this_row[k] = this_row[k] + prev_row[k];
                    }
                }
            }
        }

        return 0;
    } // if (dims == 4 && positive_axis == 2)

    if (dims == 4 && positive_axis == 3)
    {
        // sum over columns within each depth slice

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat this_channel = bottom_top_blob.channel(q);

            for (int z = 0; z < d; ++z)
            {
                Mat this_depth = this_channel.depth(z);

                for (int i = 0; i < h; ++i)
                {
                    float* ptr = this_depth.row(i);
                    for (int k = 1; k < w; ++k)
                    {
                        ptr[k] = ptr[k] + ptr[k - 1];
                    }
                }
            }
        }

        return 0;
    } // if (dims == 4 && positive_axis == 3)

    return -100;
}

} // namespace ncnn
