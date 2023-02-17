// Copyright (c) 2023 Xiaomi Corp.        (author: Fangjun Kuang)
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

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

    return -100;
}

} // namespace ncnn
