// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tile.h"

namespace ncnn {

Tile::Tile()
{
    one_blob_only = true;
    support_inplace = false;
}

int Tile::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);
    tiles = pd.get(1, 1);
    repeats = pd.get(2, Mat());

    return 0;
}

int Tile::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int repeat_w = 1;
    int repeat_h = 1;
    int repeat_d = 1;
    int repeat_c = 1;

    const int repeats_num = repeats.w;

    if (repeats.empty())
    {
        if (dims == 1) // axis == 0
        {
            repeat_w = tiles;
        }
        else if (dims == 2)
        {
            if (axis == 0) repeat_h = tiles;
            if (axis == 1) repeat_w = tiles;
        }
        else if (dims == 3)
        {
            if (axis == 0) repeat_c = tiles;
            if (axis == 1) repeat_h = tiles;
            if (axis == 2) repeat_w = tiles;
        }
        else if (dims == 4)
        {
            if (axis == 0) repeat_c = tiles;
            if (axis == 1) repeat_d = tiles;
            if (axis == 2) repeat_h = tiles;
            if (axis == 3) repeat_w = tiles;
        }
    }
    else
    {
        // numpy style tile
        const int* repeats_ptr = repeats;

        if (repeats_num == 1)
        {
            repeat_w = repeats_ptr[0];
        }
        if (repeats_num == 2)
        {
            repeat_h = repeats_ptr[0];
            repeat_w = repeats_ptr[1];
        }
        if (repeats_num == 3)
        {
            if (dims == 4)
            {
                repeat_d = repeats_ptr[0];
                repeat_h = repeats_ptr[1];
                repeat_w = repeats_ptr[2];
            }
            else
            {
                repeat_c = repeats_ptr[0];
                repeat_h = repeats_ptr[1];
                repeat_w = repeats_ptr[2];
            }
        }
        if (repeats_num == 4)
        {
            repeat_c = repeats_ptr[0];
            repeat_d = repeats_ptr[1];
            repeat_h = repeats_ptr[2];
            repeat_w = repeats_ptr[3];
        }
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    const int outdims = std::max(dims, repeats_num);
    if (repeat_w != 1 && repeat_h == 1 && repeat_d == 1 && repeat_c == 1)
    {
        if (outdims == 1)
            top_blob.create(w * repeat_w, elemsize, opt.blob_allocator);
        if (outdims == 2)
            top_blob.create(w * repeat_w, h, elemsize, opt.blob_allocator);
        if (outdims == 3)
            top_blob.create(w * repeat_w, h, channels, elemsize, opt.blob_allocator);
        if (outdims == 4)
            top_blob.create(w * repeat_w, h, d, channels, elemsize, opt.blob_allocator);
    }
    else if (repeat_h != 1 && repeat_d == 1 && repeat_c == 1)
    {
        if (outdims == 2)
            top_blob.create(w * repeat_w, h * repeat_h, elemsize, opt.blob_allocator);
        if (outdims == 3)
            top_blob.create(w * repeat_w, h * repeat_h, channels, elemsize, opt.blob_allocator);
        if (outdims == 4)
            top_blob.create(w * repeat_w, h * repeat_h, d, channels, elemsize, opt.blob_allocator);
    }
    else if (repeat_d == 1 && repeat_c != 1)
    {
        if (outdims == 3)
            top_blob.create(w * repeat_w, h * repeat_h, channels * repeat_c, elemsize, opt.blob_allocator);
        if (outdims == 4)
            top_blob.create(w * repeat_w, h * repeat_h, d, channels * repeat_c, elemsize, opt.blob_allocator);
    }
    else if (repeat_d != 1 && repeat_c != 1)
    {
        if (outdims == 4)
            top_blob.create(w * repeat_w, h * repeat_h, d * repeat_d, channels * repeat_c, elemsize, opt.blob_allocator);
    }
    else // all ones
    {
        if (repeats_num == 0 || dims == repeats_num)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (outdims == 2)
            top_blob.create(w * repeat_w, h * repeat_h, elemsize, opt.blob_allocator);
        if (outdims == 3)
            top_blob.create(w * repeat_w, h * repeat_h, channels * repeat_c, elemsize, opt.blob_allocator);
        if (outdims == 4)
            top_blob.create(w * repeat_w, h * repeat_h, d * repeat_d, channels * repeat_c, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        // repeat 0-w
        for (int z = 0; z < d; z++)
        {
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.channel(q).depth(z).row(y);
                float* outptr = top_blob.channel(q).depth(z).row(y);

                for (int p = 0; p < repeat_w; p++)
                {
                    memcpy(outptr, ptr, w * sizeof(float));
                    outptr += w;
                }
            }
        }

        // repeat 1-h
        for (int z = 0; z < d; z++)
        {
            const float* ptr = top_blob.channel(q).depth(z);
            float* outptr = top_blob.channel(q).depth(z).row(h);

            const int size = w * repeat_w * h;
            for (int p = 1; p < repeat_h; p++)
            {
                memcpy(outptr, ptr, size * sizeof(float));
                outptr += size;
            }
        }

        // repeat 1-d
        {
            const float* ptr = top_blob.channel(q);
            float* outptr = top_blob.channel(q).depth(d);

            const int size = w * repeat_w * h * repeat_h * d;
            for (int p = 1; p < repeat_d; p++)
            {
                memcpy(outptr, ptr, size * sizeof(float));
                outptr += size;
            }
        }
    }

    // repeat 1-c
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 1; p < repeat_c; p++)
    {
        const float* ptr = top_blob.channel_range(0, channels);
        float* outptr = top_blob.channel_range(p * channels, channels);

        memcpy(outptr, ptr, top_blob.cstep * channels * sizeof(float));
    }

    return 0;
}

} // namespace ncnn
