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

DEFINE_LAYER_CREATOR(Tile)

Tile::Tile()
{
    one_blob_only = true;
    support_inplace = false;
}

int Tile::load_param(const ParamDict& pd)
{
    dim = pd.get(0, 0);
    tiles = pd.get(1, 1);

    return 0;
}

int Tile::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    if (dim == 0)
    {
        top_blob.create(w, h, channels * tiles, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        int size = bottom_blob.cstep * channels;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<tiles; p++)
        {
            float* outptr = top_blob.channel(p * channels);

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i];
            }
        }
    }
    else if (dim == 1)
    {
        top_blob.create(w, h * tiles, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int p=0; p<tiles; p++)
            {
                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i];
                }

                outptr += size;
            }
        }
    }
    else if (dim == 2)
    {
        top_blob.create(w * tiles, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (int p=0; p<tiles; p++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        outptr[j] = ptr[j];
                    }

                    outptr += w;
                }

                ptr += w;
            }
        }
    }

    return 0;
}

} // namespace ncnn
