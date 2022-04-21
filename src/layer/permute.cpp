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

#include "permute.h"

namespace ncnn {

Permute::Permute()
{
    one_blob_only = true;
    support_inplace = false;
}

int Permute::load_param(const ParamDict& pd)
{
    order_type = pd.get(0, 0);

    return 0;
}

int Permute::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (dims == 2)
    {
        // order_type
        // 0 = w h
        // 1 = h w

        if (order_type == 0)
        {
            top_blob = bottom_blob;
        }
        if (order_type == 1)
        {
            top_blob.create(h, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            float* outptr = top_blob;

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    *outptr++ = bottom_blob.row(j)[i];
                }
            }
        }
    }

    if (dims == 3)
    {
        // order_type
        // 0 = w h c
        // 1 = h w c
        // 2 = w c h
        // 3 = c w h
        // 4 = h c w
        // 5 = c h w

        if (order_type == 0)
        {
            top_blob = bottom_blob;
        }
        if (order_type == 1)
        {
            top_blob.create(h, w, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        *outptr++ = m.row(j)[i];
                    }
                }
            }
        }
        if (order_type == 2)
        {
            top_blob.create(w, channels, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < channels; i++)
                {
                    const float* ptr = bottom_blob.channel(i).row(q);

                    for (int j = 0; j < w; j++)
                    {
                        *outptr++ = ptr[j];
                    }
                }
            }
        }
        if (order_type == 3)
        {
            top_blob.create(channels, w, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < channels; j++)
                    {
                        *outptr++ = bottom_blob.channel(j).row(q)[i];
                    }
                }
            }
        }
        if (order_type == 4)
        {
            top_blob.create(h, channels, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < channels; i++)
                {
                    const Mat m = bottom_blob.channel(i);

                    for (int j = 0; j < h; j++)
                    {
                        *outptr++ = m.row(j)[q];
                    }
                }
            }
        }
        if (order_type == 5)
        {
            top_blob.create(channels, h, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < channels; j++)
                    {
                        *outptr++ = bottom_blob.channel(j).row(i)[q];
                    }
                }
            }
        }
    }

    if (dims == 4)
    {
        // order_type
        // 0 = w h d c
        // 1 = h w d c
        // 2 = w d h c
        // 3 = d w h c
        // 4 = h d w c
        // 5 = d h w c
        // 6 = w h c d
        // 7 = h w c d
        // 8 = w c h d
        // 9 = c w h d
        //10 = h c w d
        //11 = c h w d
        //12 = w d c h
        //13 = d w c h
        //14 = w c d h
        //15 = c w d h
        //16 = d c w h
        //17 = c d w h
        //18 = h d c w
        //19 = d h c w
        //20 = h c d w
        //21 = c h d w
        //22 = d c h w
        //23 = c d h w

        if (order_type == 0)
        {
            top_blob = bottom_blob;
        }
        if (order_type == 1)
        {
            top_blob.create(h, w, d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    const Mat m = bottom_blob.channel(q).depth(z);

                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < h; j++)
                        {
                            *outptr++ = m.row(j)[i];
                        }
                    }
                }
            }
        }
        if (order_type == 2)
        {
            top_blob.create(w, d, h, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        const float* ptr = bottom_blob.channel(q).depth(i).row(z);

                        for (int j = 0; j < w; j++)
                        {
                            *outptr++ = ptr[j];
                        }
                    }
                }
            }
        }
        if (order_type == 3)
        {
            top_blob.create(d, w, h, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            *outptr++ = m.depth(j).row(z)[i];
                        }
                    }
                }
            }
        }
        if (order_type == 4)
        {
            top_blob.create(h, d, w, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        const Mat m = bottom_blob.channel(q).depth(i);

                        for (int j = 0; j < h; j++)
                        {
                            *outptr++ = m.row(j)[z];
                        }
                    }
                }
            }
        }
        if (order_type == 5)
        {
            top_blob.create(d, h, w, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            *outptr++ = m.depth(j).row(i)[z];
                        }
                    }
                }
            }
        }
        if (order_type == 6)
        {
            top_blob.create(w, h, channels, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr = bottom_blob.channel(z).depth(q).row(i);

                        for (int j = 0; j < w; j++)
                        {
                            *outptr++ = ptr[j];
                        }
                    }
                }
            }
        }
        if (order_type == 7)
        {
            top_blob.create(h, w, channels, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    const Mat m = bottom_blob.channel(z).depth(q);

                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < h; j++)
                        {
                            *outptr++ = m.row(j)[i];
                        }
                    }
                }
            }
        }
        if (order_type == 8)
        {
            top_blob.create(w, channels, h, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const float* ptr = bottom_blob.channel(i).depth(q).row(z);

                        for (int j = 0; j < w; j++)
                        {
                            *outptr++ = ptr[j];
                        }
                    }
                }
            }
        }
        if (order_type == 9)
        {
            top_blob.create(channels, w, h, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            *outptr++ = bottom_blob.channel(j).depth(q).row(z)[i];
                        }
                    }
                }
            }
        }
        if (order_type == 10)
        {
            top_blob.create(h, channels, w, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const Mat m = bottom_blob.channel(i).depth(q);

                        for (int j = 0; j < h; j++)
                        {
                            *outptr++ = m.row(j)[z];
                        }
                    }
                }
            }
        }
        if (order_type == 11)
        {
            top_blob.create(channels, h, w, d, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < d; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            *outptr++ = bottom_blob.channel(j).depth(q).row(i)[z];
                        }
                    }
                }
            }
        }
        if (order_type == 12)
        {
            top_blob.create(w, d, channels, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        const float* ptr = bottom_blob.channel(z).depth(i).row(q);

                        for (int j = 0; j < w; j++)
                        {
                            *outptr++ = ptr[j];
                        }
                    }
                }
            }
        }
        if (order_type == 13)
        {
            top_blob.create(d, w, channels, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    const Mat m = bottom_blob.channel(z);

                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            *outptr++ = m.depth(j).row(q)[i];
                        }
                    }
                }
            }
        }
        if (order_type == 14)
        {
            top_blob.create(w, channels, d, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const float* ptr = bottom_blob.channel(i).depth(z).row(q);

                        for (int j = 0; j < w; j++)
                        {
                            *outptr++ = ptr[j];
                        }
                    }
                }
            }
        }
        if (order_type == 15)
        {
            top_blob.create(channels, w, d, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < w; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            *outptr++ = bottom_blob.channel(j).depth(z).row(q)[i];
                        }
                    }
                }
            }
        }
        if (order_type == 16)
        {
            top_blob.create(d, channels, w, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const Mat m = bottom_blob.channel(i);

                        for (int j = 0; j < d; j++)
                        {
                            *outptr++ = m.depth(j).row(q)[z];
                        }
                    }
                }
            }
        }
        if (order_type == 17)
        {
            top_blob.create(channels, d, w, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < w; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            *outptr++ = bottom_blob.channel(j).depth(i).row(q)[z];
                        }
                    }
                }
            }
        }
        if (order_type == 18)
        {
            top_blob.create(h, d, channels, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        const Mat m = bottom_blob.channel(z).depth(i);

                        for (int j = 0; j < h; j++)
                        {
                            *outptr++ = m.row(j)[q];
                        }
                    }
                }
            }
        }
        if (order_type == 19)
        {
            top_blob.create(d, h, channels, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < channels; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        const Mat m = bottom_blob.channel(z);

                        for (int j = 0; j < d; j++)
                        {
                            *outptr++ = m.depth(j).row(i)[q];
                        }
                    }
                }
            }
        }
        if (order_type == 20)
        {
            top_blob.create(h, channels, d, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const Mat m = bottom_blob.channel(i).depth(z);

                        for (int j = 0; j < h; j++)
                        {
                            *outptr++ = m.row(j)[q];
                        }
                    }
                }
            }
        }
        if (order_type == 21)
        {
            top_blob.create(channels, h, d, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            *outptr++ = bottom_blob.channel(j).depth(z).row(i)[q];
                        }
                    }
                }
            }
        }
        if (order_type == 22)
        {
            top_blob.create(d, channels, h, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        const Mat m = bottom_blob.channel(i);

                        for (int j = 0; j < d; j++)
                        {
                            *outptr++ = m.depth(j).row(z)[q];
                        }
                    }
                }
            }
        }
        if (order_type == 23)
        {
            top_blob.create(channels, d, h, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < h; z++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        for (int j = 0; j < channels; j++)
                        {
                            *outptr++ = bottom_blob.channel(j).depth(i).row(z)[q];
                        }
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
