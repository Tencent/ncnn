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

#include "gather.h"

namespace ncnn {
Gather::Gather()
{
    one_blob_only = false;   // 是否单一输入
    support_inplace = false; // 是否支持原地运算
}

int Gather::load_param(const ParamDict& pd)
{
    dim = pd.get(0, -1); // dim = [-dim~dim-1]
    return 0;
}

static void print_int_array(const ncnn::Mat& a)
{
    const int* pa = a;

    fprintf(stderr, "[");
    for (int i = 0; i < a.w; i++)
    {
        fprintf(stderr, " %d", pa[i]);
    }
    fprintf(stderr, " ]");
}

int Gather::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& index_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0]; // 仅1个输出
    int dims = bottom_blob.dims;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int index_w = index_blob.w; // 索引数据
    int index_h = index_blob.h;
    int index_c = index_blob.c;
    int index_d = index_blob.d;
    int index_channels = index_blob.c;

    int axis = dim < 0 ? dim + dims : dim;
    // 检查k值是否有效
    if (index_w < 1 || axis >= dims)
    {
        return -1;
    }

    // 除原始dim维度不同外，输出形状与索引形状相同
    if (dims == 1)
    {
        // 创建输出blob
        top_blob.create(index_w, elemsize, opt.blob_allocator);
        const float* ptr = bottom_blob;
        float* outptr = top_blob;
        const int* index_ptr = index_blob;
        for (int i = 0; i < index_w; i++)
        {
            outptr[i] = ptr[index_ptr[i]];
        }
    }
    else if (dims == 2)
    {
        top_blob.create(index_w, index_h, elemsize, opt.blob_allocator);
        if (axis == 0) // 行方向
        {
            for (int i = 0; i < index_h; i++)
            {
                float* outptr = top_blob.row(i);
                const int* indexptr = (const int*)index_blob.row(i);

                for (int j = 0; j < index_w; j++)
                {
                    const float* ptr = bottom_blob.row(indexptr[j]);
                    outptr[j] = ptr[j];
                }
            }
        }
        else if (axis == 1) // 列方向
        {
            for (int i = 0; i < index_h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float* outptr = top_blob.row(i);
                const int* indexptr = (const int*)index_blob.row(i);

                for (int j = 0; j < index_w; j++)
                {
                    outptr[j] = ptr[indexptr[j]];
                }
            }
        }
    }
    else if (dims == 3)
    {
        top_blob.create(index_w, index_h, index_channels, elemsize, opt.blob_allocator);
        if (axis == 0) // channels维度
        {
            for (int q = 0; q < index_channels; q++)
            {
                Mat m = top_blob.channel(q);
                const Mat index_m = index_blob.channel(q);

                for (int i = 0; i < index_h; i++)
                {
                    float* outptr = m.row(i);
                    const int* indexptr = (const int*)index_m.row(i);

                    for (int j = 0; j < index_w; j++)
                    {
                        const Mat bottom_c = bottom_blob.channel(indexptr[j]);
                        const float* ptr = bottom_c.row(i);
                        outptr[j] = ptr[j];
                    }
                }
            }
        }
        else if (axis == 1) // h维度
        {
            for (int q = 0; q < index_channels; q++)
            {
                Mat m = top_blob.channel(q);
                const Mat bottom_c = bottom_blob.channel(q);
                const Mat index_m = index_blob.channel(q);

                for (int i = 0; i < index_h; i++)
                {
                    float* outptr = m.row(i);
                    const int* indexptr = (const int*)index_m.row(i);

                    for (int j = 0; j < index_w; j++)
                    {
                        const float* ptr = bottom_c.row(indexptr[j]);
                        outptr[j] = ptr[j];
                    }
                }
            }
        }
        else if (axis == 2) // w维度
        {
            for (int q = 0; q < index_channels; q++)
            {
                Mat m = top_blob.channel(q);
                const Mat bottom_c = bottom_blob.channel(q);
                const Mat index_m = index_blob.channel(q);

                for (int i = 0; i < index_h; i++)
                {
                    float* outptr = m.row(i);
                    const float* ptr = bottom_c.row(i);
                    const int* indexptr = (const int*)index_m.row(i);

                    for (int j = 0; j < index_w; j++)
                    {
                        outptr[j] = ptr[indexptr[j]];
                    }
                }
            }
        }
    }
    else if (dims == 4)
    {
        top_blob.create(index_w, index_h, index_d, index_channels, elemsize, opt.blob_allocator);
        if (axis == 0) // channels维度
        {
            for (int q = 0; q < index_channels; q++)
            {
                const Mat index_c = index_blob.channel(q);
                Mat top_c = top_blob.channel(q);

                for (int z = 0; z < index_d; z++)
                {
                    const Mat index_z = index_c.channel(z);
                    Mat top_z = top_c.channel(z);

                    for (int i = 0; i < index_h; i++)
                    {
                        float* outptr = top_z.row(i);
                        const int* indexptr = (const int*)index_z.row(i);

                        for (int j = 0; j < index_w; j++)
                        {
                            const Mat bottom_c = bottom_blob.channel(indexptr[j]);
                            const Mat bottom_z = bottom_c.channel(z);
                            const float* ptr = bottom_z.row(i);
                            outptr[j] = ptr[j];
                        }
                    }
                }
            }
        }
        else if (axis == 1) // d维度
        {
            for (int q = 0; q < index_channels; q++)
            {
                const Mat index_c = index_blob.channel(q);
                const Mat bottom_c = bottom_blob.channel(q);
                Mat top_c = top_blob.channel(q);

                for (int z = 0; z < index_d; z++)
                {
                    const Mat index_z = index_c.channel(z);
                    Mat top_z = top_c.channel(z);

                    for (int i = 0; i < index_h; i++)
                    {
                        float* outptr = top_z.row(i);
                        const int* indexptr = (const int*)index_z.row(i);

                        for (int j = 0; j < index_w; j++)
                        {
                            const Mat bottom_z = bottom_c.channel(indexptr[j]);
                            const float* ptr = bottom_z.row(i);
                            outptr[j] = ptr[j];
                        }
                    }
                }
            }
        }
        else if (axis == 2) // h维度
        {
            for (int q = 0; q < index_channels; q++)
            {
                const Mat index_c = index_blob.channel(q);
                const Mat bottom_c = bottom_blob.channel(q);
                Mat top_c = top_blob.channel(q);

                for (int z = 0; z < index_d; z++)
                {
                    const Mat index_z = index_c.channel(z);
                    const Mat bottom_z = bottom_c.channel(z);
                    Mat top_z = top_c.channel(z);

                    for (int i = 0; i < index_h; i++)
                    {
                        float* outptr = top_z.row(i);
                        const int* indexptr = (const int*)index_z.row(i);

                        for (int j = 0; j < index_w; j++)
                        {
                            const float* ptr = bottom_z.row(indexptr[j]);
                            outptr[j] = ptr[j];
                        }
                    }
                }
            }
        }
        else if (axis == 3) // w维度
        {
            for (int q = 0; q < index_channels; q++)
            {
                const Mat index_c = index_blob.channel(q);
                const Mat bottom_c = bottom_blob.channel(q);
                Mat top_c = top_blob.channel(q);

                for (int z = 0; z < index_d; z++)
                {
                    const Mat index_z = index_c.channel(z);
                    const Mat bottom_z = bottom_c.channel(z);
                    Mat top_z = top_c.channel(z);

                    for (int i = 0; i < index_h; i++)
                    {
                        float* outptr = top_z.row(i);
                        const float* ptr = bottom_z.row(i);
                        const int* indexptr = (const int*)index_z.row(i);

                        for (int j = 0; j < index_w; j++)
                        {
                            outptr[j] = ptr[indexptr[j]];
                        }
                    }
                }
            }
        }
    }
    else
    {
        return -1;
    }

    return 0;
}

} // namespace ncnn