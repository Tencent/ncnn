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

#include "indexselect.h"

namespace ncnn {
IndexSelect::IndexSelect()
{
    one_blob_only = false;   // 是否单一输入
    support_inplace = false; // 是否支持原地运算
}

int IndexSelect::load_param(const ParamDict& pd)
{
    dim = pd.get(0, -1); // dim = [-dim~dim-1]
    return 0;
}

int IndexSelect::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

    int index_len = index_blob.w; // 索引数据

    int axis = dim < 0 ? dim + dims : dim;
    // 检查k值是否有效
    if (index_len < 1 || axis >= dims)
    {
        return -1;
    }

    if (dims == 1)
    {
        // 创建输出blob
        top_blob.create(index_len, elemsize, opt.blob_allocator);
        const float* ptr = bottom_blob;
        float* outptr = top_blob;
        const int* index_ptr = index_blob;
        for (int i = 0; i < index_len; i++)
        {
            outptr[i] = ptr[index_ptr[i]];
        }
    }
    else if (dims == 2)
    {
        if (axis == 0)
        {
            top_blob.create(w, index_len, elemsize, opt.blob_allocator);
            const int* index_ptr = index_blob;

            for (int i = 0; i < index_len; i++)
            {
                int index = index_ptr[i];
                const float* ptr_row = bottom_blob.row(index);
                float* outptr_row = top_blob.row(i);
                memcpy(outptr_row, ptr_row, w * sizeof(float));
            }
        }
        else if (axis == 1)
        {
            top_blob.create(index_len, h, elemsize, opt.blob_allocator);
            const int* index_ptr = index_blob;
            for (int i = 0; i < h; i++)
            {
                const float* ptr_row = bottom_blob.row(i);
                float* outptr_row = top_blob.row(i);

                // 对每一行,根据索引选择对应列
                for (int j = 0; j < index_len; j++)
                {
                    int index = index_ptr[j];
                    outptr_row[j] = ptr_row[index];
                }
            }
        }
    }
    else if (dims == 3)
    {
        if (axis == 0) // channels维度
        {
            top_blob.create(w, h, index_len, elemsize, opt.blob_allocator);
            const int* index_ptr = index_blob;

            for (int q = 0; q < index_len; q++)
            {
                int index = index_ptr[q];
                const Mat bottom_channel = bottom_blob.channel(index);
                Mat top_channel = top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    const float* ptr = bottom_channel.row(i);
                    float* outptr = top_channel.row(i);
                    memcpy(outptr, ptr, w * sizeof(float));
                }
            }
        }
        else if (axis == 1) // h维度
        {
            top_blob.create(w, index_len, channels, elemsize, opt.blob_allocator);
            const int* index_ptr = index_blob;

            for (int q = 0; q < channels; q++)
            {
                const Mat bottom_channel = bottom_blob.channel(q);
                Mat top_channel = top_blob.channel(q);

                for (int i = 0; i < index_len; i++)
                {
                    int index = index_ptr[i];
                    const float* ptr = bottom_channel.row(index);
                    float* outptr = top_channel.row(i);
                    memcpy(outptr, ptr, w * sizeof(float));
                }
            }
        }
        else if (axis == 2) // w维度
        {
            top_blob.create(index_len, h, channels, elemsize, opt.blob_allocator);
            const int* index_ptr = index_blob;

            for (int q = 0; q < channels; q++)
            {
                const Mat bottom_channel = bottom_blob.channel(q);
                Mat top_channel = top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    const float* ptr = bottom_channel.row(i);
                    float* outptr = top_channel.row(i);
                    for (int j = 0; j < index_len; j++)
                    {
                        int index = index_ptr[j];
                        outptr[j] = ptr[index];
                    }
                }
            }
        }
    }

    else if (dims == 4)
    {
        if (axis == 0) // channels维度
        {
            top_blob.create(w, h, d, index_len, elemsize, opt.blob_allocator);
            const int* index_ptr = index_blob;

            for (int q = 0; q < index_len; q++)
            {
                int index = index_ptr[q];
                const Mat bottom_c = bottom_blob.channel(index);
                Mat top_c = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    const Mat bottom_d = bottom_c.channel(z);
                    Mat top_d = top_c.channel(z);
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr = bottom_d.row(i);
                        float* outptr = top_d.row(i);
                        memcpy(outptr, ptr, w * sizeof(float));
                    }
                }
            }
        }
        else if (axis == 1) // d维度
        {
            top_blob.create(w, h, index_len, channels, elemsize, opt.blob_allocator);
            const int* index_ptr = index_blob;

            for (int q = 0; q < channels; q++)
            {
                const Mat bottom_c = bottom_blob.channel(q);
                Mat top_c = top_blob.channel(q);

                for (int z = 0; z < index_len; z++)
                {
                    int index = index_ptr[z];
                    const Mat bottom_d = bottom_c.channel(index);
                    Mat top_d = top_c.channel(z);
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr = bottom_d.row(i);
                        float* outptr = top_d.row(i);
                        memcpy(outptr, ptr, w * sizeof(float));
                    }
                }
            }
        }
        else if (axis == 2) // h维度
        {
            top_blob.create(w, index_len, d, channels, elemsize, opt.blob_allocator);
            const int* index_ptr = index_blob;

            for (int q = 0; q < channels; q++)
            {
                const Mat bottom_c = bottom_blob.channel(q);
                Mat top_c = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    const Mat bottom_d = bottom_c.channel(z);
                    Mat top_d = top_c.channel(z);
                    for (int i = 0; i < index_len; i++)
                    {
                        int index = index_ptr[i];
                        const float* ptr = bottom_d.row(index);
                        float* outptr = top_d.row(i);
                        memcpy(outptr, ptr, w * sizeof(float));
                    }
                }
            }
        }
        else if (axis == 3) // w维度
        {
            top_blob.create(index_len, h, d, channels, elemsize, opt.blob_allocator);
            const int* index_ptr = index_blob;

            for (int q = 0; q < channels; q++)
            {
                const Mat bottom_c = bottom_blob.channel(q);
                Mat top_c = top_blob.channel(q);

                for (int z = 0; z < d; z++)
                {
                    const Mat bottom_d = bottom_c.channel(z);
                    Mat top_d = top_c.channel(z);
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr = bottom_d.row(i);
                        float* outptr = top_d.row(i);
                        for (int j = 0; j < index_len; j++)
                        {
                            int index = index_ptr[j];
                            outptr[j] = ptr[index];
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