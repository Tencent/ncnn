// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "argmax.h"

namespace ncnn {

ArgMax::ArgMax()
{
    one_blob_only = true;
}

int ArgMax::load_param(const ParamDict& pd)
{
    dim = pd.get(0, 0);     // [-dims~dims-1]
    keepdim = pd.get(1, 0); // default False
    return 0;
}

int ArgMax::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // 已知参数
    int dims = bottom_blob.dims;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    // 校准输入参数
    int axis = dim < 0 ? dim + dims : dim;
    if (axis < 0 || axis >= dims)
    {
        return -1;
    }

    if (dims == 1)
    {
        // 1D 只有一种情况
        top_blob.create(1, elemsize, opt.blob_allocator);
        const float* ptr = bottom_blob;
        int* outptr = top_blob;
        int max_index = 0;
        float max_value = ptr[0];
        for (int i = 1; i < w; i++)
        {
            if (ptr[i] > max_value)
            {
                max_value = ptr[i];
                max_index = i;
            }
        }
        outptr[0] = max_index;
        top_blob = top_blob.reshape(1);
    }
    else if (dims == 2)
    {
        if (axis == 0) // h维度
        {
            top_blob.create(w, elemsize, opt.blob_allocator);
            int* outptr = top_blob;
            std::vector<float> max_values(w);
            for (int j = 0; j < h; j++) // 外循环遍历列
            {
                const float* ptr = bottom_blob.row(j);
                for (int i = 0; i < w; i++) // 内循环遍历行
                {
                    if (j == 0)
                    {
                        outptr[i] = 0;
                        max_values[i] = ptr[i];
                    }
                    else if (ptr[i] > max_values[i])
                    {
                        max_values[i] = ptr[i];
                        outptr[i] = j;
                    }
                }
            }
            if (keepdim)
            {
                top_blob = top_blob.reshape(w, 1);
            }
        }
        else if (axis == 1) // w维度
        {
            top_blob.create(h, elemsize, opt.blob_allocator);
            int* outptr = top_blob;
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                int max_index = 0;
                float max_value = ptr[0];
                for (int j = 1; j < w; j++)
                {
                    if (ptr[j] > max_value)
                    {
                        max_value = ptr[j];
                        max_index = j;
                    }
                }
                outptr[i] = max_index;
            }
            if (keepdim)
            {
                top_blob = top_blob.reshape(1, h);
            }
        }
    }
    else if (dims == 3)
    {
        if (axis == 0) // channels维度
        {
            top_blob.create(w, h, elemsize, opt.blob_allocator);
            int* outptr = top_blob;
            std::vector<float> max_values(w * h);

            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.channel(0).row(i);
                float* max_ptr = &max_values[i * w];
                int* out_ptr = outptr + i * w;

                for (int j = 0; j < w; j++)
                {
                    max_ptr[j] = ptr[j];
                    out_ptr[j] = 0;
                }
            }

            for (int q = 1; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    const float* ptr = bottom_blob.channel(q).row(i);
                    float* max_ptr = &max_values[i * w];
                    int* out_ptr = outptr + i * w;

                    for (int j = 0; j < w; j++)
                    {
                        if (ptr[j] > max_ptr[j])
                        {
                            max_ptr[j] = ptr[j];
                            out_ptr[j] = q;
                        }
                    }
                }
            }
            if (keepdim)
            {
                top_blob = top_blob.reshape(w, h, 1);
            }
        }
        else if (axis == 1) // h维度
        {
            top_blob.create(w, channels, elemsize, opt.blob_allocator);
            int* outptr = top_blob;
            std::vector<float> max_values(w * channels);

            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                const float* ptr = m.row(0);
                float* max_ptr = &max_values[q * w];
                int* out_ptr = outptr + q * w;

                for (int i = 0; i < w; i++)
                {
                    max_ptr[i] = ptr[i];
                    out_ptr[i] = 0;
                }

                for (int i = 1; i < h; i++)
                {
                    const float* ptr = m.row(i);
                    for (int j = 0; j < w; j++)
                    {
                        if (ptr[j] > max_ptr[j])
                        {
                            max_ptr[j] = ptr[j];
                            out_ptr[j] = i;
                        }
                    }
                }
            }
            if (keepdim)
            {
                top_blob = top_blob.reshape(w, 1, channels);
            }
        }
        else if (axis == 2) // w维度
        {
            top_blob.create(h, channels, elemsize, opt.blob_allocator);
            int* outptr = top_blob;

            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                int* out_ptr = outptr + q * h;

                for (int i = 0; i < h; i++)
                {
                    const float* ptr = m.row(i);
                    float max_value = ptr[0];
                    int max_index = 0;

                    for (int j = 1; j < w; j++)
                    {
                        if (ptr[j] > max_value)
                        {
                            max_value = ptr[j];
                            max_index = j;
                        }
                    }
                    out_ptr[i] = max_index;
                }
            }
            if (keepdim)
            {
                top_blob = top_blob.reshape(1, h, channels);
            }
        }
    }
    else if (dims == 4)
    {
        if (axis == 0) // channels维度
        {
            top_blob.create(w, h, d, elemsize, opt.blob_allocator);

            for (int zi = 0; zi < d; zi++)
            {
                for (int yi = 0; yi < h; yi++)
                {
                    int* outptr = (int*)top_blob.channel(zi).row(yi);

                    // 遍历每个空间位置
                    for (int xi = 0; xi < w; xi++)
                    {
                        float maxval = bottom_blob.channel(0).depth(zi).row(yi)[xi];
                        int maxindex = 0;

                        // 在channel维度上寻找最大值
                        for (int q = 1; q < channels; q++)
                        {
                            float val = bottom_blob.channel(q).depth(zi).row(yi)[xi];
                            if (val > maxval)
                            {
                                maxval = val;
                                maxindex = q;
                            }
                        }
                        outptr[xi] = maxindex;
                    }
                }
            }
            if (keepdim)
            {
                top_blob = top_blob.reshape(w, h, d, 1);
            }
        }
        else if (axis == 1) // d维度
        {
            top_blob.create(w, h, channels, elemsize, opt.blob_allocator);

            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                Mat out_c = top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    int* out_ptr = out_c.row<int>(i);
                    const float* in_ptr = m.depth(0).row(i);

                    // 初始化每行的最大值和索引
                    std::vector<float> max_vals(w);
                    memcpy(max_vals.data(), in_ptr, w * sizeof(float));
                    memset(out_ptr, 0, w * sizeof(int));

                    // 遍历depth维度比较更新
                    for (int z = 1; z < d; z++)
                    {
                        const float* ptr = m.depth(z).row(i);
                        for (int j = 0; j < w; j++)
                        {
                            if (ptr[j] > max_vals[j])
                            {
                                max_vals[j] = ptr[j];
                                out_ptr[j] = z;
                            }
                        }
                    }
                }
            }
            if (keepdim)
            {
                top_blob = top_blob.reshape(w, h, 1, channels);
            }
        }
        else if (axis == 2) // h维度
        {
            top_blob.create(w, d, channels, elemsize, opt.blob_allocator);
            std::vector<float> max_values(w * d * channels);

            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                for (int z = 0; z < d; z++)
                {
                    const Mat n = m.channel(z);
                    float* max_ptr = &max_values[(q * d + z) * w];
                    int* out_ptr = (int*)top_blob.channel(q).row(z);

                    // 初始化使用完整行
                    const float* ptr0 = n.row(0);
                    for (int j = 0; j < w; j++)
                    {
                        max_ptr[j] = ptr0[j];
                        out_ptr[j] = 0;
                    }

                    // 逐行比较更新
                    for (int i = 1; i < h; i++)
                    {
                        const float* ptr = n.row(i);
                        for (int j = 0; j < w; j++)
                        {
                            if (ptr[j] > max_ptr[j])
                            {
                                max_ptr[j] = ptr[j];
                                out_ptr[j] = i;
                            }
                        }
                    }
                }
            }
            if (keepdim)
            {
                top_blob = top_blob.reshape(w, 1, d, channels);
            }
        }
        else if (axis == 3) // w维度
        {
            top_blob.create(h, d, channels, elemsize, opt.blob_allocator);

            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                for (int z = 0; z < d; z++)
                {
                    const Mat n = m.channel(z);
                    int* out_ptr = (int*)top_blob.channel(q).row(z); // 获取深度切片

                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr = n.row(i);
                        float max_value = ptr[0];
                        int max_index = 0;

                        for (int j = 1; j < w; j++)
                        {
                            if (ptr[j] > max_value)
                            {
                                max_value = ptr[j];
                                max_index = j;
                            }
                        }
                        out_ptr[i] = max_index;
                    }
                }
            }
            if (keepdim)
            {
                top_blob = top_blob.reshape(1, h, d, channels);
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
