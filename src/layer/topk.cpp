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

#include "topk.h"
#include <functional>

namespace ncnn {

TopK::TopK()
{
    // one_blob_only = true; // 仅有1个输入和1个输出
    // support_inplace = true; // 是否支持原地运算，即输入和输出共享一个blob
    one_blob_only = false;   // 只需要一个输入 blob
    support_inplace = false; // 是否支持原地运算
}

int TopK::load_param(const ParamDict& pd)
{
    k = pd.get(0, 1); // [获取参数，默认值1]
    axis = pd.get(1, 0);
    largest = pd.get(2, 1);
    sorted = pd.get(3, 1);
    // largest = 0;
    sorted = 0;
    // k = 2;
    printf("参数加载k=%d, axis=%d, largest=%d, sorted=%d\n", k, axis, largest, sorted);
    return 0;
}

int TopK::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    printf("dims=%d, w=%d, h=%d, d=%d, channels=%d, elemsize=%zu\n", dims, w, h, d, channels, elemsize);
    // 确保top_blobs大小正确
    top_blobs.resize(2);
    // 检查k值是否有效
    if (k <= 0 || k > w * h * channels)
    {
        return -1;
    }

    // 创建输出Mat
    Mat& top_blob_values = top_blobs[0];  // values
    Mat& top_blob_indices = top_blobs[1]; // indices

    // 根据largest参数定义比较函数
    auto comp = [this](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return this->largest ? (a.first > b.first) : (a.first < b.first);
    };
    // 根据dims创建不同维度的输出
    if (dims == 1)
    {
        // 创建输出blob
        top_blob_values.create(k, elemsize, opt.blob_allocator);
        top_blob_indices.create(k, elemsize, opt.blob_allocator);

        const float* ptr = bottom_blob;
        float* outptr = top_blob_values;
        float* indices = top_blob_indices;
        // 创建pair数组用于排序
        std::vector<std::pair<float, int> > vec(w);
        for (int i = 0; i < w; i++)
        {
            vec[i] = std::make_pair(ptr[i], i);
        }

        // 根据sorted参数选择排序方式
        if (sorted)
        {
            std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), comp);
        }
        else
        {
            std::nth_element(vec.begin(), vec.begin() + k - 1, vec.end(), comp);
            // 对前k个元素进行排序以保持一致性
            std::sort(vec.begin(), vec.begin() + k, comp);
        }

        // 保存结果
        for (int i = 0; i < k; i++)
        {
            outptr[i] = vec[i].first;
            indices[i] = vec[i].second;
        }
    }
    else if (dims == 2)
    {
        // 在每一行上进行TopK
        if (axis == 0)
        {
            top_blob_values.create(w, k, elemsize, opt.blob_allocator);
            top_blob_indices.create(w, k, sizeof(int), opt.blob_allocator);

            // #pragma omp parallel for
            for (int j = 0; j < w; j++) // 对每列进行处理
            {
                std::vector<std::pair<float, int> > vec(h);
                // 收集当前列的所有元素
                for (int i = 0; i < h; i++)
                {
                    vec[i] = std::make_pair(bottom_blob.row(i)[j], i);
                }

                if (sorted)
                {
                    std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), comp);
                }
                else
                {
                    std::nth_element(vec.begin(), vec.begin() + k - 1, vec.end(), comp);
                    std::sort(vec.begin(), vec.begin() + k, comp);
                }

                // 保存结果到对应列
                for (int i = 0; i < k; i++)
                {
                    top_blob_values.row(i)[j] = vec[i].first;
                    top_blob_indices.row(i)[j] = static_cast<float>(vec[i].second);
                }
            }
        }
        // 在每一列上进行TopK ，axis=-1等价于axis=1
        else
        {
            top_blob_values.create(h, k, elemsize, opt.blob_allocator);
            top_blob_indices.create(h, k, sizeof(int), opt.blob_allocator);

            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float* outptr = top_blob_values.row(i);
                float* indices = top_blob_indices.row<float>(i);

                std::vector<std::pair<float, int> > vec(w);
                for (int j = 0; j < w; j++)
                {
                    vec[j] = std::make_pair(ptr[j], j);
                }

                if (sorted)
                {
                    std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), comp);
                }
                else
                {
                    std::nth_element(vec.begin(), vec.begin() + k - 1, vec.end(), comp);
                    std::sort(vec.begin(), vec.begin() + k, comp);
                }

                for (int j = 0; j < k; j++)
                {
                    outptr[j] = vec[j].first;
                    indices[j] = vec[j].second;
                }
            }
        }
    }
    else if (dims == 3)
    {
        if (axis == 0)
        {
            // 深度方向上;w不变，高度h变为k
            top_blob_values.create(w, h, k, elemsize, opt.blob_allocator);
            top_blob_indices.create(w, h, k, elemsize, opt.blob_allocator);
            // #pragma omp parallel for collapse(2)
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    // 收集该位置所有channel的值
                    std::vector<std::pair<float, int> > channel_values(channels);
                    for (int c = 0; c < channels; c++)
                    {
                        const float* ptr = bottom_blob.channel(c);
                        channel_values[c] = std::make_pair(ptr[i * w + j], c);
                    }

                    // 排序
                    if (sorted)
                    {
                        std::sort(channel_values.begin(), channel_values.end(), comp);
                    }
                    else
                    {
                        std::nth_element(channel_values.begin(), channel_values.begin() + k - 1, channel_values.end(), comp);
                        std::sort(channel_values.begin(), channel_values.begin() + k, comp);
                    }

                    // 写回结果
                    for (int c = 0; c < channels; c++)
                    {
                        float* outptr = top_blob_values.channel(c);
                        float* indices = top_blob_indices.channel(c);
                        outptr[i * w + j] = channel_values[c].first;
                        indices[i * w + j] = channel_values[c].second;
                    }
                }
            }
        }
        else if (axis == 1)
        {
            // 子元素内部进行TopK;w不变，高度变为k
            top_blob_values.create(w, k, channels, elemsize, opt.blob_allocator);
            top_blob_indices.create(w, k, channels, elemsize, opt.blob_allocator);
            for (int q = 0; q < channels; q++)
            {
                // 获取每个channel的行
                std::vector<std::pair<float, int> > row_scores(h);
                for (int j = 0; j < w; j++)
                {
                    // 每列单独处理
                    for (int i = 0; i < h; i++)
                    {
                        row_scores[i] = std::make_pair(bottom_blob.channel(q).row(i)[j], i);
                    }

                    // 找到最大行的索引
                    if (sorted)
                    {
                        std::partial_sort(row_scores.begin(), row_scores.begin() + k, row_scores.end(), comp);
                    }
                    else
                    {
                        std::nth_element(row_scores.begin(), row_scores.begin() + k - 1, row_scores.end(), comp);
                    }

                    // 保存该列的结果
                    for (int i = 0; i < k; i++)
                    {
                        float* outptr = top_blob_values.channel(q).row(i);
                        float* indices = top_blob_indices.channel(q).row(i);
                        outptr[j] = row_scores[i].first;
                        indices[j] = row_scores[i].second;
                    }
                }
            }
        }
        else if (axis == 2 || axis == -1)
        {
            // 输出为k长度的向量，高度不变
            top_blob_values.create(k, h, channels, elemsize, opt.blob_allocator);
            top_blob_indices.create(k, h, channels, elemsize, opt.blob_allocator);
            for (int q = 0; q < channels; q++)
            {
                for (int j = 0; j < h; j++)
                {
                    const float* ptr = bottom_blob.channel(q).row(j);
                    float* outptr = top_blob_values.channel(q).row(j);
                    float* indices = top_blob_indices.channel(q).row<float>(j);

                    std::vector<std::pair<float, int> > vec(w);
                    for (int i = 0; i < w; i++)
                    {
                        vec[i] = std::make_pair(ptr[i], i);
                    }

                    if (sorted)
                    {
                        std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), comp);
                    }
                    else
                    {
                        std::nth_element(vec.begin(), vec.begin() + k - 1, vec.end(), comp);
                        std::sort(vec.begin(), vec.begin() + k, comp);
                    }

                    for (int i = 0; i < k; i++)
                    {
                        outptr[i] = vec[i].first;
                        indices[i] = vec[i].second;
                    }
                }
            }
        }
    }
    else if (dims == 4)
    {
        // 4D数据处理
        if (axis == 0)
        {
            // 在d维度上进行TopK
            top_blob_values.create(w, h, k, channels, elemsize, opt.blob_allocator);
            top_blob_indices.create(w, h, k, channels, elemsize, opt.blob_allocator);

            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        std::vector<std::pair<float, int> > depth_values(d);
                        // 收集depth维度数据
                        for (int z = 0; z < d; z++)
                        {
                            const float* ptr = bottom_blob.channel(q).row(i * d + z);
                            depth_values[z] = std::make_pair(ptr[j], z);
                        }

                        if (sorted)
                        {
                            std::partial_sort(depth_values.begin(),
                                              depth_values.begin() + k,
                                              depth_values.end(),
                                              comp);
                        }
                        else
                        {
                            std::nth_element(depth_values.begin(),
                                             depth_values.begin() + k - 1,
                                             depth_values.end(),
                                             comp);
                            std::sort(depth_values.begin(),
                                      depth_values.begin() + k,
                                      comp);
                        }

                        // 写回k个结果
                        for (int z = 0; z < k; z++)
                        {
                            float* outptr = top_blob_values.channel(q).row(i * k + z);
                            float* indices = top_blob_indices.channel(q).row(i * k + z);
                            outptr[j] = depth_values[z].first;
                            indices[j] = static_cast<float>(depth_values[z].second);
                        }
                    }
                }
            }
        }
        else if (axis == 1)
        {
            // 深度方向上;w不变，高度h变为k
            top_blob_values.create(w, h, d, k, elemsize, opt.blob_allocator);
            top_blob_indices.create(w, h, d, k, elemsize, opt.blob_allocator);

            for (int z = 0; z < d; z++)
            {
                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        std::vector<std::pair<float, int> > channel_values(channels);
                        // 修正: 从两个channel中收集相同位置的值
                        for (int c = 0; c < channels; c++)
                        {
                            const float* ptr = bottom_blob.channel(c);
                            int offset = (z * h + i) * w + j;
                            channel_values[c] = std::make_pair(ptr[offset], c);
                        }

                        if (sorted)
                        {
                            std::partial_sort(channel_values.begin(),
                                              channel_values.begin() + k,
                                              channel_values.end(),
                                              comp);
                        }
                        else
                        {
                            std::nth_element(channel_values.begin(),
                                             channel_values.begin() + k - 1,
                                             channel_values.end(),
                                             comp);
                        }

                        // 修正: 直接写入到对应的depth位置
                        float* outptr = top_blob_values.channel(0);
                        float* indices = top_blob_indices.channel(0);
                        int out_offset = (z * h + i) * w + j;
                        outptr[out_offset] = channel_values[0].first;
                        indices[out_offset] = static_cast<float>(channel_values[0].second);
                    }
                }
            }
        }
        else if (axis == 11)
        {
            // 创建输出blob
            top_blob_values.create(w, h, d, k, elemsize, opt.blob_allocator);
            top_blob_indices.create(w, h, d, k, elemsize, opt.blob_allocator);

            if (top_blob_values.empty() || top_blob_indices.empty())
                return -100;

            // 遍历每个位置
            float* outptr = top_blob_values.channel(0);
            float* indices = top_blob_indices.channel(0);

            for (int z = 0; z < d; z++)
            {
                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        // 比较两个channel在当前位置的值
                        const float* ptr0 = bottom_blob.channel(0);
                        const float* ptr1 = bottom_blob.channel(1);
                        int offset = (z * h + i) * w + j;

                        float val0 = ptr0[offset];
                        float val1 = ptr1[offset];

                        // 写入最大值和对应索引
                        if (val0 >= val1)
                        {
                            outptr[offset] = val0;
                            indices[offset] = 0;
                        }
                        else
                        {
                            outptr[offset] = val1;
                            indices[offset] = 1;
                        }
                    }
                }
            }
        }
        else if (axis == 2)
        {
            // 修改blob创建维度顺序
            top_blob_values.create(w, k, d, channels, elemsize, opt.blob_allocator);
            top_blob_indices.create(w, k, d, channels, elemsize, opt.blob_allocator);

            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob_values.channel(q);
                float* indices = top_blob_indices.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < w; i++)
                    {
                        std::vector<std::pair<float, int> > row_scores(h);
                        for (int j = 0; j < h; j++)
                        {
                            // 修正offset计算
                            int offset = (z * h + j) * w + i;
                            row_scores[j] = std::make_pair(ptr[offset], j);
                        }

                        if (sorted)
                        {
                            std::partial_sort(row_scores.begin(),
                                              row_scores.begin() + k,
                                              row_scores.end(),
                                              comp);
                        }
                        else
                        {
                            std::nth_element(row_scores.begin(),
                                             row_scores.begin() + k - 1,
                                             row_scores.end(),
                                             comp);
                        }

                        // 写入结果到正确位置
                        outptr[z * w + i] = row_scores[0].first;
                        indices[z * w + i] = static_cast<float>(row_scores[0].second);
                    }
                }
            }
        }
        else if (axis == 3 || axis == -1)
        {
            // 在w维度上进行TopK
            top_blob_values.create(k, h, d, channels, elemsize, opt.blob_allocator);
            top_blob_indices.create(k, h, d, channels, elemsize, opt.blob_allocator);

            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        std::vector<std::pair<float, int> > row_values(w);
                        // 收集width维度数据
                        for (int j = 0; j < w; j++)
                        {
                            const float* ptr = bottom_blob.channel(q).row(i * d + z);
                            row_values[j] = std::make_pair(ptr[j], j);
                        }

                        if (sorted)
                        {
                            std::partial_sort(row_values.begin(),
                                              row_values.begin() + k,
                                              row_values.end(),
                                              comp);
                        }
                        else
                        {
                            // 使用nth_element找到第k大的元素
                            std::nth_element(row_values.begin(),
                                             row_values.begin() + k - 1,
                                             row_values.end(),
                                             comp);
                            // 对前k个元素排序以保持一致性
                            std::sort(row_values.begin(), row_values.begin() + k, comp);
                        }

                        // 写回结果
                        for (int j = 0; j < k; j++)
                        {
                            float* outptr = top_blob_values.channel(q).row(i * d + z);
                            float* indices = top_blob_indices.channel(q).row(i * d + z);
                            outptr[j] = row_values[j].first;
                            indices[j] = static_cast<float>(row_values[j].second);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

} // namespace ncnn