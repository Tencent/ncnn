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
#if !NCNN_SIMPLESTL
// 兼容vs编译器
#include <functional>
#endif

namespace ncnn {

// auto comp = [this](const std::pair<float, int> &a, const std::pair<float, int> &b)
// {
//     if (a.first == b.first)
//         return a.second < b.second; // 值相等时按索引升序排序
//     return this->largest ? (a.first > b.first) : (a.first < b.first);
// };

// simplestl兼容写法
struct CompareFunc
{
    bool largest;
    CompareFunc(bool l)
        : largest(l)
    {
    }
    bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) const
    {
        if (a.first == b.first)
            return a.second < b.second;
        return largest ? (a.first > b.first) : (a.first < b.first);
    }
};

void TopK::do_sort(std::vector<std::pair<float, int> >& vec) const
{
    CompareFunc comp(largest);
    if (sorted)
    {
        std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), comp);
    }
    else
    {
#if !NCNN_SIMPLESTL
        std::nth_element(vec.begin(), vec.begin() + k - 1, vec.end(), comp);
        std::sort(vec.begin(), vec.begin() + k, comp);
#else
        for (int i = 0; i < k; i++)
        {
            for (int j = vec.size() - 1; j > i; j--)
            {
                if (comp(vec[j], vec[j - 1]))
                {
                    std::swap(vec[j], vec[j - 1]);
                }
            }
        }
#endif
    }
}

TopK::TopK()
{
    one_blob_only = false;   // 只需要一个输入 blob
    support_inplace = false; // 是否支持原地运算
}

int TopK::load_param(const ParamDict& pd)
{
    k = pd.get(0, 1); // [获取参数，默认值1]
    axis = pd.get(1, 0);
    largest = pd.get(2, 1);
    sorted = pd.get(3, 1);
    // printf("参数加载k=%d, axis=%d, largest=%d, sorted=%d\n", k, axis, largest, sorted);
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

    // printf("dims=%d, w=%d, h=%d, d=%d, channels=%d, elemsize=%zu\n", dims, w, h, d, channels, elemsize);
    // 检查k值是否有效
    if (k <= 0 || k > w * h * channels)
    {
        return -1;
    }

    // 创建输出Mat
    Mat& top_blob_values = top_blobs[0];  // values
    Mat& top_blob_indices = top_blobs[1]; // indices

    // 根据dims创建不同维度的输出
    if (dims == 1)
    {
        // 创建输出blob
        top_blob_values.create(k, elemsize, opt.blob_allocator);
        top_blob_indices.create(k, sizeof(int), opt.blob_allocator);

        const float* ptr = bottom_blob;
        float* outptr = top_blob_values;
        int* indices = top_blob_indices;
        // 创建pair数组用于排序
        std::vector<std::pair<float, int> > vec(w);
        for (int i = 0; i < w; i++)
        {
            vec[i] = std::make_pair(ptr[i], i);
        }

        // 根据sorted参数选择排序方式
        do_sort(vec);

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

            for (int j = 0; j < w; j++) // 对每列进行处理
            {
                std::vector<std::pair<float, int> > vec(h);
                // 收集当前列的所有元素
                for (int i = 0; i < h; i++)
                {
                    vec[i] = std::make_pair(bottom_blob.row(i)[j], i);
                }

                do_sort(vec);

                // 保存结果到对应列
                for (int i = 0; i < k; i++)
                {
                    top_blob_values.row(i)[j] = vec[i].first;
                    top_blob_indices.row<int>(i)[j] = vec[i].second;
                }
            }
        }
        // 在每一列上进行TopK ，axis=-1等价于axis=1
        else
        {
            top_blob_values.create(k, h, elemsize, opt.blob_allocator);
            top_blob_indices.create(k, h, sizeof(int), opt.blob_allocator);

            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float* outptr = top_blob_values.row(i);
                int* indices = top_blob_indices.row<int>(i);

                std::vector<std::pair<float, int> > vec(w);
                for (int j = 0; j < w; j++)
                {
                    vec[j] = std::make_pair(ptr[j], j);
                }

                do_sort(vec);

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
            top_blob_indices.create(w, h, k, sizeof(int), opt.blob_allocator);
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
                    do_sort(channel_values);

                    // 写回结果
                    for (int c = 0; c < k; c++)
                    {
                        float* outptr = top_blob_values.channel(c);
                        int* indices = (int*)top_blob_indices.channel(c);
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
            top_blob_indices.create(w, k, channels, sizeof(int), opt.blob_allocator);
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
                    do_sort(row_scores);

                    // 保存该列的结果
                    for (int i = 0; i < k; i++)
                    {
                        float* outptr = top_blob_values.channel(q).row(i);
                        int* indices = (int*)top_blob_indices.channel(q).row(i);
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
            top_blob_indices.create(k, h, channels, sizeof(int), opt.blob_allocator);
            for (int q = 0; q < channels; q++)
            {
                for (int j = 0; j < h; j++)
                {
                    const float* ptr = bottom_blob.channel(q).row(j);
                    float* outptr = top_blob_values.channel(q).row(j);
                    int* indices = top_blob_indices.channel(q).row<int>(j);

                    std::vector<std::pair<float, int> > vec(w);
                    for (int i = 0; i < w; i++)
                    {
                        vec[i] = std::make_pair(ptr[i], i);
                    }

                    do_sort(vec);

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
            // 在torch中d维度求topk
            top_blob_values.create(w, h, d, k, elemsize, opt.blob_allocator);
            top_blob_indices.create(w, h, d, k, sizeof(int), opt.blob_allocator);

            for (int z = 0; z < d; z++)
            {
                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        // 收集channel维度的值
                        std::vector<std::pair<float, int> > channel_values(channels);
                        for (int c = 0; c < channels; c++)
                        {
                            const float* ptr = bottom_blob.channel(c);
                            int offset = z * h * w + i * w + j;
                            channel_values[c] = std::make_pair(ptr[offset], c);
                        }

                        // 排序
                        do_sort(channel_values);

                        // 保存结果
                        for (int kk = 0; kk < k; kk++)
                        {
                            float* outptr = top_blob_values.channel(kk);
                            int* indptr = top_blob_indices.channel(kk);
                            int out_offset = z * h * w + i * w + j;
                            outptr[out_offset] = channel_values[kk].first;
                            indptr[out_offset] = channel_values[kk].second;
                        }
                    }
                }
            }
        }
        else if (axis == 1)
        {
            // 在torch中c维度求topk
            top_blob_values.create(w, h, k, channels, elemsize, opt.blob_allocator);
            top_blob_indices.create(w, h, k, channels, sizeof(int), opt.blob_allocator);

            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob_values.channel(q);
                int* indptr = top_blob_indices.channel(q);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        // 收集当前(h,w)位置在d维度上的所有值
                        std::vector<std::pair<float, int> > vec(d);
                        for (int z = 0; z < d; z++)
                        {
                            int offset = z * h * w + i * w + j;
                            vec[z] = std::make_pair(ptr[offset], z);
                        }

                        do_sort(vec);

                        // 保存top-k结果
                        for (int z = 0; z < k; z++)
                        {
                            int offset = z * h * w + i * w + j;
                            outptr[offset] = vec[z].first;
                            indptr[offset] = vec[z].second;
                        }
                    }
                }
            }
        }
        else if (axis == 2)
        {
            // 在h维度上进行TopK
            top_blob_values.create(w, k, d, channels, elemsize, opt.blob_allocator);
            top_blob_indices.create(w, k, d, channels, sizeof(int), opt.blob_allocator);

            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob_values.channel(q);
                int* indices = top_blob_indices.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < w; i++)
                    {
                        std::vector<std::pair<float, int> > row_scores(h);
                        for (int j = 0; j < h; j++)
                        {
                            int offset = (z * h + j) * w + i;
                            row_scores[j] = std::make_pair(ptr[offset], j);
                        }

                        do_sort(row_scores);

                        // 写回结果
                        for (int kk = 0; kk < k; kk++)
                        {
                            outptr[(z * k + kk) * w + i] = row_scores[kk].first;
                            indices[(z * k + kk) * w + i] = row_scores[kk].second;
                        }
                    }
                }
            }
        }
        else if (axis == 3 || axis == -1)
        {
            // 在w维度上进行TopK
            top_blob_values.create(k, h, d, channels, elemsize, opt.blob_allocator);
            top_blob_indices.create(k, h, d, channels, sizeof(int), opt.blob_allocator);

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

                        do_sort(row_values);

                        // 写回结果
                        for (int j = 0; j < k; j++)
                        {
                            float* outptr = top_blob_values.channel(q).row(i * d + z);
                            int* indices = top_blob_indices.channel(q).row<int>(i * d + z);
                            outptr[j] = row_values[j].first;
                            indices[j] = row_values[j].second;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

} // namespace ncnn