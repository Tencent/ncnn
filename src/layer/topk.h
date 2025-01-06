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

#ifndef LAYER_TOPK_H
#define LAYER_TOPK_H

#include "layer.h"

namespace ncnn {

class TopK : public Layer
{
public:
    TopK();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int k;
    int axis;
    int largest;
    int sorted;

private:
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
                return a.second < b.second; // 值相等时按索引升序排序
            return largest ? (a.first > b.first) : (a.first < b.first);
        }
    };
    void do_sort(std::vector<std::pair<float, int> >& vec, int k, bool sorted) const
    {
        CompareFunc comp(largest); // 兼容c++03
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
            // 替换 nth_element + sort 组合
            // 使用 bubble_sort 实现相同功能，适配sim_stl
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
};

} // namespace ncnn

#endif // LAYER_TOPK_H
