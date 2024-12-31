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
// #include <functional>

namespace ncnn {

TopK::TopK()
{
    one_blob_only = true;    // 只需要一个输入 blob
    support_inplace = false; // 是否支持原地运算
    k = 1;
    axis = 0;
    largest = 1;
    sorted = 1;
}

int TopK::load_param(const ParamDict& pd)
{
    k = pd.get(0, 1);
    axis = pd.get(1, 0);
    largest = pd.get(2, 1);
    sorted = pd.get(3, 1);
    return 0;
}
int TopK::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int size = (int)bottom_blob.total();
    int k_ = k;
    if (k_ > size) k_ = size;

    const float* ptr = bottom_blob.row(0);

    std::vector<std::pair<float, int> > vec;
#if !NCNN_SIMPLESTL
    vec.reserve(size);
#else
    for (int i = 0; i < size; i++)
    {
        vec.push_back(std::make_pair(ptr[i], i));
    }

    // [](const std::pair<float, int>& a, const std::pair<float, int>& b) {return a.first > b.first;}); // fix Lambda with lower version of C++

    if (largest == 1)
    {
        std::partial_sort(vec.begin(), vec.begin() + k_, vec.end(), std::greater<std::pair<float, int> >());
    }
    else
    {
        std::partial_sort(vec.begin(), vec.begin() + k_, vec.end(), std::less<std::pair<float, int> >());
    }

    if (sorted)
    {
        if (largest == 1)
        {
            std::partial_sort(vec.begin(), vec.begin() + k_, vec.end(), std::greater<std::pair<float, int> >());
        }
        else
        {
            std::partial_sort(vec.begin(), vec.begin() + k_, vec.end(), std::less<std::pair<float, int> >());
        }
    }

    top_blob.create(k_, 1, 4u, 1, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    float* outptr = top_blob;
    for (int i = 0; i < k_; i++)
    {
        outptr[i] = vec[i].first;
    }

    return 0;
}

} // namespace ncnn