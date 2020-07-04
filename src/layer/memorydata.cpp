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

#include "memorydata.h"

namespace ncnn {

MemoryData::MemoryData()
{
    one_blob_only = false;
    support_inplace = false;
}

int MemoryData::load_param(const ParamDict& pd)
{
    w = pd.get(0, 0);
    h = pd.get(1, 0);
    c = pd.get(2, 0);

    return 0;
}

int MemoryData::load_model(const ModelBin& mb)
{
    if (c != 0)
    {
        data = mb.load(w, h, c, 1);
    }
    else if (h != 0)
    {
        data = mb.load(w, h, 1);
    }
    else if (w != 0)
    {
        data = mb.load(w, 1);
    }
    else // 0 0 0
    {
        data.create(1);
    }
    if (data.empty())
        return -100;

    return 0;
}

int MemoryData::forward(const std::vector<Mat>& /*bottom_blobs*/, std::vector<Mat>& top_blobs, const Option& opt) const
{
    Mat& top_blob = top_blobs[0];

    top_blob = data.clone(opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
