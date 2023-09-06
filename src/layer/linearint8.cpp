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

#include "linearint8.h"

namespace ncnn {

LinearInt8::LinearInt8()
{
    one_blob_only = true;
    support_inplace = false;
}

int LinearInt8::load_param(const ParamDict& pd)
{
    in_dim = pd.get(0, 0);
    out_dim = pd.get(1, 0);
    group_size = pd.get(2, 1);
    if (in_dim * out_dim % group_size)
        return -1;
    return 0;
}

int LinearInt8::load_model(const ModelBin& mb)
{
    scales = mb.load(in_dim * out_dim / group_size, 1);
    weight = mb.load(in_dim * out_dim, 0);
    if (weight.elemsize != 1)
        return -1;
    return 0;
}

int LinearInt8::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.dims != 2 || bottom_blob.w != in_dim)
        return -1;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    top_blob.create(out_dim, h, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int8_t* wt = (const int8_t*)weight;

    for (int j = 0; j < h; j++)
    {
        const float* m = bottom_blob.row(j);
        float* out = top_blob.row(j);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_dim; p++)
        {
            int base = w * p;
            float acc = 0.0f;
            for (int i = 0; i < w; i++)
            {
                int index = base + i;
                acc += m[i] * wt[index] * scales[index / group_size];
            }
            out[p] = acc;
        }
    }

    return 0;
}

} // namespace ncnn
