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

#include "argmax.h"
#include <algorithm>
#include <functional>

namespace ncnn {

DEFINE_LAYER_CREATOR(ArgMax)

ArgMax::ArgMax()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int ArgMax::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d", &out_max_val, &topk);
    if (nscan != 2)
    {
        fprintf(stderr, "ArgMax load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int ArgMax::load_param_bin(FILE* paramfp)
{
    fread(&out_max_val, sizeof(int), 1, paramfp);

    fread(&topk, sizeof(int), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int ArgMax::load_param(const unsigned char*& mem)
{
    out_max_val = *(int*)(mem);
    mem += 4;

    topk = *(int*)(mem);
    mem += 4;

    return 0;
}

int ArgMax::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int size = bottom_blob.total();

    if (out_max_val)
        top_blob.create(topk, 2);
    else
        top_blob.create(topk, 1);
    if (top_blob.empty())
        return -100;

    const float* ptr = bottom_blob;

    // partial sort topk with index
    // optional value
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(ptr[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                        std::greater< std::pair<float, int> >());

    float* outptr = top_blob;
    if (out_max_val)
    {
        float* valptr = outptr + topk;
        for (int i=0; i<topk; i++)
        {
            outptr[i] = vec[i].first;
            valptr[i] = vec[i].second;
        }
    }
    else
    {
        for (int i=0; i<topk; i++)
        {
            outptr[i] = vec[i].second;
        }
    }

    return 0;
}

} // namespace ncnn
