// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "argmax.h"

#include <math.h>
#include <functional>

namespace ncnn {

ArgMax::ArgMax()
{
    one_blob_only = true;
    abs = 0;
}

int ArgMax::load_param(const ParamDict& pd)
{
    out_max_val = pd.get(0, 0);
    topk = pd.get(1, 1);
    abs = pd.get(2, 0);

    return 0;
}

int ArgMax::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int size = bottom_blob.total();

    if (out_max_val)
        top_blob.create(topk, 2, 4u, opt.blob_allocator);
    else
        top_blob.create(topk, 1, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* ptr = bottom_blob;

    // fast path for the common topk == 1 case: no top-k materialization.
    // two vectorizable passes: an fmax reduction, then the last index equal
    // to the max -- the same tie order as the partial sort below
    if (topk == 1)
    {
        float max_val = abs ? fabsf(ptr[0]) : ptr[0];
        for (int i = 1; i < size; i++)
        {
            const float v = abs ? fabsf(ptr[i]) : ptr[i];
            max_val = fmaxf(max_val, v);
        }

        int index = 0;
        for (int i = 0; i < size; i++)
        {
            const float v = abs ? fabsf(ptr[i]) : ptr[i];
            index = (v == max_val) ? i : index;
        }

        float* outptr = top_blob;
        if (out_max_val)
        {
            outptr[0] = max_val;
            outptr[topk] = index;
        }
        else
        {
            outptr[0] = index;
        }

        return 0;
    }

    // partial sort topk with index
    // optional value
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(ptr[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    float* outptr = top_blob;
    if (out_max_val)
    {
        float* valptr = outptr + topk;
        for (int i = 0; i < topk; i++)
        {
            outptr[i] = vec[i].first;
            valptr[i] = vec[i].second;
        }
    }
    else
    {
        for (int i = 0; i < topk; i++)
        {
            outptr[i] = vec[i].second;
        }
    }

    return 0;
}

} // namespace ncnn
