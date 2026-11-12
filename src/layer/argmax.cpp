// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "argmax.h"

#include <math.h>
#include <string.h>
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

    // fast path for the common reduce-all-over-a-1d-blob topk == 1 case:
    // no top-k materialization. two vectorizable passes: an fmax reduction
    // over the logical elements, then the last index equal to the max --
    // which for signed zeros resolves to the last zero, the same tie the
    // partial sort produces. a NaN presence check (two integer OR
    // accumulations, immune to -ffast-math NaN folding) gates the path:
    // with NaNs present the float reduction is unreliable, so fall through
    // to the original partial sort, which stays bitwise faithful
    if (topk == 1 && bottom_blob.dims == 1)
    {
        const int count = bottom_blob.w;

        float max_val = abs ? fabsf(ptr[0]) : ptr[0];
        for (int i = 1; i < count; i++)
        {
            const float v = abs ? fabsf(ptr[i]) : ptr[i];
            max_val = fmaxf(max_val, v);
        }

        int index = count - 1;
        for (int i = count - 1; i >= 0; i--)
        {
            const float v = abs ? fabsf(ptr[i]) : ptr[i];
            if (v == max_val)
            {
                index = i;
                break;
            }
        }

        // NaN presence: a NaN has the all-ones exponent and a nonzero
        // mantissa; the OR accumulations are exact and vectorize
        unsigned int nan_probe = 0;
        for (int i = 0; i < count; i++)
        {
            unsigned int u;
            memcpy(&u, &ptr[i], 4);
            nan_probe |= (u & 0x7f800000u) == 0x7f800000u ? (u & 0x007fffffu) : 0u;
        }

        if (nan_probe == 0)
        {
            float* outptr = top_blob;
            if (out_max_val)
            {
                // the selected element's own value bits: signed zeros and
                // NaN-free magnitudes included
                outptr[0] = abs ? fabsf(ptr[index]) : ptr[index];
                outptr[topk] = index;
            }
            else
            {
                outptr[0] = index;
            }

            return 0;
        }
    }

    // partial sort topk with index
    // optional value; with abs the sort key (and the value reported by
    // out_max_val) is the magnitude. iterate per channel so cstep padding
    // never enters the ranking
    const int channels = bottom_blob.c;
    const int per_channel = bottom_blob.w * bottom_blob.h * bottom_blob.d;
    std::vector<std::pair<float, int> > vec;
    vec.resize((size_t)channels * per_channel);
    for (int q = 0; q < channels; q++)
    {
        const float* chan = bottom_blob.channel(q);
        const size_t base = (size_t)q * per_channel;
        for (int i = 0; i < per_channel; i++)
        {
            vec[base + i] = std::make_pair(abs ? fabsf(chan[i]) : chan[i], base + i);
        }
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
