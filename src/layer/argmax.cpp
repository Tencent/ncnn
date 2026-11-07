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
    // no top-k materialization. a single pass in the integer domain: the
    // IEEE-754 bit pattern maps to an order-preserving unsigned key
    // (negative floats flipped, non-negative floats sign-bit set), NaN
    // entries are skipped by a raw-bit test, and equal keys resolve to the
    // larger index -- the same tie order as the partial sort below for all
    // non-NaN inputs. integer comparison keeps the reduction immune to
    // -ffast-math NaN folding, which otherwise misorders mixed-NaN inputs
    // (a no-NaN-assumption float max drops or keeps NaNs depending on the
    // emitted instruction, and an NaN == NaN equality scan folds to true)
    if (topk == 1 && bottom_blob.dims == 1)
    {
        const int count = bottom_blob.w;

        unsigned int best_key = 0;
        int index = -1;
        for (int i = 0; i < count; i++)
        {
            unsigned int u;
            memcpy(&u, &ptr[i], 4);

            // NaN: exponent all ones, mantissa nonzero -- no magnitude order
            if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0)
                continue;

            if (abs)
                u &= 0x7fffffffu; // |x|: clear the sign bit

            if (u & 0x80000000u)
                u = ~u;
            else
                u |= 0x80000000u;

            if (index < 0 || u >= best_key)
            {
                best_key = u;
                index = i;
            }
        }

        if (index < 0)
        {
            // every element was NaN: the partial sort orders NaN entries
            // with the larger index first, so the established result is
            // the last index
            index = count - 1;
        }

        float* outptr = top_blob;
        if (out_max_val)
        {
            // report the selected element's own value bits (magnitude with
            // abs) so the value always agrees with the reported index --
            // signed zero included, where the integer order ranks +0 over
            // -0 while the float comparator treated them as tied
            outptr[0] = abs ? fabsf(ptr[index]) : ptr[index];
            outptr[topk] = index;
        }
        else
        {
            outptr[0] = index;
        }

        return 0;
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
