// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "multiheadattention_powerpc.h"

namespace ncnn {

MultiHeadAttention_powerpc::MultiHeadAttention_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

// packing-aware wrapper: the generic multiheadattention kernel is not packing
// aware, so unpack every packed input (q/k/v, attn mask and past kv cache) to
// elempack 1 and delegate to the base class. the top blob is a 2D
// (qdim x src_seqlen) output, so it is returned unchanged.
int MultiHeadAttention_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // a default Mat has elempack 0. an empty kv cache is not packed, and
    // convert_packing leaves it empty, which this wrapper would report as oom.
    bool any_packed = false;
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (!bottom_blobs[i].empty() && bottom_blobs[i].elempack != 1)
        {
            any_packed = true;
            break;
        }
    }

    if (!any_packed)
        return MultiHeadAttention::forward(bottom_blobs, top_blobs, opt);

    std::vector<Mat> bottom_blobs_unpacked(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (!bottom_blobs[i].empty() && bottom_blobs[i].elempack != 1)
        {
            convert_packing(bottom_blobs[i], bottom_blobs_unpacked[i], 1, opt);
            if (bottom_blobs_unpacked[i].empty())
                return -100;
        }
        else
        {
            bottom_blobs_unpacked[i] = bottom_blobs[i];
        }
    }

    return MultiHeadAttention::forward(bottom_blobs_unpacked, top_blobs, opt);
}

} // namespace ncnn
