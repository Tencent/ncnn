// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_powerpc.h"

namespace ncnn {

SDPA_powerpc::SDPA_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

// packing-aware wrapper: the generic sdpa kernel is not packing aware, so
// unpack every packed input (query/key/value, attn mask and past kv cache) to
// elempack 1 and delegate to the base class. the top blob is a 3D
// (out_embed_dim x src_seqlen x num_heads) output whose channel dim is the
// semantic head index, so it is returned unchanged.
int SDPA_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    bool any_packed = false;
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (bottom_blobs[i].elempack != 1)
        {
            any_packed = true;
            break;
        }
    }

    if (!any_packed)
        return SDPA::forward(bottom_blobs, top_blobs, opt);

    std::vector<Mat> bottom_blobs_unpacked(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (bottom_blobs[i].elempack != 1)
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

    return SDPA::forward(bottom_blobs_unpacked, top_blobs, opt);
}

} // namespace ncnn
