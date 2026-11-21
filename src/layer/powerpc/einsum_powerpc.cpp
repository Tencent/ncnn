// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "einsum_powerpc.h"

namespace ncnn {

Einsum_powerpc::Einsum_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

// packing-aware wrapper: the generic einsum kernel is not packing aware, so
// unpack every packed input to elempack 1 and delegate to the base class.
// the output has arbitrary dims (1D..4D), so it is returned unchanged.
int Einsum_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
        return Einsum::forward(bottom_blobs, top_blobs, opt);

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

    return Einsum::forward(bottom_blobs_unpacked, top_blobs, opt);
}

} // namespace ncnn
