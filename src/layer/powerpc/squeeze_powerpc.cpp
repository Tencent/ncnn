// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "squeeze_powerpc.h"

namespace ncnn {

Squeeze_powerpc::Squeeze_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

int Squeeze_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack != 1)
    {
        // squeeze is metadata-only, but the packing axis (outermost dim) may be
        // squeezed away; operate on the unpacked layout and re-pack afterwards
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        if (bottom_blob_unpacked.empty())
            return -100;

        Mat top_blob_unpacked;
        int ret = Squeeze::forward(bottom_blob_unpacked, top_blob_unpacked, opt);
        if (ret != 0)
            return ret;

        int repack_count = 0;
        if (top_blob_unpacked.dims == 1)
            repack_count = top_blob_unpacked.w;
        if (top_blob_unpacked.dims == 2)
            repack_count = top_blob_unpacked.h;
        if (top_blob_unpacked.dims == 3 || top_blob_unpacked.dims == 4)
            repack_count = top_blob_unpacked.c;

        if (opt.use_packing_layout && repack_count % 4 == 0)
        {
            Mat top_blob_packed;
            convert_packing(top_blob_unpacked, top_blob_packed, 4, opt);
            if (!top_blob_packed.empty())
            {
                top_blob = top_blob_packed;
                return 0;
            }
        }

        top_blob = top_blob_unpacked;
        return 0;
    }

    // metadata-only, delegate to the base class
    return Squeeze::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
