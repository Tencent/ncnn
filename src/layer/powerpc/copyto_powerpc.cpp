// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "copyto_powerpc.h"

namespace ncnn {

CopyTo_powerpc::CopyTo_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

int CopyTo_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs[0].elempack != 1 || bottom_blobs[1].elempack != 1)
    {
        // copy_to_image works on unpacked rows; operate on the unpacked layout
        // and re-pack the output afterwards
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        std::vector<Mat> bottom_blobs_unpacked;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            if (bottom_blobs[b].elempack != 1)
            {
                Mat bottom_blob_unpacked;
                convert_packing(bottom_blobs[b], bottom_blob_unpacked, 1, opt_pack1);
                if (bottom_blob_unpacked.empty())
                    return -100;
                bottom_blobs_unpacked.push_back(bottom_blob_unpacked);
            }
            else
            {
                bottom_blobs_unpacked.push_back(bottom_blobs[b]);
            }
        }

        Mat top_blob_unpacked;
        std::vector<Mat> top_blobs_unpacked;
        top_blobs_unpacked.push_back(top_blob_unpacked);

        int ret = CopyTo::forward(bottom_blobs_unpacked, top_blobs_unpacked, opt);
        if (ret != 0)
            return ret;

        Mat& top_blob = top_blobs[0];
        top_blob_unpacked = top_blobs_unpacked[0];

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

    // rows are copied with memcpy, already optimal on altivec
    return CopyTo::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
