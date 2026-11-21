// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gru_powerpc.h"

namespace ncnn {

GRU_powerpc::GRU_powerpc()
{
    // the generic GRU compute runs on unpacked blobs only
    support_packing = false;
}

int GRU_powerpc::load_model(const ModelBin& mb)
{
    return GRU::load_model(mb);
}

int GRU_powerpc::create_pipeline(const Option& opt)
{
    return GRU::create_pipeline(opt);
}

int GRU_powerpc::destroy_pipeline(const Option& opt)
{
    return GRU::destroy_pipeline(opt);
}

static Mat unpack_blob(const Mat& blob, const Option& opt)
{
    if (blob.elempack == 1)
        return blob;

    Mat unpacked;
    convert_packing(blob, unpacked, 1, opt);
    return unpacked;
}

int GRU_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack != 1)
    {
        Mat unpacked = unpack_blob(bottom_blob, opt);
        if (unpacked.empty())
            return -100;

        return GRU::forward(unpacked, top_blob, opt);
    }

    return GRU::forward(bottom_blob, top_blob, opt);
}

int GRU_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];

    if (bottom_blob.elempack != 1 || (bottom_blobs.size() == 2 && bottom_blobs[1].elempack != 1))
    {
        Mat unpacked = unpack_blob(bottom_blob, opt);
        if (unpacked.empty())
            return -100;

        std::vector<Mat> unpacked_blobs;
        unpacked_blobs.push_back(unpacked);
        for (size_t i = 1; i < bottom_blobs.size(); i++)
        {
            Mat unpacked_hidden = unpack_blob(bottom_blobs[i], opt);
            if (unpacked_hidden.empty())
                return -100;
            unpacked_blobs.push_back(unpacked_hidden);
        }

        return GRU::forward(unpacked_blobs, top_blobs, opt);
    }

    return GRU::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
