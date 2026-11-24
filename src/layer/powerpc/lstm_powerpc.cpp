// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lstm_powerpc.h"

namespace ncnn {

LSTM_powerpc::LSTM_powerpc()
{
    // the generic LSTM compute runs on unpacked blobs only
    support_packing = false;
}

int LSTM_powerpc::load_model(const ModelBin& mb)
{
    return LSTM::load_model(mb);
}

int LSTM_powerpc::create_pipeline(const Option& opt)
{
    return LSTM::create_pipeline(opt);
}

int LSTM_powerpc::destroy_pipeline(const Option& opt)
{
    return LSTM::destroy_pipeline(opt);
}

static Mat unpack_blob(const Mat& blob, const Option& opt)
{
    if (blob.elempack == 1)
        return blob;

    Mat unpacked;
    convert_packing(blob, unpacked, 1, opt);
    return unpacked;
}

int LSTM_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack != 1)
    {
        Mat unpacked = unpack_blob(bottom_blob, opt);
        if (unpacked.empty())
            return -100;

        return LSTM::forward(unpacked, top_blob, opt);
    }

    return LSTM::forward(bottom_blob, top_blob, opt);
}

int LSTM_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];

    if (bottom_blob.elempack != 1 || (bottom_blobs.size() == 3 && (bottom_blobs[1].elempack != 1 || bottom_blobs[2].elempack != 1)))
    {
        Mat unpacked = unpack_blob(bottom_blob, opt);
        if (unpacked.empty())
            return -100;

        std::vector<Mat> unpacked_blobs;
        unpacked_blobs.push_back(unpacked);
        for (size_t i = 1; i < bottom_blobs.size(); i++)
        {
            Mat unpacked_state = unpack_blob(bottom_blobs[i], opt);
            if (unpacked_state.empty())
                return -100;
            unpacked_blobs.push_back(unpacked_state);
        }

        return LSTM::forward(unpacked_blobs, top_blobs, opt);
    }

    return LSTM::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
