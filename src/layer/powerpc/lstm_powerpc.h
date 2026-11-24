// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LSTM_POWERPC_H
#define LAYER_LSTM_POWERPC_H

#include "lstm.h"

namespace ncnn {

class LSTM_powerpc : public LSTM
{
public:
    LSTM_powerpc();

    virtual int load_model(const ModelBin& mb);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_LSTM_POWERPC_H
