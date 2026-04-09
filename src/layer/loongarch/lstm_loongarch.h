// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LSTM_LOONGARCH_H
#define LAYER_LSTM_LOONGARCH_H

#include "lstm.h"

namespace ncnn {

class LSTM_loongarch : public LSTM
{
public:
    LSTM_loongarch();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_LSTM_LOONGARCH_H
