// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LSTM_X86_H
#define LAYER_LSTM_X86_H

#include "lstm.h"

namespace ncnn {

class LSTM_x86 : public LSTM
{
public:
    LSTM_x86();

    virtual int create_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_INT8
    int create_pipeline_int8(const Option& opt);
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    Mat weight_xc_data_packed;
    Mat bias_c_data_packed;
    Mat weight_hc_data_packed;

    Mat weight_data_tm;

#if NCNN_INT8
    Mat weight_data_tm_int8_descales;
#endif
};

} // namespace ncnn

#endif // LAYER_LSTM_X86_H
