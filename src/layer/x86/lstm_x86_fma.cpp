// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "layer.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "lstm_fp32.h"

int lstm_fma(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    return lstm(bottom_blob, top_blob, reverse, weight_xc, bias_c, weight_hc, weight_hr, hidden_state, cell_state, opt);
}

#include "lstm_int8.h"

void lstm_int8_fma(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    lstm_int8(bottom_blob_int8, bottom_blob_int8_descales, top_blob, reverse, weight_data_tm, weight_data_tm_int8_descales, bias_c, weight_hr, hidden_state, cell_state, opt);
}

} // namespace ncnn
