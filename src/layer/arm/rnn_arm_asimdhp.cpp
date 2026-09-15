// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rnn_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"

namespace ncnn {

#include "rnn_fp16s.h"

void rnn_transform_kernel_fp16s_asimdhp(const Mat& weight_xc, const Mat& weight_hc, Mat& weight_xc_data_packed, Mat& weight_hc_data_packed, int size, int num_output, bool use_fp16_arithmetic)
{
    rnn_transform_kernel_fp16s(weight_xc, weight_hc, weight_xc_data_packed, weight_hc_data_packed, size, num_output, use_fp16_arithmetic);
}

int rnn_fp16s_asimdhp(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    return rnn_fp16s(bottom_blob, top_blob, reverse, weight_xc, bias_c, weight_hc, hidden_state, opt);
}

} // namespace ncnn
