// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lstm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"

namespace ncnn {

#include "lstm_fp16s.h"

void lstm_transform_kernel_fp16s_asimdhp(const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& weight_xc_data_packed_dr, Mat& bias_c_data_packed_dr, Mat& weight_hc_data_packed_dr, int size, int num_output, int hidden_size, bool use_fp16_arithmetic)
{
    lstm_transform_kernel_fp16s(weight_xc, bias_c, weight_hc, weight_xc_data_packed_dr, bias_c_data_packed_dr, weight_hc_data_packed_dr, size, num_output, hidden_size, use_fp16_arithmetic);
}

int lstm_fp16s_asimdhp(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    return lstm_fp16s(bottom_blob, top_blob, reverse, weight_xc, bias_c, weight_hc, weight_hr, hidden_state, cell_state, opt);
}

} // namespace ncnn
