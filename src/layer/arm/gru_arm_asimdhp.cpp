// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gru_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"

namespace ncnn {

#include "gru_fp16s.h"

void gru_transform_kernel_fp16s_asimdhp(const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& weight_xc_data_packed, Mat& bias_c_data_packed, Mat& weight_hc_data_packed, int size, int num_output)
{
    gru_transform_kernel_fp16s(weight_xc, bias_c, weight_hc, weight_xc_data_packed, bias_c_data_packed, weight_hc_data_packed, size, num_output);
}

int gru_fp16s_asimdhp(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    return gru_fp16s(bottom_blob, top_blob, reverse, weight_xc, bias_c, weight_hc, hidden_state, opt);
}

} // namespace ncnn
