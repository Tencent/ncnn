// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "cpu.h"
#include "mat.h"
#include "layer.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "lstm_int8.h"

void lstm_transform_weight_int8_avxvnni(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, int hidden_size, const Option& opt)
{
    lstm_transform_weight_int8(weight_xc, weight_xc_int8_scales, weight_hc, weight_hc_int8_scales, bias_c, weight_data_tm, weight_data_tm_int8_descales, bias_c_tm, size, num_output, num_directions, hidden_size, opt);
}

void lstm_dynamic_quantize_scale2int8_avxvnni(const float* ptr, int size, float scale, signed char* outptr)
{
    lstm_dynamic_quantize_scale2int8(ptr, size, scale, outptr);
}

void lstm_int8_avxvnni(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    lstm_int8(bottom_blob_int8, bottom_blob_int8_descales, top_blob, reverse, weight_data_tm, weight_data_tm_int8_descales, bias_c, weight_hr, hidden_state, cell_state, opt);
}

} // namespace ncnn
