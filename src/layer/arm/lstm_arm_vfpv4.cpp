// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "layer.h"
#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#include "lstm_int8.h"

void lstm_int8_gate_output_vfpv4(const Mat& gates, const Mat& weight_hr, Mat& hidden_state, Mat& tmp_hidden_state, Mat& cell_state, Mat& top_blob, int ti, int elemtype, const Option& opt)
{
    lstm_int8_gate_output(gates, weight_hr, hidden_state, tmp_hidden_state, cell_state, top_blob, ti, elemtype, opt);
}

} // namespace ncnn
