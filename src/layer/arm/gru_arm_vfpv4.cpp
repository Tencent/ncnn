// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "layer.h"
#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#include "gru_int8.h"

void gru_int8_gate_output_vfpv4(const Mat& gates, Mat& hidden_state, Mat& top_blob, int ti, int elemtype, const Option& opt)
{
    gru_int8_gate_output(gates, hidden_state, top_blob, ti, elemtype, opt);
}

} // namespace ncnn
