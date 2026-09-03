// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dequantize_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#include "dequantize_fp16s.h"

void dequantize_fp16s_asimdhp(const int* intptr, unsigned short* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    dequantize_fp16s(intptr, (__fp16*)ptr, scale_data, bias_data, elemcount, elempack);
}

} // namespace ncnn
