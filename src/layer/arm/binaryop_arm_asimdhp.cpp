// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "binaryop_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#include "binaryop_fp16s.h"

void binary_op_vector_fp16s_asimdhp(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int aw, int bw, int ap, int bp, int op_type)
{
    binary_op_vector_fp16s(ptr, ptr1, outptr, aw, bw, ap, bp, op_type);
}

} // namespace ncnn
