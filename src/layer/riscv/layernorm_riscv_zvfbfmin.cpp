// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_riscv.h"

#include <math.h>

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "layernorm_bf16s.h"

void layernorm_bf16s_zvfbfmin(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    layernorm_bf16s(ptr, gamma_ptr, beta_ptr, eps, elemcount, elempack);
}

#endif // NCNN_BF16

} // namespace ncnn
