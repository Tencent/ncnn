// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "noop.h"
#include "cpu.h"

namespace ncnn {

Noop::Noop()
{
    support_inplace = true;
    support_packing = true;
    support_fp16_storage = cpu_support_arm_asimdhp() || cpu_support_riscv_zvfh();
    support_bf16_storage = true;
}

int Noop::forward_inplace(std::vector<Mat>& /*bottom_top_blobs*/, const Option& /*opt*/) const
{
    return 0;
}

} // namespace ncnn
