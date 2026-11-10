// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "rotaryembed_bf16s.h"

int rotaryembed_bf16s_zvfbfmin(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, int interleaved, const Option& opt)
{
    return rotaryembed_bf16s(bottom_blobs, top_blobs, interleaved, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
