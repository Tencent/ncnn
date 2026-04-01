// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <float.h>

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "softmax_bf16s.h"

void softmax_bf16s_sse_avx512bf16(unsigned short* _ptr, int elemcount, int elempack)
{
    softmax_bf16s_sse(_ptr, elemcount, elempack);
}

void softmax_bf16s_pack1_sse_avx512bf16(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    softmax_bf16s_pack1_sse(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}

void softmax_bf16s_pack4_sse_avx512bf16(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    softmax_bf16s_pack4_sse(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}

void softmax_bf16s_pack8_sse_avx512bf16(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    softmax_bf16s_pack8_sse(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}

void softmax_bf16s_pack16_sse_avx512bf16(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    softmax_bf16s_pack16_sse(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}

} // namespace ncnn
