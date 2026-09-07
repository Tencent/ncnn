// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <float.h>

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "softmax_bf16s.h"

void softmax_bf16s_avxneconvert(unsigned short* _ptr, int elemcount, int elempack)
{
    softmax_bf16s(_ptr, elemcount, elempack);
}

void softmax_bf16s_pack1_avxneconvert(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    softmax_bf16s_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}

void softmax_bf16s_pack4_avxneconvert(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    softmax_bf16s_pack4(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}

void softmax_bf16s_pack8_avxneconvert(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    softmax_bf16s_pack8(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}

#endif // NCNN_BF16

} // namespace ncnn
