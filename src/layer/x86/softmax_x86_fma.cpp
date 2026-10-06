// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <float.h>

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

#include "softmax_fp32.h"

void softmax_fma(float* ptr, int elemcount, int elempack)
{
    softmax(ptr, elemcount, elempack);
}

void softmax_pack8_fma(float* ptr, int elemcount, size_t stride, int size1, float* maxptr, float* sumptr)
{
    softmax_pack8(ptr, elemcount, stride, size1, maxptr, sumptr);
}

void softmax_pack4_fma(float* ptr, int elemcount, size_t stride, int size1, float* maxptr, float* sumptr)
{
    softmax_pack4(ptr, elemcount, stride, size1, maxptr, sumptr);
}

void softmax_pack1_fma(float* ptr, int elemcount, size_t stride, int size1, float* maxptr, float* sumptr)
{
    softmax_pack1(ptr, elemcount, stride, size1, maxptr, sumptr);
}

#if NCNN_BF16
#include "softmax_bf16s.h"

void softmax_bf16s_fma(unsigned short* _ptr, int elemcount, int elempack)
{
    softmax_bf16s(_ptr, elemcount, elempack);
}

void softmax_bf16s_pack1_fma(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    softmax_bf16s_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}

void softmax_bf16s_pack4_fma(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    softmax_bf16s_pack4(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}

void softmax_bf16s_pack8_fma(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    softmax_bf16s_pack8(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}
#endif // NCNN_BF16

} // namespace ncnn
