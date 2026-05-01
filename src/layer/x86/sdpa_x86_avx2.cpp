// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
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

#include "sdpa_x86_int8.h"

#if __AVX2__
// force emit inline function symbols for non-avx2 runtime dispatch
static void __attribute__((used)) sdpa_x86_int8_avx2_dummy()
{
    // call through volatile function pointer to force instantiation
    void (* volatile f1)(const float*, signed char*, float*, int) = dynamic_quantize_blockwise_avx2;
    void (* volatile f2)(const float*, signed char*, float*, int) = dynamic_quantize_rowwise_avx2;
    int (* volatile f3)(const signed char*, const signed char*, int) = qk_int8_dot_block_avx2;
    void (* volatile f4)(float*, const signed char*, const signed char*, const float*, const float*, int, int, int, float) = decode_qk_dot_int8_avx2;
    void (* volatile f5)(float*, const signed char*, const signed char*, float, const float*, int, int, float) = qk_int8_gemm_row_avx2;
    void (* volatile f6)(float*, const signed char*, const signed char*, const float*, const float*, int, int, int, float) = qk_int8_gemm_tiled_avx2;
    void (* volatile f7)(float*, const float*, const signed char*, const float*, int, int, int) = decode_pv_gemv_int8_avx2;
    void (* volatile f8)(float*, const float*, const signed char*, const float*, int, int) = pv_float_int8_gemm_row_avx2;
    void (* volatile f9)(float*, float, const signed char*, int) = pv_float_int8_fma_block_avx2;
    void (* volatile f10)(float*, const float*, const signed char*, const float*, int, int, int) = pv_float_int8_gemm_tile_avx2;
    (void)f1; (void)f2; (void)f3; (void)f4; (void)f5;
    (void)f6; (void)f7; (void)f8; (void)f9; (void)f10;
}
#endif // __AVX2__

} // namespace ncnn
