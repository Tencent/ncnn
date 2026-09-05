// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

#if NCNN_WEIGHT_QUANT
#if NCNN_BF16
#include "gemm_wq_int8_bf16s.h"

void quantize_A_tile_wq_int8_bf16s_avx512vnni_avx512bf16(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    quantize_A_tile_wq_int8_bf16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
}

void transpose_quantize_A_tile_wq_int8_bf16s_avx512vnni_avx512bf16(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    transpose_quantize_A_tile_wq_int8_bf16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
}
#endif // NCNN_BF16
#endif // NCNN_WEIGHT_QUANT

} // namespace ncnn
