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

#if __XOP__

void decode_qk_dot_int8_xop(float* s, const signed char* q, const signed char* K, const float* qscales, const float* kscales, int n_start, int block_n, int d, float scale)
{
    decode_qk_dot_int8_xop_kernel(s, q, K, qscales, kscales, n_start, block_n, d, scale);
}

void qk_int8_gemm_row_xop(float* s_row, const signed char* q_row, const signed char* K, float qscale, const float* kscales, int n, int d, float scale)
{
    const int num_blocks = (d + 31) / 32;
    int j = 0;
    for (; j < n; j++)
    {
        const signed char* kptr = K + j * d;
        int sum = 0;
        for (int b = 0; b < num_blocks; b++)
        {
            int off = b * 32;
            int len = std::min(32, d - off);
            if (len <= 0) continue;
            sum += qk_int8_dot_block_xop_kernel(q_row + off, kptr + off, len);
        }
        s_row[j] = (float)sum * qscale * kscales[j] * scale;
    }
}

void qk_int8_gemm_tiled_xop(float* S, const signed char* Q, const signed char* K, const float* qscales, const float* kscales, int m, int n, int d, float scale)
{
    for (int i = 0; i < m; i++)
    {
        qk_int8_gemm_row_xop(S + i * n, Q + i * d, K, qscales[i], kscales, n, d, scale);
    }
}

#endif // __XOP__

} // namespace ncnn
