// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

#include "sdpa_x86_bf16s.h"

#if __AVX512BF16__

void decode_qk_dot_bf16s_avx512bf16(float* s, const float* q, const unsigned short* K, int n_start, int block_n, int d, float scale)
{
    decode_qk_dot_bf16s_avx512bf16_kernel(s, q, K, n_start, block_n, d, scale);
}

void decode_pv_gemv_bf16s_avx512bf16(float* out, const float* s, const unsigned short* V, int n_start, int block_n, int out_d)
{
    decode_pv_gemv_bf16s_avx512bf16_kernel(out, s, V, n_start, block_n, out_d);
}

// Explicit instantiations for common embed_dim values
// These allow GCC to fully unroll the inner d-loops at compile time,
// matching the performance of the fp32 template-specialized path.
template void qk_gemm_bf16s_avx512bf16_kernel_t<64>(float*, const float*, const unsigned short*, int, int, float);
template void qk_gemm_bf16s_avx512bf16_kernel_t<128>(float*, const float*, const unsigned short*, int, int, float);
template void qk_gemm_bf16s_avx512bf16_kernel_t<256>(float*, const float*, const unsigned short*, int, int, float);
template void qk_gemm_bf16s_avx512bf16_kernel_t<512>(float*, const float*, const unsigned short*, int, int, float);
template void qk_gemm_bf16s_avx512bf16_kernel_t<1024>(float*, const float*, const unsigned short*, int, int, float);
template void qk_gemm_bf16s_avx512bf16_kernel_t<4096>(float*, const float*, const unsigned short*, int, int, float);

template void pv_gemm_bf16s_avx512bf16_kernel_t<64>(float*, const float*, const unsigned short*, int, int);
template void pv_gemm_bf16s_avx512bf16_kernel_t<128>(float*, const float*, const unsigned short*, int, int);
template void pv_gemm_bf16s_avx512bf16_kernel_t<256>(float*, const float*, const unsigned short*, int, int);
template void pv_gemm_bf16s_avx512bf16_kernel_t<512>(float*, const float*, const unsigned short*, int, int);
template void pv_gemm_bf16s_avx512bf16_kernel_t<1024>(float*, const float*, const unsigned short*, int, int);
template void pv_gemm_bf16s_avx512bf16_kernel_t<4096>(float*, const float*, const unsigned short*, int, int);

void qk_gemm_bf16s_avx512bf16(float* S, const float* Q, const unsigned short* K, int m, int n, int d, float scale)
{
    switch (d)
    {
    case 64:
        qk_gemm_bf16s_avx512bf16_kernel_t<64>(S, Q, K, m, n, scale);
        break;
    case 128:
        qk_gemm_bf16s_avx512bf16_kernel_t<128>(S, Q, K, m, n, scale);
        break;
    case 256:
        qk_gemm_bf16s_avx512bf16_kernel_t<256>(S, Q, K, m, n, scale);
        break;
    case 512:
        qk_gemm_bf16s_avx512bf16_kernel_t<512>(S, Q, K, m, n, scale);
        break;
    case 1024:
        qk_gemm_bf16s_avx512bf16_kernel_t<1024>(S, Q, K, m, n, scale);
        break;
    case 4096:
        qk_gemm_bf16s_avx512bf16_kernel_t<4096>(S, Q, K, m, n, scale);
        break;
    default:
        qk_gemm_bf16s_avx512bf16_kernel(S, Q, K, m, n, d, scale);
        break;
    }
}

void pv_gemm_bf16s_avx512bf16(float* O, const float* P, const unsigned short* V, int m, int n, int d)
{
    switch (d)
    {
    case 64:
        pv_gemm_bf16s_avx512bf16_kernel_t<64>(O, P, V, m, n);
        break;
    case 128:
        pv_gemm_bf16s_avx512bf16_kernel_t<128>(O, P, V, m, n);
        break;
    case 256:
        pv_gemm_bf16s_avx512bf16_kernel_t<256>(O, P, V, m, n);
        break;
    case 512:
        pv_gemm_bf16s_avx512bf16_kernel_t<512>(O, P, V, m, n);
        break;
    case 1024:
        pv_gemm_bf16s_avx512bf16_kernel_t<1024>(O, P, V, m, n);
        break;
    case 4096:
        pv_gemm_bf16s_avx512bf16_kernel_t<4096>(O, P, V, m, n);
        break;
    default:
        pv_gemm_bf16s_avx512bf16_kernel(O, P, V, m, n, d);
        break;
    }
}

#endif // __AVX512BF16__

} // namespace ncnn
