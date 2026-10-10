// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_torq_tile_8x3vl.h"

// Unpacked, non-transposed, dynamic FP32 inputs. The existing forward path
// scales C by beta before this call and scales the result by alpha afterwards.
static int gemm_torq_tile(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int nT)
{
    (void)nT; // Also build when OpenMP pragmas are disabled.

    const int M = A.dims == 3 ? A.c : A.h;
    const int N = B.w;
    const int K = A.w;
    const size_t lda = A.dims == 3 ? A.cstep : (size_t)A.w;
    const size_t ldb = B.dims == 3 ? B.cstep : (size_t)B.w;
    const size_t ldd = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    const int tile_n = (int)vlmax * 3;
    const int nn_M = (M + 7) / 8;

    const float* a = A;
    const float* b = B;
    const float* c = C;
    float* d = top_blob;

    #pragma omp parallel for num_threads(nT)
    for (int mi = 0; mi < nn_M; mi++)
    {
        const int i = mi * 8;
        const int rows = std::min(8, M - i);

        // Seed D with the same broadcast C values used by the packed kernel.
        for (int r = 0; r < rows; r++)
        {
            float* outptr = d + (i + r) * ldd;
            const float* cp = c;
            float scalar = 0.f;
            if (c)
            {
                if (broadcast_type_C == 0)
                    scalar = c[0];
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    scalar = c[i + r];
                if (broadcast_type_C == 3)
                    cp = c + (i + r) * (size_t)N;
            }

            for (int j = 0; j < N;)
            {
                const size_t vl = __riscv_vsetvl_e32m1(N - j);
                vfloat32m1_t value;
                if (c && (broadcast_type_C == 3 || broadcast_type_C == 4))
                    value = __riscv_vle32_v_f32m1(cp + j, vl);
                else
                    value = __riscv_vfmv_v_f_f32m1(scalar, vl);
                __riscv_vse32_v_f32m1(outptr + j, value, vl);
                j += vl;
            }
        }

        int j = 0;
        if (rows == 8)
        {
            for (; j + tile_n <= N; j += tile_n)
            {
                gemm_torq_tile_8x3vl(a + i * lda, b + j, d + i * ldd + j, lda, ldb, ldd, K, vlmax);
            }
        }

        // Predicated RVV tails never read beyond the logical matrix width.
        for (int r = 0; r < rows; r++)
        {
            const float* ap = a + (i + r) * lda;
            float* outptr = d + (i + r) * ldd;
            for (int jj = j; jj < N;)
            {
                const size_t vl = __riscv_vsetvl_e32m1(N - jj);
                vfloat32m1_t sum = __riscv_vle32_v_f32m1(outptr + jj, vl);
                const float* bp = b + jj;
                for (int kk = 0; kk < K; kk++)
                {
                    vfloat32m1_t bv = __riscv_vle32_v_f32m1(bp, vl);
                    sum = __riscv_vfmacc_vf_f32m1(sum, ap[kk], bv, vl);
                    bp += ldb;
                }
                __riscv_vse32_v_f32m1(outptr + jj, sum, vl);
                jj += vl;
            }
        }
    }

    return 0;
}
