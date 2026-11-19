// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_vulkan.h"

#include "layer_shader_type.h"

#include <string.h>

namespace ncnn {

// Per logical row, the power-of-two exponent that makes its bf16 values exact normal fp16 values.
// 256 marks a row that does not fit. Returns 1 when any row does not fit.
static int compute_gemm_bf16_scales(const Mat& data_bf16, Mat& scales, int rows, int K, int trans, const Option& opt)
{
    scales.create(rows, (size_t)4u, 1);
    if (scales.empty())
        return -100;

    const unsigned short* ptr = data_bf16;
    int* scale_ptr = scales;

    // blocks of rows keep the strided layout cache friendly
    const int nn_rows = (rows + 63) / 64;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < nn_rows; ii++)
    {
        const int i0 = ii * 64;
        const int n = std::min(64, rows - i0);

        // biased exponent range of the nonzero values, emin = 0 marks subnormal or nonfinite inputs
        int emin[64];
        int emax[64];
        for (int i = 0; i < n; i++)
        {
            emin[i] = 255;
            emax[i] = 0;
        }

        if (trans)
        {
            for (int k = 0; k < K; k++)
            {
                const unsigned short* p = ptr + (size_t)k * rows + i0;
                for (int i = 0; i < n; i++)
                {
                    const int v = p[i] & 0x7fff;
                    const int e = v >> 7;
                    const int eb = e == 255 ? 0 : e;
                    emin[i] = std::min(emin[i], v ? eb : 255);
                    emax[i] = std::max(emax[i], v ? eb : 0);
                }
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                const unsigned short* p = ptr + (size_t)(i0 + i) * K;
                int e0 = 255;
                int e1 = 0;
                for (int k = 0; k < K; k++)
                {
                    const int v = p[k] & 0x7fff;
                    const int e = v >> 7;
                    const int eb = e == 255 ? 0 : e;
                    e0 = std::min(e0, v ? eb : 255);
                    e1 = std::max(e1, v ? eb : 0);
                }
                emin[i] = e0;
                emax[i] = e1;
            }
        }

        for (int i = 0; i < n; i++)
        {
            // Keep scaled values normal in fp16 and power-of-two factors normal in fp32.
            const int scale = emax[i] == 0 ? 0 : std::max(emax[i] - 127 - 15, -112);
            const bool fits = emin[i] != 0 && (emax[i] == 0 || emin[i] - 127 - scale >= -14);
            scale_ptr[i0 + i] = fits ? scale : 256;
        }
    }

    for (int i = 0; i < rows; i++)
    {
        if (scale_ptr[i] == 256)
            return 1;
    }

    return 0;
}

// Write the scaled fp16 bits of rows that fit into the tile layout loaded by gemm_bf16s_cm,
// the same layout as the fp32 tiles packed for gemm_cm and as packed_offset() in the shaders.
static void pack_gemm_bf16_tiles(const Mat& data_bf16, const Mat& scales, Mat& packed, int rows, int K, int trans, int R, int coopmat_K, int UNROLL_SG_R, int UNROLL_SG_K, int UNROLL_WG_R, int kmajor, const Option& opt)
{
    const int block_r = R * UNROLL_SG_R * UNROLL_WG_R;
    const int blocks = (rows + block_r - 1) / block_r;
    const int kk = (K + coopmat_K - 1) / coopmat_K;
    const int kfull = kk / UNROLL_SG_K * UNROLL_SG_K;

    packed.create(block_r * coopmat_K * kk, blocks, (size_t)2u);
    if (packed.empty())
        return;

    const unsigned short* ptr = data_bf16;
    const int* scale_ptr = scales;

    // each group of R rows owns whole tile rows, so every element including the zero padding is written once
    const int nn_groups = blocks * block_r / R;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int rg = 0; rg < nn_groups; rg++)
    {
        const int r0 = rg * R;
        const int rr = r0 % block_r;
        const int wr = rr / (R * UNROLL_SG_R);
        const int zr = rr / R % UNROLL_SG_R;

        unsigned short* outptr = packed.row<unsigned short>(r0 / block_r);

        for (int kb = 0; kb < kk; kb++)
        {
            // full UNROLL_SG_K groups first, then single k steps
            const int zk = kb % UNROLL_SG_K;
            const int g = kb < kfull ? (kb - zk) * UNROLL_WG_R + wr * UNROLL_SG_K + zk : kb * UNROLL_WG_R + wr;
            unsigned short* p = outptr + ((size_t)g * UNROLL_SG_R + zr) * R * coopmat_K;

            for (int ri = 0; ri < R; ri++)
            {
                const int r = r0 + ri;
                const int s = r < rows ? scale_ptr[r] : 0;

                for (int kj = 0; kj < coopmat_K; kj++)
                {
                    const int k = kb * coopmat_K + kj;
                    const unsigned short v = r < rows && k < K ? ptr[trans ? (size_t)k * rows + r : (size_t)r * K + k] : 0;

                    // exact for rows that fit, the exponent lands in [1, 30]
                    unsigned short h = v & 0x8000;
                    if (v & 0x7fff)
                        h |= (unsigned short)((((v >> 7) & 0xff) - 127 - s + 15) << 10) | ((v & 0x7f) << 3);

                    p[kmajor ? kj * R + ri : ri * coopmat_K + kj] = h;
                }
            }
        }
    }
}

#if NCNN_INT8
static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}
#endif // NCNN_INT8

Gemm_vulkan::Gemm_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;

    pipeline_gemm = 0;
    pipeline_gemm_pack4 = 0;
    pipeline_gemm_bf16_pack_A = 0;
    pipeline_gemm_bf16_pack_B = 0;
#if NCNN_INT8
    pipeline_gemm_quantize_A_int8 = 0;
    pipeline_gemm_quantize_B_absmax_int8 = 0;
    pipeline_gemm_quantize_B_descale_int8 = 0;
    pipeline_gemm_quantize_B_int8 = 0;
#endif

    use_subgroup_ops = false;

    use_cooperative_matrix = false;
    use_bf16_cooperative_matrix_emulation = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;
}

int Gemm_vulkan::create_pipeline(const Option& opt)
{
    if (weight_block_quantize)
    {
        NCNN_LOGE("Gemm weight block quantization is not supported by Vulkan");
        return -1;
    }

#if NCNN_INT8
    if (quantize_term)
    {
        return create_pipeline_int8(opt);
    }
#endif

    // const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    if (constantA)
    {
        A_data_packed = transA ? A_data.reshape(constantM, constantK) : A_data.reshape(constantK, constantM);
    }
    if (constantB)
    {
        B_data_packed = transB ? B_data.reshape(constantK, constantN) : B_data.reshape(constantN, constantK);
    }
    if (constantC)
    {
        C_data_packed = C_data;
    }

    use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

    bool use_bf16_cooperative_matrix = false;
    if (vkdev->info.support_bf16_cooperative_matrix() && opt.use_cooperative_matrix && opt.use_bf16_storage)
    {
        use_cooperative_matrix = true;
        use_bf16_cooperative_matrix = true;
    }

    use_bf16_cooperative_matrix_emulation = !use_bf16_cooperative_matrix && vkdev->info.support_cooperative_matrix()
                                            && opt.use_cooperative_matrix && (opt.use_bf16_storage || opt.use_bf16_packed);
    if (use_bf16_cooperative_matrix_emulation)
        use_cooperative_matrix = true;

    if (vkdev->info.vendor_id() == 0x5143)
    {
        // adreno produces incorrect results even for complete cooperative matrix tiles
        use_cooperative_matrix = false;
    }

    const int subgroup_size = vkdev->info.subgroup_size();
    const uint32_t support_subgroup_ops = vkdev->info.support_subgroup_ops();
    const uint32_t required_subgroup_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    use_subgroup_ops = opt.use_subgroup_ops && ((support_subgroup_ops & required_subgroup_ops) == required_subgroup_ops);
    if (subgroup_size < 4 || subgroup_size > 128)
    {
        // sanitize wired subgroup_size
        use_subgroup_ops = false;
    }
    if (opt.use_fp16_arithmetic && !opt.use_bf16_storage && !opt.use_bf16_packed && !vkdev->info.queryShaderSubgroupExtendedTypesFeatures().shaderSubgroupExtendedTypes)
    {
        // gemm_sg shuffles fp16 vectors, which requires subgroup extended types
        use_subgroup_ops = false;
    }

    if (vkdev->info.vendor_id() == 0x5143 && !vkdev->info.queryMaintenance4Features().maintenance4)
    {
        // adreno subgroup shuffle workaround requires spirv-1.6 LocalSizeId
        use_subgroup_ops = false;
    }

    const int M = constantM ? constantM : 1024;
    const int N = constantN ? constantN : 1024;
    const int K = constantK ? constantK : 1024;

    Option opt_cm = opt;

    if (use_cooperative_matrix)
    {
        VkComponentTypeKHR coopmat_acctype;
        if (use_bf16_cooperative_matrix)
        {
            vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size, coopmat_acctype);
        }
        else
        {
            const VkComponentTypeKHR acctype = opt.use_fp16_arithmetic && !use_bf16_cooperative_matrix_emulation ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR;
            vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_FLOAT16_KHR, acctype, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size, coopmat_acctype);
        }

        // the selected tile may use fp32 accumulation even when fp16 arithmetic was requested
        opt_cm.use_fp16_arithmetic = coopmat_acctype == VK_COMPONENT_TYPE_FLOAT16_KHR;

        const bool use_khr_cooperative_matrix = vkdev->info.support_VK_KHR_cooperative_matrix();
        if (coopmat_M == 0 || (use_khr_cooperative_matrix && ((coopmat_M != 4 && coopmat_M % 8) || coopmat_N % 8 || coopmat_K % 8)))
        {
            // allow four-row tiles while retaining eight-element N/K tiles
            use_cooperative_matrix = false;
        }
        else
        {
            UNROLL_SG_M = std::min((M + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((N + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((K + coopmat_K - 1) / coopmat_K, 2);

            if (vkdev->info.vendor_id() == 0x5143)
            {
                // adreno gemm register allocation fails with subgroup tile unrolling
                UNROLL_SG_M = 1;
                UNROLL_SG_N = 1;
            }

            UNROLL_WG_M = std::min((M + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((N + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

            // match the shared-memory row strides in gemm_cm
            const int Md4 = coopmat_M / 4;
            const int Nd4 = coopmat_N / 4;
            const int Kd4 = coopmat_K / 4;
            const size_t shared_a = 8 * std::max(coopmat_M * Kd4, coopmat_K * Md4);
            const size_t shared_b = 8 * std::max(coopmat_K * Nd4, coopmat_N * Kd4);
            const size_t shared_o = (use_bf16_cooperative_matrix_emulation ? 4 : 2) * coopmat_M * coopmat_N;
            // the bf16 emulation also stages one int scale per tile row and column, and a fallback list of two outputs per invocation
            const size_t shared_s = use_bf16_cooperative_matrix_emulation ? 4 : 0;

            for (;;)
            {
                const size_t shared_bytes = shared_a * UNROLL_WG_M * UNROLL_SG_M * UNROLL_SG_K
                                            + shared_b * UNROLL_WG_N * UNROLL_SG_N * UNROLL_SG_K
                                            + shared_o * UNROLL_WG_M * UNROLL_WG_N * UNROLL_SG_M * UNROLL_SG_N
                                            + shared_s * (coopmat_M * UNROLL_SG_M * UNROLL_WG_M + coopmat_N * UNROLL_SG_N * UNROLL_WG_N + coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N * 2 + 1);
                const uint32_t invocations = coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N;
                if (shared_bytes <= vkdev->info.max_shared_memory_size() && invocations <= vkdev->info.max_workgroup_invocations()
                        && invocations <= vkdev->info.max_workgroup_size_x() && (uint32_t)(UNROLL_WG_M * UNROLL_WG_N) <= vkdev->info.max_compute_workgroup_subgroups())
                    break;

                // reduce K first to preserve output reuse
                if (UNROLL_SG_K > 1)
                    UNROLL_SG_K = 1;
                else if (UNROLL_WG_N > 1)
                    UNROLL_WG_N = 1;
                else if (UNROLL_WG_M > 1)
                    UNROLL_WG_M = 1;
                else if (UNROLL_SG_N > 1)
                    UNROLL_SG_N = 1;
                else if (UNROLL_SG_M > 1)
                    UNROLL_SG_M = 1;
                else
                {
                    use_cooperative_matrix = false;
                    break;
                }
            }
        }
    }

    if (!use_cooperative_matrix)
        use_bf16_cooperative_matrix_emulation = false;

    Mat A_data_bf16;
    Mat B_data_bf16;
    if (use_bf16_cooperative_matrix_emulation)
    {
        // Constant operands are converted once into exact fp16 tiles.
        // A weight row that does not fit keeps the whole layer on the fp32 shader.
        int unsafe = 0;
        if (constantA)
        {
            // Match the bf16 conversion performed by record_upload().
            cast_float32_to_bfloat16(A_data, A_data_bf16, opt);
            int ret = compute_gemm_bf16_scales(A_data_bf16, A_data_bf16_scales, constantM, constantK, transA, opt);
            if (ret < 0)
                return ret;
            unsafe |= ret;
        }
        if (constantB)
        {
            cast_float32_to_bfloat16(B_data, B_data_bf16, opt);
            int ret = compute_gemm_bf16_scales(B_data_bf16, B_data_bf16_scales, constantN, constantK, !transB, opt);
            if (ret < 0)
                return ret;
            unsafe |= ret;
        }

        if (unsafe)
        {
            use_bf16_cooperative_matrix_emulation = false;
            use_cooperative_matrix = false;
            A_data_bf16_scales.release();
            B_data_bf16_scales.release();
        }
    }

    if (use_cooperative_matrix)
    {
        if (use_bf16_cooperative_matrix_emulation)
        {
            if (constantA == 1)
            {
                pack_gemm_bf16_tiles(A_data_bf16, A_data_bf16_scales, A_data_packed, M, K, transA, coopmat_M, coopmat_K, UNROLL_SG_M, UNROLL_SG_K, UNROLL_WG_M, 0, opt);
                if (A_data_packed.empty())
                    return -100;
            }
            if (constantB == 1)
            {
                pack_gemm_bf16_tiles(B_data_bf16, B_data_bf16_scales, B_data_packed, N, K, !transB, coopmat_N, coopmat_K, UNROLL_SG_N, UNROLL_SG_K, UNROLL_WG_N, 1, opt);
                if (B_data_packed.empty())
                    return -100;
            }
        }
        else if (constantA == 1)
        {
            //        +-K-+
            //        M   |
            //        +- -+
            //      SG_UM |
            //     ^  +---+
            //     |  |   |
            //   SG_UK+- -+
            //     |  |   |
            //   ^ v  +---+
            //   |    |   |
            //   |    +- -+
            //   |    |   |
            // WG_UM  +---+
            //   |    |   |
            //   |    +- -+
            //   |    |   |
            //   v    +---+

            //      +-K-+
            //      M   |
            //      +SG_UM
            //      |   |
            //   ^  +---+
            //   |  |   |
            // WG_UM+- -+
            //   |  |   |
            //   v  +---+

            const int blocks_m = (M + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int kk = (K + coopmat_K - 1) / coopmat_K;

            A_data_packed.create(coopmat_M * coopmat_K * UNROLL_SG_M * UNROLL_WG_M * kk, blocks_m);

            if (transA == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bm = 0; bm < blocks_m; bm++)
                {
                    float* p = A_data_packed.row(bm);

                    int k = 0;
                    for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                    {
                        for (int wm = 0; wm < UNROLL_WG_M; wm++)
                        {
                            for (int zk = 0; zk < UNROLL_SG_K; zk++)
                            {
                                for (int zm = 0; zm < UNROLL_SG_M; zm++)
                                {
                                    for (int i = 0; i < coopmat_M; i++)
                                    {
                                        for (int j = 0; j < coopmat_K; j++)
                                        {
                                            const int gmi = ((bm * UNROLL_WG_M + wm) * UNROLL_SG_M + zm) * coopmat_M + i;
                                            const int gki = (k + zk) * coopmat_K + j;

                                            if (gmi < M && gki < K)
                                            {
                                                *p++ = A_data[gmi * K + gki];
                                            }
                                            else
                                            {
                                                *p++ = 0.f;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for (; k < kk; k++)
                    {
                        for (int wm = 0; wm < UNROLL_WG_M; wm++)
                        {
                            for (int zm = 0; zm < UNROLL_SG_M; zm++)
                            {
                                for (int i = 0; i < coopmat_M; i++)
                                {
                                    for (int j = 0; j < coopmat_K; j++)
                                    {
                                        const int gmi = ((bm * UNROLL_WG_M + wm) * UNROLL_SG_M + zm) * coopmat_M + i;
                                        const int gki = k * coopmat_K + j;

                                        if (gmi < M && gki < K)
                                        {
                                            *p++ = A_data[gmi * K + gki];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bm = 0; bm < blocks_m; bm++)
                {
                    float* p = A_data_packed.row(bm);

                    int k = 0;
                    for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                    {
                        for (int wm = 0; wm < UNROLL_WG_M; wm++)
                        {
                            for (int zk = 0; zk < UNROLL_SG_K; zk++)
                            {
                                for (int zm = 0; zm < UNROLL_SG_M; zm++)
                                {
                                    for (int i = 0; i < coopmat_M; i++)
                                    {
                                        for (int j = 0; j < coopmat_K; j++)
                                        {
                                            const int gmi = ((bm * UNROLL_WG_M + wm) * UNROLL_SG_M + zm) * coopmat_M + i;
                                            const int gki = (k + zk) * coopmat_K + j;

                                            if (gmi < M && gki < K)
                                            {
                                                *p++ = A_data[gki * M + gmi];
                                            }
                                            else
                                            {
                                                *p++ = 0.f;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for (; k < kk; k++)
                    {
                        for (int wm = 0; wm < UNROLL_WG_M; wm++)
                        {
                            for (int zm = 0; zm < UNROLL_SG_M; zm++)
                            {
                                for (int i = 0; i < coopmat_M; i++)
                                {
                                    for (int j = 0; j < coopmat_K; j++)
                                    {
                                        const int gmi = ((bm * UNROLL_WG_M + wm) * UNROLL_SG_M + zm) * coopmat_M + i;
                                        const int gki = k * coopmat_K + j;

                                        if (gmi < M && gki < K)
                                        {
                                            *p++ = A_data[gki * M + gmi];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (constantB == 1 && !use_bf16_cooperative_matrix_emulation)
        {
            //        +-N-+
            //        K   |
            //        +SG_UN
            //        |   |
            //     ^  +---+
            //     |  |   |
            //   SG_UK+- -+
            //     |  |   |
            //   ^ v  +---+
            //   |    |   |
            //   |    +- -+
            //   |    |   |
            // WG_UN  +---+
            //   |    |   |
            //   |    +- -+
            //   |    |   |
            //   v    +---+

            //      +-N-+
            //      K   |
            //      +SG_UN
            //      |   |
            //   ^  +---+
            //   |  |   |
            // WG_UN+- -+
            //   |  |   |
            //   v  +---+

            const int blocks_n = (N + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            const int kk = (K + coopmat_K - 1) / coopmat_K;

            B_data_packed.create(coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk, blocks_n);

            if (transB == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bn = 0; bn < blocks_n; bn++)
                {
                    float* p = B_data_packed.row(bn);

                    int k = 0;
                    for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                    {
                        for (int wn = 0; wn < UNROLL_WG_N; wn++)
                        {
                            for (int zk = 0; zk < UNROLL_SG_K; zk++)
                            {
                                for (int zn = 0; zn < UNROLL_SG_N; zn++)
                                {
                                    for (int i = 0; i < coopmat_K; i++)
                                    {
                                        for (int j = 0; j < coopmat_N; j++)
                                        {
                                            const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                            const int gki = (k + zk) * coopmat_K + i;

                                            if (gni < N && gki < K)
                                            {
                                                *p++ = B_data[gni * K + gki];
                                            }
                                            else
                                            {
                                                *p++ = 0.f;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for (; k < kk; k++)
                    {
                        for (int wn = 0; wn < UNROLL_WG_N; wn++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        const int gki = k * coopmat_K + i;

                                        if (gni < N && gki < K)
                                        {
                                            *p++ = B_data[gni * K + gki];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bn = 0; bn < blocks_n; bn++)
                {
                    float* p = B_data_packed.row(bn);

                    int k = 0;
                    for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                    {
                        for (int wn = 0; wn < UNROLL_WG_N; wn++)
                        {
                            for (int zk = 0; zk < UNROLL_SG_K; zk++)
                            {
                                for (int zn = 0; zn < UNROLL_SG_N; zn++)
                                {
                                    for (int i = 0; i < coopmat_K; i++)
                                    {
                                        for (int j = 0; j < coopmat_N; j++)
                                        {
                                            const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                            const int gki = (k + zk) * coopmat_K + i;

                                            if (gni < N && gki < K)
                                            {
                                                *p++ = B_data[gki * N + gni];
                                            }
                                            else
                                            {
                                                *p++ = 0.f;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    for (; k < kk; k++)
                    {
                        for (int wn = 0; wn < UNROLL_WG_N; wn++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        const int gki = k * coopmat_K + i;

                                        if (gni < N && gki < K)
                                        {
                                            *p++ = B_data[gki * N + gni];
                                        }
                                        else
                                        {
                                            *p++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        int outh = output_transpose ? constantN : constantM;
        int out_elempack = outh ? (outh % 4 == 0 ? 4 : 1) : 0;

        if (use_bf16_cooperative_matrix_emulation)
        {
            // dynamic operands are scaled and packed into the same fp16 tiles before the matrix kernel
            if (!constantA)
            {
                std::vector<vk_specialization_type> specializations(7);
                specializations[0].i = transA;
                specializations[1].i = 0;
                specializations[2].u32 = coopmat_M;
                specializations[3].u32 = coopmat_K;
                specializations[4].u32 = UNROLL_SG_M;
                specializations[5].u32 = UNROLL_SG_K;
                specializations[6].u32 = UNROLL_WG_M;

                pipeline_gemm_bf16_pack_A = new Pipeline(vkdev);
                pipeline_gemm_bf16_pack_A->set_local_size_xyz(256, 1, 1);
                pipeline_gemm_bf16_pack_A->create(LayerShaderType::gemm_bf16s_cm_pack, opt_cm, specializations);
            }

            if (!constantB)
            {
                std::vector<vk_specialization_type> specializations(7);
                specializations[0].i = !transB;
                specializations[1].i = 1;
                specializations[2].u32 = coopmat_N;
                specializations[3].u32 = coopmat_K;
                specializations[4].u32 = UNROLL_SG_N;
                specializations[5].u32 = UNROLL_SG_K;
                specializations[6].u32 = UNROLL_WG_N;

                pipeline_gemm_bf16_pack_B = new Pipeline(vkdev);
                pipeline_gemm_bf16_pack_B->set_local_size_xyz(256, 1, 1);
                pipeline_gemm_bf16_pack_B->create(LayerShaderType::gemm_bf16s_cm_pack, opt_cm, specializations);
            }

            std::vector<vk_specialization_type> specializations(11 + 9);
            specializations[0].f = alpha;
            specializations[1].f = beta;
            specializations[2].i = transA;
            specializations[3].i = transB;
            specializations[4].i = constantA;
            specializations[5].i = constantB;
            specializations[6].i = output_transpose;
            specializations[7].i = constantM;
            specializations[8].i = constantN;
            specializations[9].i = constantK;
            specializations[10].i = output_elempack ? output_elempack : out_elempack;

            specializations[11 + 0].u32 = coopmat_M;
            specializations[11 + 1].u32 = coopmat_N;
            specializations[11 + 2].u32 = coopmat_K;
            specializations[11 + 3].u32 = coopmat_subgroup_size;
            specializations[11 + 4].u32 = UNROLL_SG_M;
            specializations[11 + 5].u32 = UNROLL_SG_N;
            specializations[11 + 6].u32 = UNROLL_SG_K;
            specializations[11 + 7].u32 = UNROLL_WG_M;
            specializations[11 + 8].u32 = UNROLL_WG_N;

            pipeline_gemm = new Pipeline(vkdev);
            pipeline_gemm->set_subgroup_size(coopmat_subgroup_size);
            pipeline_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_gemm->create(LayerShaderType::gemm_bf16s_cm, opt_cm, specializations);
        }
        else
        {
            std::vector<vk_specialization_type> specializations(18 + 9);
            specializations[0].f = alpha;
            specializations[1].f = beta;
            specializations[2].i = transA;
            specializations[3].i = transB;
            specializations[4].i = constantA;
            specializations[5].i = constantB;
            specializations[6].i = constantC;
            specializations[7].i = constantM;
            specializations[8].i = constantN;
            specializations[9].i = constantK;
            specializations[10].i = constant_broadcast_type_C;
            specializations[11].i = output_N1M;
            specializations[12].i = output_elempack;
            specializations[13].i = output_elemtype;
            specializations[14].i = output_transpose;
            specializations[15].i = A_data_packed.elempack;
            specializations[16].i = B_data_packed.elempack;
            specializations[17].i = output_elempack ? output_elempack : out_elempack;

            specializations[18 + 0].u32 = coopmat_M;
            specializations[18 + 1].u32 = coopmat_N;
            specializations[18 + 2].u32 = coopmat_K;
            specializations[18 + 3].u32 = coopmat_subgroup_size;
            specializations[18 + 4].u32 = UNROLL_SG_M;
            specializations[18 + 5].u32 = UNROLL_SG_N;
            specializations[18 + 6].u32 = UNROLL_SG_K;
            specializations[18 + 7].u32 = UNROLL_WG_M;
            specializations[18 + 8].u32 = UNROLL_WG_N;

            // specialize dynamic output packing to avoid adreno store layout branches
            const bool use_output_pack_variants = vkdev->info.vendor_id() == 0x5143 && specializations[17].i == 0;
            if (use_output_pack_variants)
                specializations[17].i = 1;

            pipeline_gemm = new Pipeline(vkdev);
            pipeline_gemm->set_subgroup_size(coopmat_subgroup_size);
            pipeline_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_gemm->create(LayerShaderType::gemm_cm, opt_cm, specializations);

            if (use_output_pack_variants)
            {
                specializations[17].i = 4;

                pipeline_gemm_pack4 = new Pipeline(vkdev);
                pipeline_gemm_pack4->set_subgroup_size(coopmat_subgroup_size);
                pipeline_gemm_pack4->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_gemm_pack4->create(LayerShaderType::gemm_cm, opt_cm, specializations);
            }
        }
    }
    else if (opt.use_shader_local_memory)
    {
        std::vector<vk_specialization_type> specializations(15);
        specializations[0].f = alpha;
        specializations[1].f = beta;
        specializations[2].i = transA;
        specializations[3].i = transB;
        specializations[4].i = constantA;
        specializations[5].i = constantB;
        specializations[6].i = constantC;
        specializations[7].i = constantM;
        specializations[8].i = constantN;
        specializations[9].i = constantK;
        specializations[10].i = constant_broadcast_type_C;
        specializations[11].i = output_N1M;
        specializations[12].i = output_elempack;
        specializations[13].i = output_elemtype;
        specializations[14].i = output_transpose;

        pipeline_gemm = new Pipeline(vkdev);
        pipeline_gemm->set_local_size_xyz(8, 8, 1);
        pipeline_gemm->create(LayerShaderType::gemm, opt, specializations);
    }
    else if (use_subgroup_ops)
    {
        if (subgroup_size == 128)
        {
            UNROLL_SG_M = 16;
            UNROLL_SG_N = 8;
            UNROLL_SG_K = 8;
        }
        if (subgroup_size == 64)
        {
            UNROLL_SG_M = 8;
            UNROLL_SG_N = 8;
            UNROLL_SG_K = 8;
        }
        if (subgroup_size == 32)
        {
            UNROLL_SG_M = 8;
            UNROLL_SG_N = 4;
            UNROLL_SG_K = 4;
        }
        if (subgroup_size == 16)
        {
            UNROLL_SG_M = 4;
            UNROLL_SG_N = 4;
            UNROLL_SG_K = 4;
        }
        if (subgroup_size == 8)
        {
            UNROLL_SG_M = 4;
            UNROLL_SG_N = 2;
            UNROLL_SG_K = 2;
        }
        if (subgroup_size == 4)
        {
            UNROLL_SG_M = 2;
            UNROLL_SG_N = 2;
            UNROLL_SG_K = 2;
        }

        std::vector<vk_specialization_type> specializations(18);
        specializations[0].f = alpha;
        specializations[1].f = beta;
        specializations[2].i = transA;
        specializations[3].i = transB;
        specializations[4].i = constantA;
        specializations[5].i = constantB;
        specializations[6].i = constantC;
        specializations[7].u32 = constantM;
        specializations[8].u32 = constantN;
        specializations[9].u32 = constantK;
        specializations[10].i = constant_broadcast_type_C;
        specializations[11].i = output_N1M;
        specializations[12].i = output_elempack;
        specializations[13].i = output_elemtype;
        specializations[14].i = output_transpose;
        specializations[15].u32 = UNROLL_SG_M;
        specializations[16].u32 = UNROLL_SG_N;
        specializations[17].u32 = UNROLL_SG_K;

        pipeline_gemm = new Pipeline(vkdev);
        pipeline_gemm->set_subgroup_size(subgroup_size);
        pipeline_gemm->set_local_size_xyz(subgroup_size, 1, 1);
        pipeline_gemm->create(LayerShaderType::gemm_sg, opt, specializations);
    }
    else
    {
        std::vector<vk_specialization_type> specializations(15);
        specializations[0].f = alpha;
        specializations[1].f = beta;
        specializations[2].i = transA;
        specializations[3].i = transB;
        specializations[4].i = constantA;
        specializations[5].i = constantB;
        specializations[6].i = constantC;
        specializations[7].i = constantM;
        specializations[8].i = constantN;
        specializations[9].i = constantK;
        specializations[10].i = constant_broadcast_type_C;
        specializations[11].i = output_N1M;
        specializations[12].i = output_elempack;
        specializations[13].i = output_elemtype;
        specializations[14].i = output_transpose;

        Mat local_size_xyz;

        pipeline_gemm = new Pipeline(vkdev);
        pipeline_gemm->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gemm->create(LayerShaderType::gemm, opt, specializations);
    }

    if (opt.lightmode)
    {
        A_data.release();
        B_data.release();
        C_data.release();
    }

    return 0;
}

int Gemm_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_gemm;
    pipeline_gemm = 0;

    delete pipeline_gemm_pack4;
    pipeline_gemm_pack4 = 0;

    delete pipeline_gemm_bf16_pack_A;
    pipeline_gemm_bf16_pack_A = 0;

    delete pipeline_gemm_bf16_pack_B;
    pipeline_gemm_bf16_pack_B = 0;

#if NCNN_INT8
    delete pipeline_gemm_quantize_A_int8;
    pipeline_gemm_quantize_A_int8 = 0;

    delete pipeline_gemm_quantize_B_absmax_int8;
    pipeline_gemm_quantize_B_absmax_int8 = 0;

    delete pipeline_gemm_quantize_B_descale_int8;
    pipeline_gemm_quantize_B_descale_int8 = 0;

    delete pipeline_gemm_quantize_B_int8;
    pipeline_gemm_quantize_B_int8 = 0;
#endif

    use_subgroup_ops = false;

    use_cooperative_matrix = false;
    use_bf16_cooperative_matrix_emulation = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;

    return 0;
}

int Gemm_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (weight_block_quantize)
    {
        NCNN_LOGE("Gemm weight block quantization is not supported by Vulkan");
        return -1;
    }

#if NCNN_INT8
    if (quantize_term)
    {
        return upload_model_int8(cmd, opt);
    }
#endif

    // the bf16 emulation uploads int32 scales, its packed fp16 tiles are uploaded as they are
    if (use_bf16_cooperative_matrix_emulation)
    {
        Option opt_scale = opt;
        opt_scale.use_fp16_packed = false;
        opt_scale.use_fp16_storage = false;
        opt_scale.use_bf16_packed = false;
        opt_scale.use_bf16_storage = false;

        if (constantA)
        {
            cmd.record_upload(A_data_bf16_scales, A_data_bf16_scales_gpu, opt_scale);
            A_data_bf16_scales.release();
        }
        if (constantB)
        {
            cmd.record_upload(B_data_bf16_scales, B_data_bf16_scales_gpu, opt_scale);
            B_data_bf16_scales.release();
        }
    }

    if (constantA)
    {
        cmd.record_upload(A_data_packed, A_data_gpu, opt);

        A_data_packed.release();
    }

    if (constantB)
    {
        cmd.record_upload(B_data_packed, B_data_gpu, opt);

        B_data_packed.release();
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        cmd.record_upload(C_data_packed, C_data_gpu, opt);

        C_data_packed.release();
    }

    return 0;
}

int Gemm_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    if (weight_block_quantize)
    {
        NCNN_LOGE("Gemm weight block quantization is not supported by Vulkan");
        return -1;
    }

#if NCNN_INT8
    if (quantize_term)
    {
        return forward_int8(bottom_blobs, top_blobs, cmd, opt);
    }
#endif

    const VkMat& A = constantA ? A_data_gpu : bottom_blobs[0];
    const VkMat& B = constantB ? B_data_gpu : constantA ? bottom_blobs[0] : bottom_blobs[1];

    const int A_elempack = A.elempack;
    const int B_elempack = B.elempack;
    const int M = constantM ? constantM : transA ? A.w : (A.dims == 3 ? A.c * A_elempack : A.h * A_elempack);
    const int K = constantK ? constantK : transA ? (A.dims == 3 ? A.c * A_elempack : A.h * A_elempack) : A.w;
    const int N = constantN ? constantN : transB ? (B.dims == 3 ? B.c * B_elempack : B.h * B_elempack) : B.w;

    VkMat C;
    int broadcast_type_C = -1;
    if (constantC && constant_broadcast_type_C != -1)
    {
        vkdev->convert_packing(C_data_gpu, C, 1, cmd, opt);
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        VkMat C0;
        if (constantA && constantB)
        {
            C0 = bottom_blobs.size() == 1 ? bottom_blobs[0] : VkMat();
        }
        else if (constantA)
        {
            C0 = bottom_blobs.size() == 2 ? bottom_blobs[1] : VkMat();
        }
        else if (constantB)
        {
            C0 = bottom_blobs.size() == 2 ? bottom_blobs[1] : VkMat();
        }
        else
        {
            C0 = bottom_blobs.size() == 3 ? bottom_blobs[2] : VkMat();
        }

        if (!C0.empty())
        {
            vkdev->convert_packing(C0, C, 1, cmd, opt);

            if (C.dims == 1 && C.w == 1)
            {
                // scalar
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w == M)
            {
                // M
                // auto broadcast from h to w is the ncnn-style convention
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w == N)
            {
                // N
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h == M)
            {
                // Mx1
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h == M)
            {
                // MxN
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h == 1)
            {
                // 1xN
                broadcast_type_C = 4;
            }
        }
    }

    size_t elemsize = A.elemsize / A_elempack;

    int out_elempack = 1;
    {
        int outh = output_transpose ? N : M;
        out_elempack = outh % 4 == 0 ? 4 : 1;
        if (output_elempack)
            out_elempack = output_elempack;
    }

    size_t out_elemsize = elemsize * out_elempack;

    VkMat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        else
            top_blob.create(M, N / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        else
            top_blob.create(N, M / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    if (use_bf16_cooperative_matrix_emulation)
    {
        const int A_hstep = A.dims == 3 ? A.cstep : A.dims == 2 ? A.w : transA ? M : K;
        const int B_hstep = B.dims == 3 ? B.cstep : B.dims == 2 ? B.w : transB ? K : N;
        const int kk = (K + coopmat_K - 1) / coopmat_K;

        VkMat A_packed = A_data_gpu;
        VkMat A_scales = A_data_bf16_scales_gpu;
        if (!constantA)
        {
            const int block_m = coopmat_M * UNROLL_SG_M * UNROLL_WG_M;
            const int blocks_m = (M + block_m - 1) / block_m;

            A_packed.create(block_m * coopmat_K * kk * blocks_m, (size_t)2u, 1, opt.workspace_vkallocator);
            A_scales.create(M, (size_t)4u, 1, opt.workspace_vkallocator);
            if (A_packed.empty() || A_scales.empty())
                return -100;

            std::vector<VkMat> bindings(4);
            bindings[0] = A;
            bindings[1] = A;
            bindings[2] = A_packed;
            bindings[3] = A_scales;

            std::vector<vk_constant_type> constants(4);
            constants[0].i = M;
            constants[1].i = K;
            constants[2].i = A_hstep;
            constants[3].i = A_elempack;

            // one workgroup scales and packs four rows, including the zero padding
            VkMat dispatcher;
            dispatcher.w = blocks_m * block_m / 4 * 256;
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_gemm_bf16_pack_A, bindings, constants, dispatcher);
        }

        VkMat B_packed = B_data_gpu;
        VkMat B_scales = B_data_bf16_scales_gpu;
        if (!constantB)
        {
            const int block_n = coopmat_N * UNROLL_SG_N * UNROLL_WG_N;
            const int blocks_n = (N + block_n - 1) / block_n;

            B_packed.create(block_n * coopmat_K * kk * blocks_n, (size_t)2u, 1, opt.workspace_vkallocator);
            B_scales.create(N, (size_t)4u, 1, opt.workspace_vkallocator);
            if (B_packed.empty() || B_scales.empty())
                return -100;

            std::vector<VkMat> bindings(4);
            bindings[0] = B;
            bindings[1] = B;
            bindings[2] = B_packed;
            bindings[3] = B_scales;

            std::vector<vk_constant_type> constants(4);
            constants[0].i = N;
            constants[1].i = K;
            constants[2].i = B_hstep;
            constants[3].i = B_elempack;

            VkMat dispatcher;
            dispatcher.w = blocks_n * block_n / 4 * 256;
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_gemm_bf16_pack_B, bindings, constants, dispatcher);
        }

        // the original bf16 inputs serve the fp32 fallback of rows and columns that do not fit
        std::vector<VkMat> bindings(9);
        bindings[0] = top_blob;
        bindings[1] = top_blob;
        bindings[2] = A_packed;
        bindings[3] = B_packed;
        bindings[4] = C;
        bindings[5] = A_scales;
        bindings[6] = B_scales;
        bindings[7] = constantA ? VkMat() : A;
        bindings[8] = constantB ? VkMat() : B;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = M;
        constants[1].i = N;
        constants[2].i = K;
        constants[3].i = broadcast_type_C;
        constants[4].i = A_hstep;
        constants[5].i = B_hstep;
        constants[6].i = A_elempack;
        constants[7].i = B_elempack;
        constants[8].i = top_blob.dims == 3 ? top_blob.cstep : top_blob.w;
        constants[9].i = out_elempack;

        const int blocks_x = (M + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
        const int blocks_y = (N + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

        VkMat dispatcher;
        dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
        dispatcher.h = 1;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_gemm, bindings, constants, dispatcher);
    }
    else if (use_cooperative_matrix)
    {
        std::vector<VkMat> bindings(5);
        bindings[0] = top_blob;
        bindings[1] = A;
        bindings[2] = B;
        bindings[3] = C;
        bindings[4] = top_blob;

        std::vector<vk_constant_type> constants(13);
        constants[0].i = M;
        constants[1].i = N;
        constants[2].i = K;
        constants[3].i = broadcast_type_C;
        constants[4].i = A.dims;
        constants[5].i = A.dims == 3 ? A.cstep : A.dims == 2 ? A.w : transA ? M : K;
        constants[6].i = B.dims;
        constants[7].i = B.dims == 3 ? B.cstep : B.dims == 2 ? B.w : transB ? K : N;
        constants[8].i = top_blob.dims;
        constants[9].i = top_blob.dims == 3 ? top_blob.cstep : top_blob.w;
        constants[10].i = A_elempack;
        constants[11].i = B_elempack;
        constants[12].i = out_elempack;

        const int blocks_x = (M + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
        const int blocks_y = (N + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

        VkMat dispatcher;
        dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
        dispatcher.h = 1;
        dispatcher.c = 1;

        const Pipeline* pipeline = out_elempack == 4 && pipeline_gemm_pack4 ? pipeline_gemm_pack4 : pipeline_gemm;
        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
    }
    else
    {
        std::vector<VkMat> bindings(5);
        bindings[0] = top_blob;
        bindings[1] = A;
        bindings[2] = B;
        bindings[3] = C;
        bindings[4] = top_blob;

        std::vector<vk_constant_type> constants(13);
        constants[0].i = M;
        constants[1].i = N;
        constants[2].i = K;
        constants[3].i = broadcast_type_C;
        constants[4].i = A.dims;
        constants[5].i = A.dims == 3 ? A.cstep : A.dims == 2 ? A.w : transA ? M : K;
        constants[6].i = B.dims;
        constants[7].i = B.dims == 3 ? B.cstep : B.dims == 2 ? B.w : transB ? K : N;
        constants[8].i = top_blob.dims;
        constants[9].i = top_blob.dims == 3 ? top_blob.cstep : top_blob.w;
        constants[10].i = out_elempack;
        constants[11].i = A_elempack;
        constants[12].i = B_elempack;

        if (opt.use_shader_local_memory)
        {
            VkMat dispatcher;
            dispatcher.w = (N + 3) / 4;
            dispatcher.h = (M + 3) / 4;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_gemm, bindings, constants, dispatcher);
        }
        else if (use_subgroup_ops)
        {
            bindings.resize(7);
            bindings[5] = A;
            bindings[6] = B;

            const int subgroup_size = vkdev->info.subgroup_size();

            const int blocks_x = (M + (UNROLL_SG_M * 4 - 1)) / (UNROLL_SG_M * 4);
            const int blocks_y = (N + (UNROLL_SG_N * 4 - 1)) / (UNROLL_SG_N * 4);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * subgroup_size;
            dispatcher.h = 1;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_gemm, bindings, constants, dispatcher);
        }
        else
        {
            VkMat dispatcher;
            dispatcher.w = (N + 3) / 4;
            dispatcher.h = (M + 3) / 4;
            dispatcher.c = 1;
            cmd.record_pipeline(pipeline_gemm, bindings, constants, dispatcher);
        }
    }

    return 0;
}

int Gemm_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    std::vector<VkMat> bottom_blobs(1);
    std::vector<VkMat> top_blobs(1);
    bottom_blobs[0] = bottom_blob;
    int ret = forward(bottom_blobs, top_blobs, cmd, opt);
    top_blob = top_blobs[0];
    return ret;
}

#if NCNN_INT8
int Gemm_vulkan::create_pipeline_int8(const Option& opt)
{
    Option opt_int8 = opt;
    opt_int8.use_fp16_arithmetic = false;
    opt_int8.use_int16_packed = false;
    opt_int8.use_int16_storage = false;

    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;

    use_cooperative_matrix = vkdev->info.support_int8_cooperative_matrix() && opt.use_cooperative_matrix && opt.use_int8_arithmetic;
    if (use_cooperative_matrix)
    {
        int M = constantM ? constantM : 1024;
        int N = constantN ? constantN : 1024;
        int K = constantK ? constantK : 1024;

        vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_SINT8_KHR, VK_COMPONENT_TYPE_SINT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);

        if (coopmat_M == 0 || coopmat_N == 0 || coopmat_K == 0)
        {
            use_cooperative_matrix = false;
        }
        else
        {
            UNROLL_SG_M = std::min((M + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((N + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((K + coopmat_K - 1) / coopmat_K, 2);

            if (vkdev->info.vendor_id() == 0x5143)
            {
                // adreno int8 register allocation fails with subgroup tile unrolling
                UNROLL_SG_M = 1;
                UNROLL_SG_N = 1;
            }

            UNROLL_WG_M = std::min((M + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((N + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

            // match the aligned packed int8 row strides in gemm_int8_cm
            const bool use_khr_cooperative_matrix = vkdev->info.support_VK_KHR_cooperative_matrix();
            const int Kd4p = use_khr_cooperative_matrix ? (((coopmat_K + 15) / 16) | 1) * 4 : coopmat_K / 4;
            const int Nd4p = use_khr_cooperative_matrix ? (((coopmat_N + 15) / 16) | 1) * 4 : coopmat_N / 4;
            const size_t shared_a = 4 * coopmat_M * Kd4p;
            const size_t shared_b = 4 * coopmat_K * Nd4p;
            const size_t shared_o = 4 * coopmat_M * coopmat_N;

            for (;;)
            {
                const size_t shared_bytes = shared_a * UNROLL_WG_M * UNROLL_SG_M * UNROLL_SG_K
                                            + shared_b * UNROLL_WG_N * UNROLL_SG_N * UNROLL_SG_K
                                            + shared_o * UNROLL_WG_M * UNROLL_WG_N * UNROLL_SG_M * UNROLL_SG_N;
                const uint32_t invocations = coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N;
                if (shared_bytes <= vkdev->info.max_shared_memory_size() && invocations <= vkdev->info.max_workgroup_invocations()
                        && invocations <= vkdev->info.max_workgroup_size_x() && (uint32_t)(UNROLL_WG_M * UNROLL_WG_N) <= vkdev->info.max_compute_workgroup_subgroups())
                    break;

                // reduce K first to preserve output reuse
                if (UNROLL_SG_K > 1)
                    UNROLL_SG_K = 1;
                else if (UNROLL_WG_N > 1)
                    UNROLL_WG_N = 1;
                else if (UNROLL_WG_M > 1)
                    UNROLL_WG_M = 1;
                else if (UNROLL_SG_N > 1)
                    UNROLL_SG_N = 1;
                else if (UNROLL_SG_M > 1)
                    UNROLL_SG_M = 1;
                else
                {
                    use_cooperative_matrix = false;
                    break;
                }
            }
        }
    }

    if (constantA)
    {
        if (use_cooperative_matrix)
        {
            const int blocks_m = (constantM + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int M_aligned = blocks_m * coopmat_M * UNROLL_SG_M * UNROLL_WG_M;
            const int M4_aligned = (M_aligned + 3) / 4;

            A_data_int8_descales.create(M4_aligned, (size_t)16u, 4);
            if (A_data_int8_descales.empty())
                return -100;
            memset(A_data_int8_descales.data, 0, A_data_int8_descales.total() * A_data_int8_descales.elemsize);

            if (A_data.elemsize == (size_t)1u)
            {
                for (int i = 0; i < constantM; i++)
                {
                    const float scale = A_data_int8_scales[i];
                    A_data_int8_descales[i] = scale == 0.f ? 0.f : 1.f / scale;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < constantM; i++)
                {
                    float absmax = 0.f;
                    for (int k = 0; k < constantK; k++)
                    {
                        const float v = transA ? A_data.row(k)[i] : A_data.row(i)[k];
                        absmax = std::max(absmax, v < 0.f ? -v : v);
                    }

                    A_data_int8_descales[i] = absmax == 0.f ? 0.f : absmax * (1.f / 127.f);
                }
            }

            const int kk = (constantK + coopmat_K - 1) / coopmat_K;
            const int kkg = (kk + UNROLL_SG_K - 1) / UNROLL_SG_K;
            const int coopmat_Kd4 = coopmat_K / 4;
            const int coopmat_Kd4p = vkdev->info.support_VK_KHR_cooperative_matrix() ? (((coopmat_Kd4 + 3) / 4) | 1) * 4 : coopmat_Kd4;

            const int A_data_int8_packed_size = coopmat_M * coopmat_Kd4p * UNROLL_SG_K * UNROLL_SG_M * UNROLL_WG_M * kkg;
            A_data_int8_packed.create(A_data_int8_packed_size, blocks_m, (size_t)4u, 4);
            if (A_data_int8_packed.empty())
                return -100;
            A_data_int8_packed.fill(0);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bm = 0; bm < blocks_m; bm++)
            {
                signed char* p = A_data_int8_packed.row<signed char>(bm);

                for (int kg = 0; kg < kkg; kg++)
                {
                    for (int wm = 0; wm < UNROLL_WG_M; wm++)
                    {
                        for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            const int kt = kg * UNROLL_SG_K + zk;
                            for (int zm = 0; zm < UNROLL_SG_M; zm++)
                            {
                                for (int i = 0; i < coopmat_M; i++)
                                {
                                    for (int k4 = 0; k4 < coopmat_Kd4; k4++)
                                    {
                                        for (int r = 0; r < 4; r++)
                                        {
                                            const int gmi = ((bm * UNROLL_WG_M + wm) * UNROLL_SG_M + zm) * coopmat_M + i;
                                            const int gki = kt * coopmat_K + k4 * 4 + r;

                                            if (kt < kk && gmi < constantM && gki < constantK)
                                            {
                                                const int offset = ((kg * UNROLL_WG_M + wm) * (UNROLL_SG_K * UNROLL_SG_M * coopmat_M * coopmat_Kd4p) + ((zk * UNROLL_SG_M + zm) * coopmat_M + i) * coopmat_Kd4p + k4) * 4 + r;
                                                if (A_data.elemsize == (size_t)1u)
                                                {
                                                    p[offset] = transA ? A_data.row<const signed char>(gki)[gmi] : A_data.row<const signed char>(gmi)[gki];
                                                }
                                                else
                                                {
                                                    const float v = transA ? A_data.row(gki)[gmi] : A_data.row(gmi)[gki];
                                                    const float descale = A_data_int8_descales[gmi];
                                                    const float scale = descale == 0.f ? 1.f : 1.f / descale;
                                                    p[offset] = float2int8(v * scale);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            const int K4 = (constantK + 3) / 4;
            const int K4p = (int)alignSize((size_t)K4, 2);
            const int M4 = (constantM + 3) / 4;
            const int M4_aligned = (M4 + 7) / 8 * 8;
            A_data_int8_packed.create(K4p, M4_aligned * 4, (size_t)4u, 4);
            if (A_data_int8_packed.empty())
                return -100;
            A_data_int8_packed.fill(0);

            A_data_int8_descales.create(M4_aligned, (size_t)16u, 4);
            if (A_data_int8_descales.empty())
                return -100;
            memset(A_data_int8_descales.data, 0, A_data_int8_descales.total() * A_data_int8_descales.elemsize);

            if (A_data.elemsize == (size_t)1u)
            {
                for (int i = 0; i < constantM; i++)
                {
                    const float scale = A_data_int8_scales[i];
                    A_data_int8_descales[i] = scale == 0.f ? 0.f : 1.f / scale;
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < constantM; i++)
                {
                    signed char* outptr = A_data_int8_packed.row<signed char>(i);

                    const int K4_full = constantK / 4;
                    for (int k4 = 0; k4 < K4_full; k4++)
                    {
                        const int k = k4 * 4;

                        outptr[0] = transA ? A_data.row<const signed char>(k + 0)[i] : A_data.row<const signed char>(i)[k + 0];
                        outptr[1] = transA ? A_data.row<const signed char>(k + 1)[i] : A_data.row<const signed char>(i)[k + 1];
                        outptr[2] = transA ? A_data.row<const signed char>(k + 2)[i] : A_data.row<const signed char>(i)[k + 2];
                        outptr[3] = transA ? A_data.row<const signed char>(k + 3)[i] : A_data.row<const signed char>(i)[k + 3];
                        outptr += 4;
                    }

                    if (K4_full < K4)
                    {
                        const int k = K4_full * 4;
                        for (int r = 0; k + r < constantK; r++)
                        {
                            outptr[r] = transA ? A_data.row<const signed char>(k + r)[i] : A_data.row<const signed char>(i)[k + r];
                        }
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < constantM; i++)
                {
                    float absmax = 0.f;
                    for (int k = 0; k < constantK; k++)
                    {
                        const float v = transA ? A_data.row(k)[i] : A_data.row(i)[k];
                        absmax = std::max(absmax, v < 0.f ? -v : v);
                    }

                    const float A_int8_scale = absmax == 0.f ? 1.f : 127.f / absmax;
                    A_data_int8_descales[i] = absmax == 0.f ? 0.f : absmax * (1.f / 127.f);

                    signed char* outptr = A_data_int8_packed.row<signed char>(i);

                    const int K4_full = constantK / 4;
                    for (int k4 = 0; k4 < K4_full; k4++)
                    {
                        const int k = k4 * 4;

                        outptr[0] = float2int8((transA ? A_data.row(k + 0)[i] : A_data.row(i)[k + 0]) * A_int8_scale);
                        outptr[1] = float2int8((transA ? A_data.row(k + 1)[i] : A_data.row(i)[k + 1]) * A_int8_scale);
                        outptr[2] = float2int8((transA ? A_data.row(k + 2)[i] : A_data.row(i)[k + 2]) * A_int8_scale);
                        outptr[3] = float2int8((transA ? A_data.row(k + 3)[i] : A_data.row(i)[k + 3]) * A_int8_scale);
                        outptr += 4;
                    }

                    if (K4_full < K4)
                    {
                        const int k = K4_full * 4;
                        for (int r = 0; k + r < constantK; r++)
                        {
                            outptr[r] = float2int8((transA ? A_data.row(k + r)[i] : A_data.row(i)[k + r]) * A_int8_scale);
                        }
                    }
                }
            }
        }
    }

    if (constantB)
    {
        B_data_int8_descale.create(1);
        if (B_data_int8_descale.empty())
            return -100;

        if (use_cooperative_matrix)
        {
            if (B_data.elemsize == (size_t)1u)
            {
                B_data_int8_descale[0] = B_data_int8_scale == 0.f ? 0.f : 1.f / B_data_int8_scale;
            }
            else
            {
                float absmax = 0.f;
                for (int j = 0; j < constantN; j++)
                {
                    for (int k = 0; k < constantK; k++)
                    {
                        const float v = transB ? B_data.row(j)[k] : B_data.row(k)[j];
                        absmax = std::max(absmax, v < 0.f ? -v : v);
                    }
                }

                B_data_int8_descale[0] = absmax == 0.f ? 0.f : absmax * (1.f / 127.f);
            }

            const int blocks_n = (constantN + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            const int kk = (constantK + coopmat_K - 1) / coopmat_K;
            const int kkg = (kk + UNROLL_SG_K - 1) / UNROLL_SG_K;
            const int coopmat_Nd4 = coopmat_N / 4;
            const int coopmat_Nd4p = vkdev->info.support_VK_KHR_cooperative_matrix() ? (((coopmat_Nd4 + 3) / 4) | 1) * 4 : coopmat_Nd4;

            const int B_data_int8_packed_size = coopmat_K * coopmat_Nd4p * UNROLL_SG_K * UNROLL_SG_N * UNROLL_WG_N * kkg;
            B_data_int8_packed.create(B_data_int8_packed_size, blocks_n, (size_t)4u, 4);
            if (B_data_int8_packed.empty())
                return -100;
            B_data_int8_packed.fill(0);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bn = 0; bn < blocks_n; bn++)
            {
                signed char* p = B_data_int8_packed.row<signed char>(bn);

                for (int kg = 0; kg < kkg; kg++)
                {
                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            const int kt = kg * UNROLL_SG_K + zk;
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j4 = 0; j4 < coopmat_Nd4; j4++)
                                    {
                                        for (int r = 0; r < 4; r++)
                                        {
                                            const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j4 * 4 + r;
                                            const int gki = kt * coopmat_K + i;

                                            if (kt < kk && gni < constantN && gki < constantK)
                                            {
                                                const int offset = ((kg * UNROLL_WG_N + wn) * (UNROLL_SG_K * UNROLL_SG_N * coopmat_K * coopmat_Nd4p) + ((zk * UNROLL_SG_N + zn) * coopmat_K + i) * coopmat_Nd4p + j4) * 4 + r;
                                                if (B_data.elemsize == (size_t)1u)
                                                {
                                                    p[offset] = transB ? B_data.row<const signed char>(gni)[gki] : B_data.row<const signed char>(gki)[gni];
                                                }
                                                else
                                                {
                                                    const float v = transB ? B_data.row(gni)[gki] : B_data.row(gki)[gni];
                                                    const float descale = B_data_int8_descale[0];
                                                    const float scale = descale == 0.f ? 1.f : 1.f / descale;
                                                    p[offset] = float2int8(v * scale);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            const int K4 = (constantK + 3) / 4;
            const int K4p = (int)alignSize((size_t)K4, 2);
            const int N4 = (constantN + 3) / 4;
            const int N4_aligned = (N4 + 7) / 8 * 8;
            B_data_int8_packed.create(K4p, N4_aligned * 4, (size_t)4u, 4);
            if (B_data_int8_packed.empty())
                return -100;
            B_data_int8_packed.fill(0);

            if (B_data.elemsize == (size_t)1u)
            {
                B_data_int8_descale[0] = B_data_int8_scale == 0.f ? 0.f : 1.f / B_data_int8_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int j = 0; j < constantN; j++)
                {
                    signed char* outptr = B_data_int8_packed.row<signed char>(j);

                    const int K4_full = constantK / 4;
                    for (int k4 = 0; k4 < K4_full; k4++)
                    {
                        const int k = k4 * 4;

                        outptr[0] = transB ? B_data.row<const signed char>(j)[k + 0] : B_data.row<const signed char>(k + 0)[j];
                        outptr[1] = transB ? B_data.row<const signed char>(j)[k + 1] : B_data.row<const signed char>(k + 1)[j];
                        outptr[2] = transB ? B_data.row<const signed char>(j)[k + 2] : B_data.row<const signed char>(k + 2)[j];
                        outptr[3] = transB ? B_data.row<const signed char>(j)[k + 3] : B_data.row<const signed char>(k + 3)[j];
                        outptr += 4;
                    }

                    if (K4_full < K4)
                    {
                        const int k = K4_full * 4;
                        for (int r = 0; k + r < constantK; r++)
                        {
                            outptr[r] = transB ? B_data.row<const signed char>(j)[k + r] : B_data.row<const signed char>(k + r)[j];
                        }
                    }
                }
            }
            else
            {
                float absmax = 0.f;
                for (int j = 0; j < constantN; j++)
                {
                    for (int k = 0; k < constantK; k++)
                    {
                        const float v = transB ? B_data.row(j)[k] : B_data.row(k)[j];
                        absmax = std::max(absmax, v < 0.f ? -v : v);
                    }
                }

                const float B_int8_scale = absmax == 0.f ? 1.f : 127.f / absmax;
                B_data_int8_descale[0] = absmax == 0.f ? 0.f : absmax * (1.f / 127.f);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int j = 0; j < constantN; j++)
                {
                    signed char* outptr = B_data_int8_packed.row<signed char>(j);

                    const int K4_full = constantK / 4;
                    for (int k4 = 0; k4 < K4_full; k4++)
                    {
                        const int k = k4 * 4;

                        outptr[0] = float2int8((transB ? B_data.row(j)[k + 0] : B_data.row(k + 0)[j]) * B_int8_scale);
                        outptr[1] = float2int8((transB ? B_data.row(j)[k + 1] : B_data.row(k + 1)[j]) * B_int8_scale);
                        outptr[2] = float2int8((transB ? B_data.row(j)[k + 2] : B_data.row(k + 2)[j]) * B_int8_scale);
                        outptr[3] = float2int8((transB ? B_data.row(j)[k + 3] : B_data.row(k + 3)[j]) * B_int8_scale);
                        outptr += 4;
                    }

                    if (K4_full < K4)
                    {
                        const int k = K4_full * 4;
                        for (int r = 0; k + r < constantK; r++)
                        {
                            outptr[r] = float2int8((transB ? B_data.row(j)[k + r] : B_data.row(k + r)[j]) * B_int8_scale);
                        }
                    }
                }
            }
        }
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        C_data_packed = C_data;
    }

    if (!constantA)
    {
        pipeline_gemm_quantize_A_int8 = new Pipeline(vkdev);
        pipeline_gemm_quantize_A_int8->set_optimal_local_size_xyz(Mat(64, 1, 1, (void*)0));
        if (use_cooperative_matrix)
        {
            std::vector<vk_specialization_type> specializations(6);
            specializations[0].i = transA;
            specializations[1].i = coopmat_M;
            specializations[2].i = coopmat_K;
            specializations[3].i = UNROLL_SG_M;
            specializations[4].i = UNROLL_SG_K;
            specializations[5].i = UNROLL_WG_M;
            pipeline_gemm_quantize_A_int8->create(LayerShaderType::gemm_quantize_A_int8, opt_int8, specializations);
        }
        else
        {
            std::vector<vk_specialization_type> specializations(1);
            specializations[0].i = transA;
            pipeline_gemm_quantize_A_int8->create(LayerShaderType::gemm_quantize_A_packed_int8, opt_int8, specializations);
        }
    }

    if (!constantB)
    {
        std::vector<vk_specialization_type> specializations_absmax(1);
        specializations_absmax[0].i = transB;

        pipeline_gemm_quantize_B_absmax_int8 = new Pipeline(vkdev);
        pipeline_gemm_quantize_B_absmax_int8->set_local_size_xyz(128, 1, 1);
        pipeline_gemm_quantize_B_absmax_int8->create(LayerShaderType::gemm_quantize_B_absmax_int8, opt_int8, specializations_absmax);

        pipeline_gemm_quantize_B_descale_int8 = new Pipeline(vkdev);
        pipeline_gemm_quantize_B_descale_int8->set_local_size_xyz(128, 1, 1);
        pipeline_gemm_quantize_B_descale_int8->create(LayerShaderType::gemm_quantize_B_descale_int8, opt_int8, std::vector<vk_specialization_type>());

        pipeline_gemm_quantize_B_int8 = new Pipeline(vkdev);
        pipeline_gemm_quantize_B_int8->set_optimal_local_size_xyz(Mat(64, 1, 1, (void*)0));
        if (use_cooperative_matrix)
        {
            std::vector<vk_specialization_type> specializations(6);
            specializations[0].i = transB;
            specializations[1].i = coopmat_N;
            specializations[2].i = coopmat_K;
            specializations[3].i = UNROLL_SG_N;
            specializations[4].i = UNROLL_SG_K;
            specializations[5].i = UNROLL_WG_N;
            pipeline_gemm_quantize_B_int8->create(LayerShaderType::gemm_quantize_B_int8, opt_int8, specializations);
        }
        else
        {
            std::vector<vk_specialization_type> specializations(1);
            specializations[0].i = transB;
            pipeline_gemm_quantize_B_int8->create(LayerShaderType::gemm_quantize_B_packed_int8, opt_int8, specializations);
        }
    }

    if (use_cooperative_matrix)
    {
        int outh = output_transpose ? constantN : constantM;
        int out_elempack = outh ? (outh % 4 == 0 ? 4 : 1) : 0;
        if (output_elempack)
            out_elempack = output_elempack;

        std::vector<vk_specialization_type> specializations(9 + 9);
        specializations[0].f = alpha;
        specializations[1].f = beta;
        specializations[2].i = constantC;
        specializations[3].i = constant_broadcast_type_C;
        specializations[4].i = output_transpose;
        specializations[5].u32 = constantM;
        specializations[6].u32 = constantN;
        specializations[7].u32 = constantK;
        specializations[8].u32 = out_elempack;

        specializations[9 + 0].u32 = coopmat_M;
        specializations[9 + 1].u32 = coopmat_N;
        specializations[9 + 2].u32 = coopmat_K;
        specializations[9 + 3].u32 = coopmat_subgroup_size;
        specializations[9 + 4].u32 = UNROLL_SG_M;
        specializations[9 + 5].u32 = UNROLL_SG_N;
        specializations[9 + 6].u32 = UNROLL_SG_K;
        specializations[9 + 7].u32 = UNROLL_WG_M;
        specializations[9 + 8].u32 = UNROLL_WG_N;

        // specialize dynamic output packing to avoid adreno store layout branches
        const bool use_output_pack_variants = vkdev->info.vendor_id() == 0x5143 && output_transpose == 0 && out_elempack == 0;
        if (use_output_pack_variants)
            specializations[8].u32 = 1;

        pipeline_gemm = new Pipeline(vkdev);
        pipeline_gemm->set_subgroup_size(coopmat_subgroup_size);
        pipeline_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
        pipeline_gemm->create(LayerShaderType::gemm_int8_cm, opt_int8, specializations);

        if (use_output_pack_variants)
        {
            specializations[8].u32 = 4;

            pipeline_gemm_pack4 = new Pipeline(vkdev);
            pipeline_gemm_pack4->set_subgroup_size(coopmat_subgroup_size);
            pipeline_gemm_pack4->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_gemm_pack4->create(LayerShaderType::gemm_int8_cm, opt_int8, specializations);
        }
    }
    else
    {
        std::vector<vk_specialization_type> specializations(5);
        specializations[0].f = alpha;
        specializations[1].f = beta;
        specializations[2].i = constantC;
        specializations[3].i = constant_broadcast_type_C;
        specializations[4].i = output_transpose;

        pipeline_gemm = new Pipeline(vkdev);
        pipeline_gemm->set_local_size_xyz(8, 8, 1);
        pipeline_gemm->create(LayerShaderType::gemm_int8, opt_int8, specializations);
    }

    if (opt.lightmode)
    {
        A_data.release();
        B_data.release();
        C_data.release();
    }

    return 0;
}

int Gemm_vulkan::upload_model_int8(VkTransfer& cmd, const Option& opt)
{
    Option opt_fp32 = opt;
    opt_fp32.use_fp16_packed = false;
    opt_fp32.use_fp16_storage = false;
    opt_fp32.use_bf16_packed = false;
    opt_fp32.use_bf16_storage = false;

    if (constantA)
    {
        cmd.record_upload(A_data_int8_packed, A_data_gpu, opt);

        A_data_int8_packed.release();

        cmd.record_upload(A_data_int8_descales, A_data_int8_descales_gpu, opt_fp32);

        A_data_int8_descales.release();
        A_data_int8_scales.release();
    }

    if (constantB)
    {
        cmd.record_upload(B_data_int8_packed, B_data_gpu, opt);

        B_data_int8_packed.release();

        cmd.record_upload(B_data_int8_descale, B_data_int8_descale_gpu, opt_fp32);

        B_data_int8_descale.release();
    }

    if (constantC && constant_broadcast_type_C != -1)
    {
        cmd.record_upload(C_data_packed, C_data_gpu, opt);

        C_data_packed.release();
    }

    return 0;
}

int Gemm_vulkan::forward_int8(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& A0 = constantA ? A_data_gpu : bottom_blobs[0];
    const VkMat& B0 = constantB ? B_data_gpu : constantA ? bottom_blobs[0] : bottom_blobs[1];

    VkMat A = A0;
    VkMat B = B0;

    // dynamic int8 input has no scale
    if (!constantA && A.elembits() == 8)
    {
        NCNN_LOGE("Gemm_vulkan int8 dynamic int8 A is not supported without input scale");
        return -1;
    }

    if (!constantB && B.elembits() == 8)
    {
        NCNN_LOGE("Gemm_vulkan int8 dynamic int8 B is not supported without input scale");
        return -1;
    }

    if (!constantA && A.elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        VkMat A_unpacked;
        vkdev->convert_packing(A, A_unpacked, 1, cmd, opt_pack1);
        A = A_unpacked;
    }

    if (!constantB && B.elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        VkMat B_unpacked;
        vkdev->convert_packing(B, B_unpacked, 1, cmd, opt_pack1);
        B = B_unpacked;
    }

    const int M = constantM ? constantM : transA ? A.w : (A.dims == 3 ? A.c : A.h);
    const int K = constantK ? constantK : transA ? (A.dims == 3 ? A.c : A.h) : A.w;
    const int N = constantN ? constantN : transB ? (B.dims == 3 ? B.c : B.h) : B.w;

    VkMat C;
    int broadcast_type_C = -1;
    if (constantC && constant_broadcast_type_C != -1)
    {
        C = C_data_gpu;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        VkMat C0;
        if (constantA && constantB)
        {
            C0 = bottom_blobs.size() == 1 ? bottom_blobs[0] : VkMat();
        }
        else if (constantA)
        {
            C0 = bottom_blobs.size() == 2 ? bottom_blobs[1] : VkMat();
        }
        else if (constantB)
        {
            C0 = bottom_blobs.size() == 2 ? bottom_blobs[1] : VkMat();
        }
        else
        {
            C0 = bottom_blobs.size() == 3 ? bottom_blobs[2] : VkMat();
        }

        if (!C0.empty())
        {
            C = C0;
            if (C.elempack != 1)
            {
                Option opt_pack1 = opt;
                opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

                VkMat C_unpacked;
                vkdev->convert_packing(C, C_unpacked, 1, cmd, opt_pack1);
                C = C_unpacked;
            }

            if (C.dims == 1 && C.w == 1)
            {
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w == M)
            {
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w == N)
            {
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h == M)
            {
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h == M)
            {
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h == 1)
            {
                broadcast_type_C = 4;
            }
        }
    }

    if (!C.empty() && C.elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        VkMat C_unpacked;
        vkdev->convert_packing(C, C_unpacked, 1, cmd, opt_pack1);
        C = C_unpacked;
    }

    int out_elempack = 1;
    {
        int outh = output_transpose ? N : M;
        out_elempack = outh % 4 == 0 ? 4 : 1;
        if (output_elempack)
            out_elempack = output_elempack;
    }

    size_t elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
    {
        elemsize = 2u;
    }
    else
    {
        elemsize = 4u;
    }
    size_t out_elemsize = elemsize * out_elempack;

    VkMat A_int8 = A;
    VkMat A_int8_descales = A_data_int8_descales_gpu;
    if (!constantA)
    {
        if (use_cooperative_matrix)
        {
            const int blocks_m = (M + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int kk = (K + coopmat_K - 1) / coopmat_K;
            const int kkg = (kk + UNROLL_SG_K - 1) / UNROLL_SG_K;
            const int coopmat_Kd4 = coopmat_K / 4;
            const int coopmat_Kd4p = vkdev->info.support_VK_KHR_cooperative_matrix() ? (((coopmat_Kd4 + 3) / 4) | 1) * 4 : coopmat_Kd4;
            const int M_aligned = blocks_m * coopmat_M * UNROLL_SG_M * UNROLL_WG_M;
            const int M4_aligned = (M_aligned + 3) / 4;
            const int A_int8_packed_size = coopmat_M * coopmat_Kd4p * UNROLL_SG_K * UNROLL_SG_M * UNROLL_WG_M * kkg;

            A_int8.create(A_int8_packed_size, blocks_m, (size_t)4u, 4, opt.workspace_vkallocator);
            if (A_int8.empty())
                return -100;

            A_int8_descales.create(M4_aligned, (size_t)16u, 4, opt.workspace_vkallocator);
            if (A_int8_descales.empty())
                return -100;

            std::vector<VkMat> bindings(3);
            bindings[0] = A;
            bindings[1] = A_int8;
            bindings[2] = A_int8_descales;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = M;
            constants[1].i = K;
            constants[2].i = A.dims == 3 ? A.cstep : A.dims == 2 ? A.w : transA ? M : K;

            VkMat dispatcher;
            dispatcher.w = M_aligned;
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_gemm_quantize_A_int8, bindings, constants, dispatcher);
        }
        else
        {
            const int K4 = (K + 3) / 4;
            const int K4p = (int)alignSize((size_t)K4, 2);
            const int M4 = (M + 3) / 4;
            const int M4_aligned = (M4 + 7) / 8 * 8;
            A_int8.create(K4p, M4_aligned * 4, (size_t)4u, 4, opt.workspace_vkallocator);
            if (A_int8.empty())
                return -100;

            A_int8_descales.create(M4_aligned, (size_t)16u, 4, opt.workspace_vkallocator);
            if (A_int8_descales.empty())
                return -100;

            std::vector<VkMat> bindings(3);
            bindings[0] = A;
            bindings[1] = A_int8;
            bindings[2] = A_int8_descales;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = M;
            constants[1].i = K;
            constants[2].i = A.dims == 3 ? A.cstep : A.dims == 2 ? A.w : transA ? M : K;

            VkMat dispatcher;
            dispatcher.w = M4_aligned * 4;
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_gemm_quantize_A_int8, bindings, constants, dispatcher);
        }
    }

    VkMat B_int8 = B;
    VkMat B_int8_descale = B_data_int8_descale_gpu;
    if (!constantB)
    {
        const int size = N * K;
        const int blocks = (size + 1023) / 1024;
        const int K4 = (K + 3) / 4;
        const int K4p = (int)alignSize((size_t)K4, 2);
        const int N4 = (N + 3) / 4;
        const int N4_aligned = (N4 + 7) / 8 * 8;

        if (use_cooperative_matrix)
        {
            const int blocks_n = (N + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            const int kk = (K + coopmat_K - 1) / coopmat_K;
            const int kkg = (kk + UNROLL_SG_K - 1) / UNROLL_SG_K;
            const int coopmat_Nd4 = coopmat_N / 4;
            const int coopmat_Nd4p = vkdev->info.support_VK_KHR_cooperative_matrix() ? (((coopmat_Nd4 + 3) / 4) | 1) * 4 : coopmat_Nd4;
            const int B_int8_packed_size = coopmat_K * coopmat_Nd4p * UNROLL_SG_K * UNROLL_SG_N * UNROLL_WG_N * kkg;
            B_int8.create(B_int8_packed_size, blocks_n, (size_t)4u, 4, opt.workspace_vkallocator);
        }
        else
        {
            B_int8.create(K4p, N4_aligned * 4, (size_t)4u, 4, opt.workspace_vkallocator);
        }
        if (B_int8.empty())
            return -100;

        B_int8_descale.create(1, (size_t)4u, 1, opt.workspace_vkallocator);
        if (B_int8_descale.empty())
            return -100;

        VkMat B_absmax;
        B_absmax.create(blocks, (size_t)4u, 1, opt.workspace_vkallocator);
        if (B_absmax.empty())
            return -100;

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = B;
            bindings[1] = B_absmax;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = N;
            constants[1].i = K;
            constants[2].i = B.dims == 3 ? B.cstep : B.dims == 2 ? B.w : transB ? K : N;

            VkMat dispatcher;
            dispatcher.w = blocks * 128;
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_gemm_quantize_B_absmax_int8, bindings, constants, dispatcher);
        }

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = B_absmax;
            bindings[1] = B_int8_descale;

            std::vector<vk_constant_type> constants(1);
            constants[0].i = blocks;

            VkMat dispatcher;
            dispatcher.w = 1;
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_gemm_quantize_B_descale_int8, bindings, constants, dispatcher);
        }

        if (use_cooperative_matrix)
        {
            std::vector<VkMat> bindings(3);
            bindings[0] = B;
            bindings[1] = B_int8;
            bindings[2] = B_int8_descale;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = N;
            constants[1].i = K;
            constants[2].i = B.dims == 3 ? B.cstep : B.dims == 2 ? B.w : transB ? K : N;

            VkMat dispatcher;
            dispatcher.w = B_int8.w * B_int8.h;
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_gemm_quantize_B_int8, bindings, constants, dispatcher);
        }
        else
        {
            std::vector<VkMat> bindings(3);
            bindings[0] = B;
            bindings[1] = B_int8;
            bindings[2] = B_int8_descale;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = N;
            constants[1].i = K;
            constants[2].i = B.dims == 3 ? B.cstep : B.dims == 2 ? B.w : transB ? K : N;

            VkMat dispatcher;
            dispatcher.w = N4_aligned * 4 * K4p;
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_gemm_quantize_B_int8, bindings, constants, dispatcher);
        }
    }

    VkMat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        else
            top_blob.create(M, N / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        else
            top_blob.create(N, M / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    if (use_cooperative_matrix)
    {
        std::vector<VkMat> bindings(7);
        bindings[0] = top_blob;
        bindings[1] = A_int8;
        bindings[2] = B_int8;
        bindings[3] = C;
        bindings[4] = top_blob;
        bindings[5] = A_int8_descales;
        bindings[6] = B_int8_descale;

        std::vector<vk_constant_type> constants(6);
        constants[0].u32 = M;
        constants[1].u32 = N;
        constants[2].u32 = K;
        constants[3].i = broadcast_type_C;
        constants[4].u32 = top_blob.dims == 3 ? top_blob.cstep : top_blob.w;
        constants[5].u32 = out_elempack;

        VkMat dispatcher;
        const int blocks_x = (M + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
        const int blocks_y = (N + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

        dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
        dispatcher.h = 1;
        dispatcher.c = 1;

        const Pipeline* pipeline = out_elempack == 4 && pipeline_gemm_pack4 ? pipeline_gemm_pack4 : pipeline_gemm;
        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
    }
    else
    {
        std::vector<VkMat> bindings(7);
        bindings[0] = top_blob;
        bindings[1] = A_int8;
        bindings[2] = B_int8;
        bindings[3] = C;
        bindings[4] = top_blob;
        bindings[5] = A_int8_descales;
        bindings[6] = B_int8_descale;

        std::vector<vk_constant_type> constants(6);
        constants[0].i = M;
        constants[1].i = N;
        constants[2].i = K;
        constants[3].i = broadcast_type_C;
        constants[4].i = top_blob.dims == 3 ? top_blob.cstep : top_blob.w;
        constants[5].i = out_elempack;

        VkMat dispatcher;
        dispatcher.w = (N + 3) / 4;
        dispatcher.h = (M + 3) / 4;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_gemm, bindings, constants, dispatcher);
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
