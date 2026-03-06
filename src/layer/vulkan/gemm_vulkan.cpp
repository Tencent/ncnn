// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Gemm_vulkan::Gemm_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;

    pipeline_gemm = 0;

    use_subgroup_ops = false;

    use_cooperative_matrix = false;
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

int Gemm_vulkan::load_param(const ParamDict& pd)
{
    int ret = Gemm::load_param(pd);

    if (int8_scale_term)
    {
        support_vulkan = false;
    }

    return ret;
}

int Gemm_vulkan::create_pipeline(const Option& opt)
{
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
    if (vkdev->info.support_bf16_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_bf16_storage || opt.use_bf16_packed))
    {
        use_cooperative_matrix = true;
        use_bf16_cooperative_matrix = true;
    }

    const int subgroup_size = vkdev->info.subgroup_size();
    use_subgroup_ops = opt.use_subgroup_ops && (vkdev->info.support_subgroup_ops() & (VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT));
    if (subgroup_size < 4 || subgroup_size > 128)
    {
        // sanitize wired subgroup_size
        use_subgroup_ops = false;
    }

    if (use_cooperative_matrix)
    {
        int M = constantM ? constantM : 1024;
        int N = constantN ? constantN : 1024;
        int K = constantK ? constantK : 1024;

        if (use_bf16_cooperative_matrix)
        {
            vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);
        }
        else
        {
            vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);
        }

        // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

        UNROLL_SG_M = std::min((M + coopmat_M - 1) / coopmat_M, 2);
        UNROLL_SG_N = std::min((N + coopmat_N - 1) / coopmat_N, 2);
        UNROLL_SG_K = std::min((K + coopmat_K - 1) / coopmat_K, 2);

        UNROLL_WG_M = std::min((M + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
        UNROLL_WG_N = std::min((N + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

        if (constantA == 1)
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

        if (constantB == 1)
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

        pipeline_gemm = new Pipeline(vkdev);
        pipeline_gemm->set_subgroup_size(coopmat_subgroup_size);
        pipeline_gemm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
        pipeline_gemm->create(LayerShaderType::gemm_cm, opt, specializations);
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

    use_subgroup_ops = false;

    use_cooperative_matrix = false;
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
    const VkMat& A0 = constantA ? A_data_gpu : bottom_blobs[0];
    const VkMat& B0 = constantB ? B_data_gpu : constantA ? bottom_blobs[0] : bottom_blobs[1];

    VkMat A = A0;
    VkMat B = B0;

    if (constantA && !vkdev->is_device_local(A0.data->memory_type_index))
    {
        cmd.record_clone(A0, A, opt);
    }
    if (constantB && !vkdev->is_device_local(B0.data->memory_type_index))
    {
        cmd.record_clone(B0, B, opt);
    }

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

    if (use_cooperative_matrix)
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

        cmd.record_pipeline(pipeline_gemm, bindings, constants, dispatcher);
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

} // namespace ncnn
