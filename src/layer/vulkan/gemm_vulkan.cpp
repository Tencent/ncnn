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

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
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

    // int elempack = 1;
    // if (shape.dims == 2) elempack = shape.h % 4 == 0 ? 4 : 1;

    // size_t elemsize;
    // if (opt.use_fp16_storage || opt.use_fp16_packed)
    // {
    //     elemsize = elempack * 2u;
    // }
    // else
    // {
    //     elemsize = elempack * 4u;
    // }

    // Mat shape_packed;
    // if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);

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

    if (use_cooperative_matrix)
    {
        int M = constantM ? constantM : 1024;
        int N = constantN ? constantN : 1024;
        int K = constantK ? constantK : 1024;

        vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K);

        // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

        UNROLL_SG_M = std::min((M + coopmat_M - 1) / coopmat_M, 2);
        UNROLL_SG_N = std::min((N + coopmat_N - 1) / coopmat_N, 2);
        UNROLL_SG_K = std::min((K + coopmat_K - 1) / coopmat_K, 2);

        UNROLL_WG_M = std::min((M + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
        UNROLL_WG_N = std::min((N + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

        std::vector<vk_specialization_type> specializations(15 + 8);
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

        specializations[15].u32 = coopmat_M;
        specializations[16].u32 = coopmat_N;
        specializations[17].u32 = coopmat_K;
        specializations[18].u32 = UNROLL_SG_M;
        specializations[19].u32 = UNROLL_SG_N;
        specializations[20].u32 = UNROLL_SG_K;
        specializations[21].u32 = UNROLL_WG_M;
        specializations[22].u32 = UNROLL_WG_N;

        const int subgroup_size = vkdev->info.subgroup_size();

        pipeline_gemm = new Pipeline(vkdev);
        pipeline_gemm->set_subgroup_size(subgroup_size);
        pipeline_gemm->set_local_size_xyz(subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
        pipeline_gemm->create(LayerShaderType::gemm_cm, opt, specializations);
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
        // if (shape_packed.dims == 2)
        // {
        //     local_size_xyz.w = std::min(8, shape_packed.w);
        //     local_size_xyz.h = std::min(8, shape_packed.h);
        //     local_size_xyz.c = 1;
        // }

        // pack1
        // if (shape.dims == 0 || elempack == 1)
        {
            pipeline_gemm = new Pipeline(vkdev);
            pipeline_gemm->set_optimal_local_size_xyz(local_size_xyz);
            if (opt.use_shader_local_memory)
            {
                pipeline_gemm->set_local_size_xyz(8, 8, 1);
            }
            pipeline_gemm->create(LayerShaderType::gemm, opt, specializations);
        }
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

    VkMat A;
    VkMat B;
    vkdev->convert_packing(A0, A, 1, cmd, opt);
    vkdev->convert_packing(B0, B, 1, cmd, opt);

    const int M = constantM ? constantM : transA ? A.w : (A.dims == 3 ? A.c : A.h);
    const int K = constantK ? constantK : transA ? (A.dims == 3 ? A.c : A.h) : A.w;
    const int N = constantN ? constantN : transB ? (B.dims == 3 ? B.c : B.h) : B.w;

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

    int elempack = A.elempack;
    size_t elemsize = A.elemsize;

    VkMat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N, elemsize, opt.blob_vkallocator);
        else
            top_blob.create(M, N, elemsize, opt.blob_vkallocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M, elemsize, opt.blob_vkallocator);
        else
            top_blob.create(N, M, elemsize, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = top_blob;
    bindings[1] = A;
    bindings[2] = B;
    bindings[3] = C;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = M;
    constants[1].i = N;
    constants[2].i = K;
    constants[3].i = broadcast_type_C;
    constants[4].i = A.dims;
    constants[5].i = A.dims == 3 ? A.cstep : transA ? M : K;
    constants[6].i = B.dims;
    constants[7].i = B.dims == 3 ? B.cstep : transB ? K : N;
    constants[8].i = top_blob.dims;
    constants[9].i = top_blob.dims == 3 ? top_blob.cstep : top_blob.w;

    if (use_cooperative_matrix)
    {
        const int blocks_x = (M + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
        const int blocks_y = (N + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

        const int subgroup_size = vkdev->info.subgroup_size();

        VkMat dispatcher;
        dispatcher.w = (blocks_x * blocks_y) * (subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_gemm, bindings, constants, dispatcher);
    }
    else
    {
        const Pipeline* pipeline = pipeline_gemm;

        VkMat dispatcher;
        dispatcher.w = (N + 1) / 2;
        dispatcher.h = (M + 1) / 2;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
    }

    int out_elempack = 1;
    {
        int outh = output_transpose ? N : M;
        out_elempack = outh % 4 == 0 ? 4 : 1;
    }
    if (output_elempack)
        out_elempack = output_elempack;

    if (out_elempack != 1)
    {
        VkMat top_blob0;
        vkdev->convert_packing(top_blob, top_blob0, out_elempack, cmd, opt);
        top_blobs[0] = top_blob0;
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
