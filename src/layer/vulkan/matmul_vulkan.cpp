// Copyright 2026 MYQ
// SPDX-License-Identifier: BSD-3-Clause

#include "matmul_vulkan.h"

#include <algorithm>
#include <cstdint>

#include "layer_shader_type.h"

namespace ncnn {

MatMul_vulkan::MatMul_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;

    pipeline_matmul = 0;
    pipeline_matmul_sg = 0;
    pipeline_matmul_cm = 0;

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

int MatMul_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> matmul_specializations(1);
    matmul_specializations[0].i = transB;

    pipeline_matmul = new Pipeline(vkdev);
    if (opt.use_shader_local_memory)
    {
        pipeline_matmul->set_local_size_xyz(8, 8, 1);
    }
    else
    {
        Mat local_size_xyz;
        pipeline_matmul->set_optimal_local_size_xyz(local_size_xyz);
    }

    int ret = pipeline_matmul->create(LayerShaderType::matmul, opt, matmul_specializations);
    if (ret != 0)
    {
        destroy_pipeline(opt);
        return ret;
    }

    const int subgroup_size = vkdev->info.subgroup_size();
    const uint32_t subgroup_features = vkdev->info.support_subgroup_ops();
    const bool support_subgroup_shuffle = (subgroup_features & (VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT)) == (VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT);

    use_subgroup_ops = opt.use_subgroup_ops && support_subgroup_shuffle;
    if (subgroup_size < 4 || subgroup_size > 128)
    {
        // sanitize weird subgroup_size
        use_subgroup_ops = false;
    }

    if (use_subgroup_ops)
    {
        std::vector<vk_specialization_type> sg_specializations(2);
        sg_specializations[0].i = transB;
        sg_specializations[1].u32 = subgroup_size;

        pipeline_matmul_sg = new Pipeline(vkdev);
        pipeline_matmul_sg->set_subgroup_size(subgroup_size);
        pipeline_matmul_sg->set_local_size_xyz(subgroup_size, 1, 1);
        ret = pipeline_matmul_sg->create(LayerShaderType::matmul_sg, opt, sg_specializations);
        if (ret != 0)
        {
            delete pipeline_matmul_sg;
            pipeline_matmul_sg = 0;
            use_subgroup_ops = false;
        }
    }

    if (vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed))
    {
        int M = 1024;
        int N = 1024;
        int K = 1024;
        vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);
        if (coopmat_M > 0 && coopmat_N > 0 && coopmat_K > 0 && coopmat_subgroup_size >= 4 && coopmat_subgroup_size <= 128)
        {
            use_cooperative_matrix = true;

            UNROLL_SG_M = std::min((M + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((N + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((K + coopmat_K - 1) / coopmat_K, 2);
            UNROLL_WG_M = std::min((M + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((N + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

            std::vector<vk_specialization_type> cm_specializations(5);
            cm_specializations[0].i = transB;
            cm_specializations[1].u32 = coopmat_M;
            cm_specializations[2].u32 = coopmat_N;
            cm_specializations[3].u32 = coopmat_K;
            cm_specializations[4].u32 = coopmat_subgroup_size;

            pipeline_matmul_cm = new Pipeline(vkdev);
            pipeline_matmul_cm->set_subgroup_size(coopmat_subgroup_size);
            pipeline_matmul_cm->set_local_size_xyz(coopmat_subgroup_size, 1, 1);
            ret = pipeline_matmul_cm->create(LayerShaderType::matmul_cm, opt, cm_specializations);            
            if (ret != 0)
            {
                delete pipeline_matmul_cm;
                pipeline_matmul_cm = 0;

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
        }
    }

    return 0;
}

int MatMul_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_matmul;
    pipeline_matmul = 0;

    delete pipeline_matmul_sg;
    pipeline_matmul_sg = 0;

    delete pipeline_matmul_cm;
    pipeline_matmul_cm = 0;

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

int MatMul_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& A0 = bottom_blobs[0];
    const VkMat& B0 = bottom_blobs[1];

    VkMat A;
    VkMat B;
    vkdev->convert_packing(A0, A, 1, cmd, opt);
    vkdev->convert_packing(B0, B, 1, cmd, opt);

    const int Adims = A.dims;
    const int Bdims = B.dims;

    if (Adims < 1 || Adims > 4 || Bdims < 1 || Bdims > 4)
    {
        NCNN_LOGE("unsupported matmul dims A=%d B=%d", Adims, Bdims);
        return -1;
    }

    const int max_ABdims = std::max(Adims, Bdims);

    // For max rank 4, MatMul cpu semantics reshape 3d tensor (w,h,c) to (w,h,d=c,c=1).
    const bool A_reshape_3d_to_4d = max_ABdims == 4 && Adims == 3;
    const bool B_reshape_3d_to_4d = max_ABdims == 4 && Bdims == 3;

    const int A_batch_c = A_reshape_3d_to_4d ? 1 : (Adims >= 3 ? A.c : 1);
    const int A_batch_d = A_reshape_3d_to_4d ? A.c : (Adims == 4 ? A.d : 1);
    const int B_batch_c = B_reshape_3d_to_4d ? 1 : (Bdims >= 3 ? B.c : 1);
    const int B_batch_d = B_reshape_3d_to_4d ? B.c : (Bdims == 4 ? B.d : 1);

    const int A_dstep = A_reshape_3d_to_4d ? (int)A.cstep : A.w * A.h;
    const int B_dstep = B_reshape_3d_to_4d ? (int)B.cstep : B.w * B.h;

    const int M = Adims == 1 ? 1 : A.h;
    const int K = A.w;
    const int N = Bdims == 1 ? 1 : (transB ? B.h : B.w);

    const int BK = Bdims == 1 ? B.w : (transB ? B.w : B.h);
    if (K != BK)
    {
        NCNN_LOGE("matmul K mismatch A=%d B=%d transB=%d", K, BK, transB);
        return -1;
    }

    const bool batch_c_compatible = A_batch_c == B_batch_c || A_batch_c == 1 || B_batch_c == 1;
    const bool batch_d_compatible = A_batch_d == B_batch_d || A_batch_d == 1 || B_batch_d == 1;
    if (!batch_c_compatible || !batch_d_compatible)
    {
        NCNN_LOGE("matmul batch mismatch A(c=%d d=%d) B(c=%d d=%d)", A_batch_c, A_batch_d, B_batch_c, B_batch_d);
        return -1;
    }

    const int A_layout = Adims == 1 ? 2 : (Adims <= 2 ? 0 : 1);
    const int B_layout = Bdims == 1 ? 2 : (Bdims <= 2 ? 0 : 1);

    int out_layout = 0;
    int out_batch_c = 1;
    int out_batch_d = 1;

    VkMat& top_blob = top_blobs[0];
    const size_t elemsize = A.elemsize;

    if (Adims == 1 && Bdims == 1)
    {
        out_layout = 1;
        top_blob.create(1, elemsize, opt.blob_vkallocator);
    }
    else if (Adims == 2 && Bdims == 2)
    {
        out_layout = 0;
        top_blob.create(N, M, elemsize, opt.blob_vkallocator);
    }
    else if (Adims == 1 && Bdims == 2)
    {
        out_layout = 1;
        top_blob.create(N, elemsize, opt.blob_vkallocator);
    }
    else if (Adims == 2 && Bdims == 1)
    {
        out_layout = 2;
        top_blob.create(M, elemsize, opt.blob_vkallocator);
    }
    else if (Adims == 1 && Bdims > 2)
    {
        out_layout = 4;

        if (Bdims == 3)
        {
            out_batch_d = B_batch_d * B_batch_c;
            out_batch_c = 1;
            top_blob.create(N, out_batch_d, elemsize, opt.blob_vkallocator);
        }
        else
        {
            out_batch_d = B_batch_d;
            out_batch_c = B_batch_c;
            top_blob.create(N, out_batch_d, out_batch_c, elemsize, opt.blob_vkallocator);
        }
    }
    else if (Adims > 2 && Bdims == 1)
    {
        out_layout = 5;

        if (Adims == 3)
        {
            out_batch_d = A_batch_d * A_batch_c;
            out_batch_c = 1;
            top_blob.create(M, out_batch_d, elemsize, opt.blob_vkallocator);
        }
        else
        {
            out_batch_d = A_batch_d;
            out_batch_c = A_batch_c;
            top_blob.create(M, out_batch_d, out_batch_c, elemsize, opt.blob_vkallocator);
        }
    }
    else if (max_ABdims == 3)
    {
        out_layout = 3;
        out_batch_d = 1;
        out_batch_c = std::max(A_batch_c, B_batch_c);
        top_blob.create(N, M, out_batch_c, elemsize, opt.blob_vkallocator);
    }
    else if (max_ABdims == 4)
    {
        out_layout = 3;
        out_batch_d = std::max(A_batch_d, B_batch_d);
        out_batch_c = std::max(A_batch_c, B_batch_c);
        top_blob.create(N, M, out_batch_d, out_batch_c, elemsize, opt.blob_vkallocator);
    }
    else
    {
        NCNN_LOGE("impossible matmul %d %d", Adims, Bdims);
        return -1;
    }

    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = top_blob;
    bindings[1] = A;
    bindings[2] = B;

    std::vector<vk_constant_type> constants(23);
    constants[0].i = M;
    constants[1].i = N;
    constants[2].i = K;
    constants[3].i = out_batch_c * out_batch_d;
    constants[4].i = A_layout;
    constants[5].i = B_layout;
    constants[6].i = out_layout;

    constants[7].i = A.w;
    constants[8].i = (int)A.cstep;
    constants[9].i = A_dstep;
    constants[10].i = A_batch_c;
    constants[11].i = A_batch_d;

    // B_hstep is the physical row stride (w) of B in memory.
    // Indexing formula in shader switches on transB to access B[k, n] or B[n, k].
    constants[12].i = B.w;
    constants[13].i = (int)B.cstep;
    constants[14].i = B_dstep;
    constants[15].i = B_batch_c;
    constants[16].i = B_batch_d;

    constants[17].i = top_blob.dims >= 2 ? top_blob.w : 0;
    constants[18].i = top_blob.dims >= 3 ? (int)top_blob.cstep : 0;
    constants[19].i = top_blob.w * (top_blob.dims >= 2 ? top_blob.h : 1);
    constants[20].i = out_batch_c;
    constants[21].i = out_batch_d;
    constants[22].i = transB;

    const int batch = out_batch_c * out_batch_d;

    Pipeline* selected_pipeline = pipeline_matmul;

    VkMat dispatcher;
    dispatcher.w = N;
    dispatcher.h = M;
    dispatcher.c = batch;

    if (pipeline_matmul_cm && use_cooperative_matrix)
    {
        const bool fp16_input = A.elemsize == 2u && B.elemsize == 2u;
        const int64_t work = (int64_t)M * N * K;
        const int64_t tile = (int64_t)coopmat_M * coopmat_N * coopmat_K;

        if (fp16_input && work >= tile)
        {
            // NCNN_LOGE("pipeline_matmul_cm");
            selected_pipeline = pipeline_matmul_cm;
            dispatcher.w = ((N + coopmat_N - 1) / coopmat_N) * coopmat_subgroup_size;
            dispatcher.h = (M + coopmat_M - 1) / coopmat_M;
            dispatcher.c = batch;
        }
    }

    if (selected_pipeline == pipeline_matmul && pipeline_matmul_sg && use_subgroup_ops)
    {
        const int64_t work = (int64_t)M * N * K;
        if (work >= 512)
        {
            // NCNN_LOGE("pipeline_matmul_sg");
            selected_pipeline = pipeline_matmul_sg;
            dispatcher.w = N * vkdev->info.subgroup_size();
            dispatcher.h = M;
            dispatcher.c = batch;
        }
    }
    cmd.record_pipeline(selected_pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
