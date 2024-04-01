// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "gemm_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Gemm_vulkan::Gemm_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_gemm = 0;
}

int Gemm_vulkan::create_pipeline(const Option& opt)
{
    // const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // int elempack = 1;
    // if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;

    // size_t elemsize;
    // if (opt.use_fp16_storage)
    // {
    //     elemsize = elempack * 2u;
    // }
    // else if (opt.use_fp16_packed)
    // {
    //     elemsize = elempack == 1 ? 4u : elempack * 2u;
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

    A_data.release();
    B_data.release();
    C_data.release();

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
        if (support_image_storage && opt.use_image_storage)
        {
            cmd.record_upload(A_data_packed, A_data_gpu_image, opt);
        }
        else
        {
            cmd.record_upload(A_data_packed, A_data_gpu, opt);
        }

        A_data_packed.release();
    }

    if (constantB)
    {
        if (support_image_storage && opt.use_image_storage)
        {
            cmd.record_upload(B_data_packed, B_data_gpu_image, opt);
        }
        else
        {
            cmd.record_upload(B_data_packed, B_data_gpu, opt);
        }

        B_data_packed.release();
    }

    if (constantC)
    {
        if (support_image_storage && opt.use_image_storage)
        {
            cmd.record_upload(C_data_packed, C_data_gpu_image, opt);
        }
        else
        {
            cmd.record_upload(C_data_packed, C_data_gpu, opt);
        }

        C_data_packed.release();
    }

    return 0;
}

int Gemm_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& A0 = constantA ? A_data_gpu : bottom_blobs[0];
    const VkMat& B0 = constantB ? B_data_gpu : constantA ? bottom_blobs[0] : bottom_blobs[1];
    const VkMat& C0 = constantC ? C_data_gpu : bottom_blobs[bottom_blobs.size() - 1];

    VkMat A;
    VkMat B;
    VkMat C;
    vkdev->convert_packing(A0, A, 1, cmd, opt);
    vkdev->convert_packing(B0, B, 1, cmd, opt);
    vkdev->convert_packing(C0, C, 1, cmd, opt);

    const int M = constantM ? constantM : transA ? A.w : (A.dims == 3 ? A.c : A.h);
    const int K = constantK ? constantK : transA ? (A.dims == 3 ? A.c : A.h) : A.w;
    const int N = constantN ? constantN : transB ? (B.dims == 3 ? B.c : B.h) : B.w;

    int broadcast_type_C;
    if (constantC)
    {
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
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

    const Pipeline* pipeline = pipeline_gemm;

    VkMat dispatcher;
    dispatcher.w = (N + 1) / 2;
    dispatcher.h = (M + 1) / 2;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    int out_elempack = 1;
    {
        int outh = output_transpose ? N : M;
        out_elempack = opt.use_shader_pack8 && outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
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

int Gemm_vulkan::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkImageMat& A0 = constantA ? A_data_gpu_image : bottom_blobs[0];
    const VkImageMat& B0 = constantB ? B_data_gpu_image : constantA ? bottom_blobs[0] : bottom_blobs[1];
    const VkImageMat& C0 = constantC ? C_data_gpu_image : bottom_blobs[bottom_blobs.size() - 1];

    VkImageMat A;
    VkImageMat B;
    VkImageMat C;
    vkdev->convert_packing(A0, A, 1, cmd, opt);
    vkdev->convert_packing(B0, B, 1, cmd, opt);
    vkdev->convert_packing(C0, C, 1, cmd, opt);

    const int M = constantM ? constantM : transA ? A.w : (A.dims == 3 ? A.c : A.h);
    const int K = constantK ? constantK : transA ? (A.dims == 3 ? A.c : A.h) : A.w;
    const int N = constantN ? constantN : transB ? (B.dims == 3 ? B.c : B.h) : B.w;

    int broadcast_type_C;
    if (constantC)
    {
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
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

    int elempack = A.elempack;
    size_t elemsize = A.elemsize;

    VkImageMat& top_blob = top_blobs[0];
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

    std::vector<VkImageMat> bindings(4);
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
    constants[5].i = 0; //A.w;
    constants[6].i = B.dims;
    constants[7].i = 0; //B.w;
    constants[8].i = top_blob.dims;
    constants[9].i = 0; //top_blob.w;

    const Pipeline* pipeline = pipeline_gemm;

    VkImageMat dispatcher;
    dispatcher.w = (N + 1) / 2;
    dispatcher.h = (M + 1) / 2;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    int out_elempack = 1;
    {
        int outh = output_transpose ? N : M;
        out_elempack = opt.use_shader_pack8 && outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
    }
    if (output_elempack)
        out_elempack = output_elempack;

    if (out_elempack != 1)
    {
        VkImageMat top_blob0;
        vkdev->convert_packing(top_blob, top_blob0, out_elempack, cmd, opt);
        top_blobs[0] = top_blob0;
    }

    return 0;
}

int Gemm_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    std::vector<VkImageMat> bottom_blobs(1);
    std::vector<VkImageMat> top_blobs(1);
    bottom_blobs[0] = bottom_blob;
    int ret = forward(bottom_blobs, top_blobs, cmd, opt);
    top_blob = top_blobs[0];
    return ret;
}

} // namespace ncnn
