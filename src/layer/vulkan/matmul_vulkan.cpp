// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "matmul_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

MatMul_vulkan::MatMul_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;

    pipeline_matmul = 0;
}

int MatMul_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = transB;

    Mat local_size_xyz;

    pipeline_matmul = new Pipeline(vkdev);
    pipeline_matmul->set_optimal_local_size_xyz(local_size_xyz);

    if (opt.use_shader_local_memory)
    {
        pipeline_matmul->set_local_size_xyz(8, 8, 1);
    }

    pipeline_matmul->create(LayerShaderType::matmul, opt, specializations);

    return 0;
}

int MatMul_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_matmul;
    pipeline_matmul = 0;

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
    const int max_ABdims = std::max(Adims, Bdims);
    const size_t elemsize = A.elemsize;

    int mode = 0;

    int M = 1;
    int N = 1;
    int K = 1;

    int a_w = 1, a_h = 1, a_d = 1, a_c = 1;
    int b_w = 1, b_h = 1, b_d = 1, b_c = 1;

    int out_dims = 0;
    int out_w = 1, out_h = 1, out_d = 1, out_c = 1;

    if (Adims == 1 && Bdims == 1)
    {
        mode = 0;
        K = A.w;
        M = 1;
        N = 1;

        a_w = K;
        a_h = 1;
        a_d = 1;
        a_c = 1;

        if (transB == 0)
        {
            b_w = 1;
            b_h = K;
        }
        else
        {
            b_w = K;
            b_h = 1;
        }
        b_d = 1;
        b_c = 1;

        out_dims = 1;
        out_w = 1;
        out_h = 1;
    }
    else if (Adims == 2 && Bdims == 2)
    {
        mode = 0;
        M = A.h;
        K = A.w;
        N = transB == 0 ? B.w : B.h;

        a_w = A.w;
        a_h = A.h;
        a_d = 1;
        a_c = 1;

        b_w = B.w;
        b_h = B.h;
        b_d = 1;
        b_c = 1;

        out_dims = 2;
        out_w = N;
        out_h = M;
    }
    else if (Adims == 1 && Bdims == 2)
    {
        mode = 1;
        K = A.w;
        M = 1;
        N = transB == 0 ? B.w : B.h;

        a_w = K;
        a_h = 1;
        a_d = 1;
        a_c = 1;

        b_w = B.w;
        b_h = B.h;
        b_d = 1;
        b_c = 1;

        out_dims = 1;
        out_w = N;
        out_h = 1;
    }
    else if (Adims == 2 && Bdims == 1)
    {
        mode = 2;
        M = A.h;
        K = A.w;
        N = 1;

        a_w = A.w;
        a_h = A.h;
        a_d = 1;
        a_c = 1;

        if (transB == 0)
        {
            b_w = 1;
            b_h = K;
        }
        else
        {
            b_w = K;
            b_h = 1;
        }

        b_d = 1;
        b_c = 1;

        out_dims = 1;
        out_w = M;
        out_h = 1;
    }
    else if (Adims == 1 && Bdims > 2)
    {
        // Vector @ batched-matrix -> reduce one dimension.
        mode = 1;
        K = A.w;
        N = transB == 0 ? B.w : B.h;

        a_w = K;
        a_h = 1;
        a_d = 1;
        a_c = 1;

        b_w = B.w;
        b_h = B.h;

        b_d = B.dims == 4 ? B.d : 1;
        b_c = B.dims >= 3 ? B.c : 1;

        if (Bdims == 3)
        {
            out_dims = 2;
            out_w = N;
            out_h = B.d * B.c;
            out_d = 1;
            out_c = 1;
        }
        else
        {
            out_dims = 3;
            out_w = N;
            out_h = B.d;
            out_c = B.c;
            out_d = 1;
        }
    }
    else if (Adims > 2 && Bdims == 1)
    {
        // Batched-matrix @ vector -> reduce one dimension.
        mode = 2;
        M = A.h;
        K = A.w;
        N = 1;

        a_w = A.w;
        a_h = A.h;
        a_d = A.dims == 4 ? A.d : 1;
        a_c = A.dims >= 3 ? A.c : 1;

        if (transB == 0)
        {
            b_w = 1;
            b_h = K;
        }
        else
        {
            b_w = K;
            b_h = 1;
        }
        b_d = 1;
        b_c = 1;

        if (Adims == 3)
        {
            out_dims = 2;
            out_w = M;
            out_h = A.d * A.c;
            out_d = 1;
            out_c = 1;
        }
        else
        {
            out_dims = 3;
            out_w = M;
            out_h = A.d;
            out_c = A.c;
            out_d = 1;
        }
    }
    else
    {
        // Batched matmul follows CPU reshape/broadcast rules.
        mode = 0;
        M = A.h;
        K = A.w;
        N = transB == 0 ? B.w : B.h;

        a_w = A.w;
        a_h = A.h;
        b_w = B.w;
        b_h = B.h;

        if (max_ABdims == 3)
        {
            a_d = 1;
            b_d = 1;
            a_c = (Adims == 3) ? A.c : 1;
            b_c = (Bdims == 3) ? B.c : 1;

            out_dims = 3;
            out_w = N;
            out_h = M;
            out_c = std::max(a_c, b_c);
            out_d = 1;
        }
        else
        {
            // dims3 -> reshape(w,h,d=orig_c,c=1), dims4 stays (w,h,d,c)
            if (Adims == 4)
            {
                a_d = A.d;
                a_c = A.c;
            }
            else if (Adims == 3) {
                a_d = A.c;
                a_c = 1;
            }
            else
            {
                a_d = 1;
                a_c = 1;
            }

            if (Bdims == 4)
            {
                b_d = B.d;
                b_c = B.c;
            }
            else if (Bdims == 3)
            {
                b_d = B.c;
                b_c = 1;
            }
            else {
                b_d = 1;
                b_c = 1;
            }

            out_dims = 4;
            out_w = N;
            out_h = M;
            out_d = std::max(a_d, b_d);
            out_c = std::max(a_c, b_c);
        }
    }

    VkMat& top_blob = top_blobs[0];

    if (out_dims == 1)
        top_blob.create(out_w, elemsize, opt.blob_vkallocator);
    else if (out_dims == 2)
        top_blob.create(out_w, out_h, elemsize, opt.blob_vkallocator);
    else if (out_dims == 3)
        top_blob.create(out_w, out_h, out_c, elemsize, opt.blob_vkallocator);
    else
        top_blob.create(out_w, out_h, out_d, out_c, elemsize, opt.blob_vkallocator);

    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = top_blob;
    bindings[1] = A;
    bindings[2] = B;

    const int out_cstep = (top_blob.dims >= 3) ? (int)top_blob.cstep : (top_blob.w * top_blob.h);
    const int a_cstep_real = (A.dims >= 3) ? (int)A.cstep : (A.w * A.h);
    const int b_cstep_real = (B.dims >= 3) ? (int)B.cstep : (B.w * B.h);

    const int out_dstep = out_w * out_h;

    int a_dstep_real = a_w * a_h;
    int b_dstep_real = b_w * b_h;

    if (max_ABdims == 4)
    {
        if (A.dims == 3) a_dstep_real = (int)A.cstep;
        if (B.dims == 3) b_dstep_real = (int)B.cstep;
    }

    std::vector<vk_constant_type> constants(23);
    constants[0].i = M;
    constants[1].i = N;
    constants[2].i = K;
    constants[3].i = mode;

    constants[4].i = out_dims;
    constants[5].i = out_w;
    constants[6].i = out_h;
    constants[7].i = out_d;
    constants[8].i = out_c;
    constants[9].i = out_cstep;
    constants[10].i = out_dstep;

    constants[11].i = a_w;
    constants[12].i = a_h;
    constants[13].i = a_d;
    constants[14].i = a_c;
    constants[15].i = a_cstep_real;
    constants[16].i = a_dstep_real;

    constants[17].i = b_w;
    constants[18].i = b_h;
    constants[19].i = b_d;
    constants[20].i = b_c;
    constants[21].i = b_cstep_real;
    constants[22].i = b_dstep_real;

    const Pipeline* pipeline = pipeline_matmul;

    VkMat dispatcher;

    if (mode == 0 && out_dims >= 2)
    {
        dispatcher.w = (out_w + 1) / 2;
        dispatcher.h = (out_h + 1) / 2;
    }
    else
    {
        dispatcher.w = out_w;
        dispatcher.h = out_h;
    }

    if (out_dims == 4)
        dispatcher.c = out_c * out_d;
    else if (out_dims == 3)
        dispatcher.c = out_c;
    else
        dispatcher.c = 1;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
