// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "matmul_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

MatMul_vulkan::MatMul_vulkan()
{
    support_vulkan = true;

    // The shader indexes elements one at a time, so it needs the plain
    // (unpacked) layout. support_vulkan_packing already defaults to false,
    // which keeps batch slicing a simple stride and avoids re-deriving
    // pack4/pack8 addressing for every rank.

    pipeline_matmul = 0;
}

int MatMul_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(1 + 3);
    specializations[0].i = transB;
    // Shapes are resolved per call, so leave the shape constants unspecialised.
    specializations[1 + 0].i = 0;
    specializations[1 + 1].i = 0;
    specializations[1 + 2].i = 0;

    pipeline_matmul = new Pipeline(vkdev);
    pipeline_matmul->set_optimal_local_size_xyz(8, 8, 1);
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
    const VkMat& A = bottom_blobs[0];
    const VkMat& B = bottom_blobs[1];

    const int Adims = A.dims;
    const int Bdims = B.dims;

    // Normalise both operands to (batch, rows, cols), mirroring MatMul::forward:
    //   1-D A is a row vector (1, K); 1-D B is a column vector (K, 1).
    const int M = Adims == 1 ? 1 : A.h;
    const int K = A.w;
    const int N = Bdims == 1 ? 1 : (transB == 0 ? B.w : B.h);

    // c and d broadcast independently, exactly as MatMul::forward does.
    //
    // Mind the promotion: when the ranks are mixed, MatMul::forward lifts a 3-D
    // operand with A.reshape(w, h, c, 1), which turns its CHANNEL axis into the
    // DEPTH axis. Depth then strides by the original cstep rather than by w*h.
    // Reading A.c as channels in that case rejects valid broadcasts.
    const int max_ABdims = std::max(Adims, Bdims);

    int a_c = 1, a_d = 1, a_cstep = 0, a_dstep = 0;
    int b_c = 1, b_d = 1, b_cstep = 0, b_dstep = 0;

    if (max_ABdims == 4 && Adims > 2)
    {
        if (Adims == 4)
        {
            a_c = A.c;
            a_d = std::max(A.d, 1);
            a_cstep = (int)A.cstep;
            a_dstep = A.w * A.h;
        }
        else // Adims == 3, promoted to (w, h, c, 1)
        {
            a_d = A.c;
            a_dstep = (int)A.cstep;
        }
    }
    else if (Adims > 2)
    {
        a_c = A.c;
        a_d = Adims <= 3 ? 1 : std::max(A.d, 1);
        a_cstep = (int)A.cstep;
        a_dstep = A.w * A.h;
    }

    if (max_ABdims == 4 && Bdims > 2)
    {
        if (Bdims == 4)
        {
            b_c = B.c;
            b_d = std::max(B.d, 1);
            b_cstep = (int)B.cstep;
            b_dstep = B.w * B.h;
        }
        else // Bdims == 3, promoted to (w, h, c, 1)
        {
            b_d = B.c;
            b_dstep = (int)B.cstep;
        }
    }
    else if (Bdims > 2)
    {
        b_c = B.c;
        b_d = Bdims <= 3 ? 1 : std::max(B.d, 1);
        b_cstep = (int)B.cstep;
        b_dstep = B.w * B.h;
    }

    // A 1-D operand pairs with a batched one without promotion: the output keeps
    // the batched operand's own d and c, so leave those descriptors as read.
    if (Adims == 1 || Bdims == 1)
    {
        if (Bdims > 2)
        {
            b_c = B.c;
            b_d = Bdims <= 3 ? 1 : std::max(B.d, 1);
            b_cstep = (int)B.cstep;
            b_dstep = B.w * B.h;
        }
        if (Adims > 2)
        {
            a_c = A.c;
            a_d = Adims <= 3 ? 1 : std::max(A.d, 1);
            a_cstep = (int)A.cstep;
            a_dstep = A.w * A.h;
        }
    }

    const int outc = std::max(a_c, b_c);
    const int outd = std::max(a_d, b_d);

    if ((a_c != b_c && a_c != 1 && b_c != 1) || (a_d != b_d && a_d != 1 && b_d != 1))
        return -1;

    const size_t elemsize = A.elemsize;

    VkMat& top_blob = top_blobs[0];

    // Output shape follows MatMul::forward exactly. A 1-D operand contributes no
    // dimension of its own, so the CPU path computes into a degenerate (N,1) or
    // (1,M) blob and reshapes the extra axis away. Create the final shape here
    // and describe it to the shader with explicit strides, because a reshaped
    // blob does not have the strides the general case would imply.
    int c_cstep = 0;
    int c_dstep = 0;

    if (Adims == 1 && Bdims == 1)
    {
        // dot product
        top_blob.create(1, elemsize, opt.blob_vkallocator);
    }
    else if (Adims == 1 && Bdims == 2)
    {
        top_blob.create(N, elemsize, opt.blob_vkallocator);
    }
    else if (Adims == 2 && Bdims == 1)
    {
        top_blob.create(M, elemsize, opt.blob_vkallocator);
    }
    else if (Adims == 1 && Bdims == 3)
    {
        top_blob.create(N, outc, elemsize, opt.blob_vkallocator);
        c_cstep = N;
    }
    else if (Adims == 1 && Bdims == 4)
    {
        top_blob.create(N, outd, outc, elemsize, opt.blob_vkallocator);
        c_cstep = (int)top_blob.cstep;
        c_dstep = N;
    }
    else if (Adims == 3 && Bdims == 1)
    {
        top_blob.create(M, outc, elemsize, opt.blob_vkallocator);
        c_cstep = M;
    }
    else if (Adims == 4 && Bdims == 1)
    {
        top_blob.create(M, outd, outc, elemsize, opt.blob_vkallocator);
        c_cstep = (int)top_blob.cstep;
        c_dstep = M;
    }
    else if (outc == 1 && outd == 1)
    {
        top_blob.create(N, M, elemsize, opt.blob_vkallocator);
    }
    else if (outd == 1)
    {
        top_blob.create(N, M, outc, elemsize, opt.blob_vkallocator);
        c_cstep = (int)top_blob.cstep;
        c_dstep = N * M;
    }
    else
    {
        top_blob.create(N, M, outd, outc, elemsize, opt.blob_vkallocator);
        c_cstep = (int)top_blob.cstep;
        c_dstep = N * M;
    }

    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = A;
    bindings[1] = B;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(15);
    constants[0].i = M;
    constants[1].i = N;
    constants[2].i = K;
    constants[3].i = outc;
    constants[4].i = outd;
    constants[5].i = a_cstep;
    constants[6].i = a_dstep;
    constants[7].i = a_c;
    constants[8].i = a_d;
    constants[9].i = b_cstep;
    constants[10].i = b_dstep;
    constants[11].i = b_c;
    constants[12].i = b_d;
    constants[13].i = c_cstep;
    constants[14].i = c_dstep;

    // Dispatch over the LOGICAL (N, M, outd, outc) grid rather than over
    // top_blob: when a degenerate axis has been reshaped away the blob no
    // longer carries M and outd separately, and the shader decodes depth
    // out of y as gy / M. A shape-only Mat supplies the group counts.
    Mat dispatcher;
    dispatcher.w = N;
    dispatcher.h = M;
    dispatcher.d = outd;
    dispatcher.c = outc;

    cmd.record_pipeline(pipeline_matmul, bindings, std::vector<VkImageMat>(), constants, dispatcher);

    return 0;
}

} // namespace ncnn
