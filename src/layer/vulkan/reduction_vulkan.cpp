// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "reduction_vulkan.h"

#include <vector>

#include "layer_shader_type.h"

namespace ncnn {

Reduction_vulkan::Reduction_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;

    pipeline_reduction = 0;
}

int Reduction_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = operation;

    pipeline_reduction = new Pipeline(vkdev);
    pipeline_reduction->set_local_size_xyz(256, 1, 1);
    pipeline_reduction->create(LayerShaderType::reduction, opt, specializations);

    return 0;
}

int Reduction_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_reduction;
    pipeline_reduction = 0;

    return 0;
}

int Reduction_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    bool reduce_w, reduce_h, reduce_d, reduce_c;
    int outdims, outw, outh, outd, outc;
    resolve_reduce_flags_and_output_shape(bottom_blob.shape(), reduce_w, reduce_h, reduce_d, reduce_c, outdims, outw, outh, outd, outc);

    const int dims = bottom_blob.dims;
    const size_t elemsize = bottom_blob.elemsize;

    if (outdims == 0)
    {
        top_blob.create(1, elemsize, opt.blob_vkallocator);
    }
    if (outdims == 1)
    {
        top_blob.create(outw, elemsize, opt.blob_vkallocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, elemsize, opt.blob_vkallocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, elemsize, opt.blob_vkallocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, elemsize, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    // resolve_output_mapping
    int map_out_w, map_out_h, map_out_d, map_out_c;
    {
        if (keepdims)
        {
            map_out_w = 3;
            map_out_h = (dims >= 2) ? 2 : -1;
            map_out_d = (dims >= 4) ? 1 : -1;
            map_out_c = (dims >= 3) ? 0 : -1;
        }
        else
        {
            map_out_w = -1;
            map_out_h = -1;
            map_out_d = -1;
            map_out_c = -1;

            // Collect surviving input axes in order: c(0), d(1), h(2), w(3)
            // Only include axes that exist in the input and are not reduced
            int surviving[4];
            int num_surviving = 0;

            if (dims >= 3 && !reduce_c) surviving[num_surviving++] = 0;
            if (dims >= 4 && !reduce_d) surviving[num_surviving++] = 1;
            if (dims >= 2 && !reduce_h) surviving[num_surviving++] = 2;
            if (!reduce_w) surviving[num_surviving++] = 3;

            // Map output dims to surviving input axes
            // outdims convention: 1D=w, 2D=w,h, 3D=w,h,c, 4D=w,h,d,c
            // Assign surviving axes (ordered c,d,h,w) to output axes (ordered c,d,h,w)
            // The highest surviving axis maps to out_w, next to out_h, etc.
            // i.e. surviving axes are assigned to output slots from w upward

            if (num_surviving >= 1)
            {
                map_out_w = surviving[num_surviving - 1];
            }
            if (num_surviving >= 2)
            {
                map_out_h = surviving[num_surviving - 2];
            }
            if (num_surviving >= 3)
            {
                if (outdims >= 4)
                    map_out_d = surviving[num_surviving - 3];
                else
                    map_out_c = surviving[num_surviving - 3];
            }
            if (num_surviving >= 4)
            {
                map_out_c = surviving[num_surviving - 4];
            }
        }
    }

    float coeff2 = coeff;
    if (operation == ReductionOp_MEAN)
    {
        int scale = 1;
        if (dims == 1)
        {
            scale = bottom_blob.w;
        }
        if (dims == 2)
        {
            if (reduce_w) scale *= bottom_blob.w;
            if (reduce_h) scale *= bottom_blob.h;
        }
        if (dims == 3)
        {
            if (reduce_w) scale *= bottom_blob.w;
            if (reduce_h) scale *= bottom_blob.h;
            if (reduce_c) scale *= bottom_blob.c;
        }
        if (dims == 4)
        {
            if (reduce_w) scale *= bottom_blob.w;
            if (reduce_h) scale *= bottom_blob.h;
            if (reduce_d) scale *= bottom_blob.d;
            if (reduce_c) scale *= bottom_blob.c;
        }

        coeff2 = coeff / scale;
    }

    std::vector<VkMat> bindings(2);
    bindings[0] = top_blob;
    bindings[1] = bottom_blob;

    std::vector<vk_constant_type> constants(21);
    constants[0].i = bottom_blob.w;
    constants[1].i = bottom_blob.h;
    constants[2].i = bottom_blob.d;
    constants[3].i = bottom_blob.c;
    constants[4].i = (int)bottom_blob.cstep;
    constants[5].i = bottom_blob.dims;

    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.d;
    constants[9].i = top_blob.c;
    constants[10].i = top_blob.dims;
    constants[11].i = (int)top_blob.cstep;

    constants[12].i = reduce_w ? 1 : 0;
    constants[13].i = reduce_h ? 1 : 0;
    constants[14].i = reduce_d ? 1 : 0;
    constants[15].i = reduce_c ? 1 : 0;

    constants[16].i = map_out_w;
    constants[17].i = map_out_h;
    constants[18].i = map_out_d;
    constants[19].i = map_out_c;

    constants[20].f = coeff2;

    int out_total = outw * outh * outd * outc;

    VkMat dispatcher;
    dispatcher.w = 256;
    dispatcher.h = out_total;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_reduction, bindings, constants, dispatcher);
    return 0;
}

} // namespace ncnn
