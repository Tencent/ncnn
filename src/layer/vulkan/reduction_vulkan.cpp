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

static inline int axis_size_from_vkmat(int axis, int dims, const VkMat& m)
{
    if (dims == 1)
        return axis == 3 ? m.w : 1;

    if (dims == 2)
    {
        if (axis == 2) return m.h;
        if (axis == 3) return m.w;
        return 1;
    }

    if (dims == 3)
    {
        if (axis == 0) return m.c;
        if (axis == 2) return m.h;
        if (axis == 3) return m.w;
        return 1;
    }

    if (axis == 0) return m.c;
    if (axis == 1) return m.d;
    if (axis == 2) return m.h;
    if (axis == 3) return m.w;
    return 1;
}

static inline void resolve_reduce_flags(int dims, int reduce_all, const Mat& axes,
                                        bool& reduce_w, bool& reduce_h, bool& reduce_d, bool& reduce_c)
{
    reduce_w = false;
    reduce_h = false;
    reduce_d = false;
    reduce_c = false;

    if (reduce_all)
    {
        reduce_w = true;
        reduce_h = true;
        reduce_d = true;
        reduce_c = true;
        return;
    }

    int axes_flag[4] = {0, 0, 0, 0};
    const int* axes_ptr = axes;
    const int axes_count = axes.w;

    for (int i = 0; i < axes_count; i++)
    {
        int axis = axes_ptr[i];
        if (axis < 0) axis += dims;
        if (axis >= 0 && axis < 4) axes_flag[axis] = 1;
    }

    if (dims == 1)
    {
        reduce_w = true;
    }
    else if (dims == 2)
    {
        if (axes_flag[0]) reduce_h = true;
        if (axes_flag[1]) reduce_w = true;
    }
    else if (dims == 3)
    {
        if (axes_flag[0]) reduce_c = true;
        if (axes_flag[1]) reduce_h = true;
        if (axes_flag[2]) reduce_w = true;
    }
    else
    {
        if (axes_flag[0]) reduce_c = true;
        if (axes_flag[1]) reduce_d = true;
        if (axes_flag[2]) reduce_h = true;
        if (axes_flag[3]) reduce_w = true;
    }
}

static inline void resolve_output_shape_and_mapping(const VkMat& a,
        bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c,
        int keepdims,
        int& outdims, int& out_w, int& out_h, int& out_d, int& out_c,
        int& map_out_w, int& map_out_h, int& map_out_d, int& map_out_c)
{
    const int dims = a.dims;

    outdims = 1;
    out_w = 1;
    out_h = 1;
    out_d = 1;
    out_c = 1;

    map_out_w = -1;
    map_out_h = -1;
    map_out_d = -1;
    map_out_c = -1;

    auto is_reduced_axis = [&](int axis) -> bool {
        if (axis == 0) return reduce_c;
        if (axis == 1) return reduce_d;
        if (axis == 2) return reduce_h;
        if (axis == 3) return reduce_w;
        return false;
    };

    int in_axes[4];
    int in_axes_count = 0;
    if (dims == 1)
    {
        in_axes[0] = 3;
        in_axes_count = 1;
    }
    else if (dims == 2)
    {
        in_axes[0] = 2;
        in_axes[1] = 3;
        in_axes_count = 2;
    }
    else if (dims == 3)
    {
        in_axes[0] = 0;
        in_axes[1] = 2;
        in_axes[2] = 3;
        in_axes_count = 3;
    }
    else
    {
        in_axes[0] = 0;
        in_axes[1] = 1;
        in_axes[2] = 2;
        in_axes[3] = 3;
        in_axes_count = 4;
    }

    if (keepdims)
    {
        outdims = dims;

        if (dims == 1)
        {
            out_w = reduce_w ? 1 : a.w;
            map_out_w = 3;
        }
        else if (dims == 2)
        {
            out_h = reduce_h ? 1 : a.h;
            out_w = reduce_w ? 1 : a.w;
            map_out_h = 2;
            map_out_w = 3;
        }
        else if (dims == 3)
        {
            out_c = reduce_c ? 1 : a.c;
            out_h = reduce_h ? 1 : a.h;
            out_w = reduce_w ? 1 : a.w;
            map_out_c = 0;
            map_out_h = 2;
            map_out_w = 3;
        }
        else
        {
            out_c = reduce_c ? 1 : a.c;
            out_d = reduce_d ? 1 : a.d;
            out_h = reduce_h ? 1 : a.h;
            out_w = reduce_w ? 1 : a.w;
            map_out_c = 0;
            map_out_d = 1;
            map_out_h = 2;
            map_out_w = 3;
        }

        return;
    }

    int keep_axes[4];
    int keep_count = 0;
    for (int i = 0; i < in_axes_count; i++)
    {
        if (!is_reduced_axis(in_axes[i]))
            keep_axes[keep_count++] = in_axes[i];
    }

    if (keep_count == 0)
    {
        outdims = 1;
        out_w = 1;
        return;
    }

    outdims = keep_count;

    if (outdims == 1)
    {
        map_out_w = keep_axes[0];
        out_w = axis_size_from_vkmat(map_out_w, dims, a);
    }
    else if (outdims == 2)
    {
        map_out_h = keep_axes[0];
        map_out_w = keep_axes[1];
        out_h = axis_size_from_vkmat(map_out_h, dims, a);
        out_w = axis_size_from_vkmat(map_out_w, dims, a);
    }
    else if (outdims == 3)
    {
        map_out_c = keep_axes[0];
        map_out_h = keep_axes[1];
        map_out_w = keep_axes[2];
        out_c = axis_size_from_vkmat(map_out_c, dims, a);
        out_h = axis_size_from_vkmat(map_out_h, dims, a);
        out_w = axis_size_from_vkmat(map_out_w, dims, a);
    }
    else
    {
        map_out_c = keep_axes[0];
        map_out_d = keep_axes[1];
        map_out_h = keep_axes[2];
        map_out_w = keep_axes[3];
        out_c = axis_size_from_vkmat(map_out_c, dims, a);
        out_d = axis_size_from_vkmat(map_out_d, dims, a);
        out_h = axis_size_from_vkmat(map_out_h, dims, a);
        out_w = axis_size_from_vkmat(map_out_w, dims, a);
    }
}

static inline float compute_coeff2_for_mean(const VkMat& a,
        bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c,
        float coeff)
{
    int scale = 1;
    const int dims = a.dims;

    if (dims == 1)
    {
        scale = a.w;
    }
    else if (dims == 2)
    {
        if (reduce_w) scale *= a.w;
        if (reduce_h) scale *= a.h;
    }
    else if (dims == 3)
    {
        if (reduce_w) scale *= a.w;
        if (reduce_h) scale *= a.h;
        if (reduce_c) scale *= a.c;
    }
    else
    {
        if (reduce_w) scale *= a.w;
        if (reduce_h) scale *= a.h;
        if (reduce_d) scale *= a.d;
        if (reduce_c) scale *= a.c;
    }

    return coeff / scale;
}

int Reduction_vulkan::create_pipeline(const Option& opt)
{
    pipeline_reduction = new Pipeline(vkdev);
    pipeline_reduction->set_local_size_xyz(256, 1, 1);

    std::vector<vk_specialization_type> specializations(9);
    specializations[0].i = operation;
    specializations[1].i = reduce_all ? 1 : 0;
    specializations[2].i = keepdims ? 1 : 0;
    specializations[3].f = coeff;

    int axes_count = axes.w;
    if (axes_count < 0) axes_count = 0;
    if (axes_count > 4) axes_count = 4;
    specializations[4].i = axes_count;

    int ax0 = 0, ax1 = 0, ax2 = 0, ax3 = 0;
    if (axes_count > 0) ax0 = ((const int*)axes)[0];
    if (axes_count > 1) ax1 = ((const int*)axes)[1];
    if (axes_count > 2) ax2 = ((const int*)axes)[2];
    if (axes_count > 3) ax3 = ((const int*)axes)[3];

    specializations[5].i = ax0;
    specializations[6].i = ax1;
    specializations[7].i = ax2;
    specializations[8].i = ax3;

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
    VkMat a = bottom_blob;

    if (a.empty())
        return -100;

    bool reduce_w, reduce_h, reduce_d, reduce_c;
    resolve_reduce_flags(a.dims, reduce_all, axes, reduce_w, reduce_h, reduce_d, reduce_c);

    int outdims, out_w, out_h, out_d, out_c;
    int map_out_w, map_out_h, map_out_d, map_out_c;
    resolve_output_shape_and_mapping(a, reduce_w, reduce_h, reduce_d, reduce_c, keepdims,
                                     outdims, out_w, out_h, out_d, out_c,
                                     map_out_w, map_out_h, map_out_d, map_out_c);

    const size_t elemsize = a.elemsize;

    if (outdims == 1)
        top_blob.create(out_w, elemsize, opt.blob_vkallocator);
    else if (outdims == 2)
        top_blob.create(out_w, out_h, elemsize, opt.blob_vkallocator);
    else if (outdims == 3)
        top_blob.create(out_w, out_h, out_c, elemsize, opt.blob_vkallocator);
    else
        top_blob.create(out_w, out_h, out_d, out_c, elemsize, opt.blob_vkallocator);

    if (top_blob.empty())
        return -100;

    float coeff2 = coeff;
    if (operation == ReductionOp_MEAN)
        coeff2 = compute_coeff2_for_mean(a, reduce_w, reduce_h, reduce_d, reduce_c, coeff);

    std::vector<VkMat> bindings(2);
    bindings[0] = top_blob;
    bindings[1] = a;

    std::vector<vk_constant_type> constants(21);
    constants[0].i = a.w;
    constants[1].i = a.h;
    constants[2].i = a.d;
    constants[3].i = a.c;
    constants[4].i = (int)a.cstep;
    constants[5].i = a.dims;

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

    VkMat dispatcher;
    dispatcher.w = 256;
    dispatcher.h = (int)top_blob.total();
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_reduction, bindings, constants, dispatcher);
    return 0;
}

} // namespace ncnn
