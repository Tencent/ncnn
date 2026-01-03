// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "pooling_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

static inline void calc_same_pad(int w, int h, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_mode, int& pl, int& pr, int& pt, int& pb)
{
    int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
    int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
    if (wpad < 0) wpad = 0;
    if (hpad < 0) hpad = 0;

    if (pad_mode == 2)
    {
        pl = wpad / 2;
        pr = wpad - pl;
        pt = hpad / 2;
        pb = hpad - pt;
    }
    else
    {
        pl = wpad - wpad / 2;
        pr = wpad / 2;
        pt = hpad - hpad / 2;
        pb = hpad / 2;
    }
}

static inline void calc_output_and_pad(int w, int h,
                                       int kernel_w, int kernel_h,
                                       int stride_w, int stride_h,
                                       int pad_left, int pad_right, int pad_top, int pad_bottom,
                                       int pad_mode, int& outw, int& outh,
                                       int& pl, int& pr, int& pt, int& pb)
{
    pl = 0; pr = 0; pt = 0; pb = 0;

    if (pad_mode == 0 || pad_mode == 1)
    {
        pl = pad_left;
        pr = pad_right;
        pt = pad_top;
        pb = pad_bottom;

        if (pad_mode == 0)
        {
            int wtail = (w + pl + pr - kernel_w) % stride_w;
            int htail = (h + pt + pb - kernel_h) % stride_h;

            if (wtail != 0) pr += stride_w - wtail;
            if (htail != 0) pb += stride_h - htail;
        }
    }
    else
    {
        calc_same_pad(w, h, kernel_w, kernel_h, stride_w, stride_h, pad_mode, pl, pr, pt, pb);
    }

    outw = (w + pl + pr - kernel_w) / stride_w + 1;
    outh = (h + pt + pb - kernel_h) / stride_h + 1;
}

Pooling_vulkan::Pooling_vulkan()
{
    support_vulkan = true;

    support_vulkan_packing = false;
    support_vulkan_any_packing = false;

    pipeline_pooling = 0;
    pipeline_pooling_tile = 0;

    pipeline_pooling_global = 0;
    pipeline_pooling_global_stage1 = 0;
    pipeline_pooling_global_stage2 = 0;

    pipeline_pooling_adaptive = 0;
}

int Pooling_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    int out_elempack = 1;

    size_t elemsize = (opt.use_fp16_storage || opt.use_fp16_packed) ? 2u : 4u;
    size_t out_elemsize = elemsize;

    Mat shape_packed;
    Mat out_shape_packed;

    if (shape.dims == 1) shape_packed = Mat(shape.w, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c, (void*)0, elemsize, elempack);

    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c, (void*)0, out_elemsize, out_elempack);

    if (global_pooling)
    {
        {
            std::vector<vk_specialization_type> specializations(1);
            specializations[0].i = pooling_type;

            pipeline_pooling_global = new Pipeline(vkdev);
            pipeline_pooling_global->set_local_size_xyz(256, 1, 1);
            pipeline_pooling_global->create(LayerShaderType::pooling_global, opt, specializations);
        }

        {
            std::vector<vk_specialization_type> specializations(1);
            specializations[0].i = pooling_type;

            pipeline_pooling_global_stage1 = new Pipeline(vkdev);
            pipeline_pooling_global_stage1->set_local_size_xyz(256, 1, 1);
            pipeline_pooling_global_stage1->create(LayerShaderType::pooling_global_stage1, opt, specializations);
        }

        {
            std::vector<vk_specialization_type> specializations(1);
            specializations[0].i = pooling_type;

            pipeline_pooling_global_stage2 = new Pipeline(vkdev);
            pipeline_pooling_global_stage2->set_local_size_xyz(256, 1, 1);
            pipeline_pooling_global_stage2->create(LayerShaderType::pooling_global_stage2, opt, specializations);
        }

        return 0;
    }

    if (adaptive_pooling)
    {
        std::vector<vk_specialization_type> specializations(5);
        specializations[0].i = pooling_type;
        specializations[1].i = out_w;
        specializations[2].i = out_h;
        specializations[3].i = 0;
        specializations[4].i = 0;

        pipeline_pooling_adaptive = new Pipeline(vkdev);
        pipeline_pooling_adaptive->set_local_size_xyz(8, 8, 1);
        pipeline_pooling_adaptive->create(LayerShaderType::pooling_adaptive, opt, specializations);
        return 0;
    }

    bool use_tile = true;
    {
        const int tile_out_w = 8;
        const int tile_out_h = 8;
        const int tile_in_w = (tile_out_w - 1) * stride_w + kernel_w;
        const int tile_in_h = (tile_out_h - 1) * stride_h + kernel_h;

        if (tile_in_w > 36 || tile_in_h > 36) use_tile = false;
        if (kernel_w <= 0 || kernel_h <= 0) use_tile = false;
        if (stride_w <= 0 || stride_h <= 0) use_tile = false;
    }

    std::vector<vk_specialization_type> specializations(11 + 12);
    specializations[0].i = pooling_type;
    specializations[1].i = kernel_w;
    specializations[2].i = kernel_h;
    specializations[3].i = stride_w;
    specializations[4].i = stride_h;
    specializations[5].i = pad_left;
    specializations[6].i = pad_right;
    specializations[7].i = pad_top;
    specializations[8].i = pad_bottom;
    specializations[9].i = pad_mode;
    specializations[10].i = avgpool_count_include_pad;

    specializations[11 + 0].i = shape_packed.dims;
    specializations[11 + 1].i = shape_packed.w;
    specializations[11 + 2].i = shape_packed.h;
    specializations[11 + 3].i = shape_packed.d;
    specializations[11 + 4].i = shape_packed.c;
    specializations[11 + 5].i = shape_packed.cstep;

    specializations[11 + 6].i = out_shape_packed.dims;
    specializations[11 + 7].i = out_shape_packed.w;
    specializations[11 + 8].i = out_shape_packed.h;
    specializations[11 + 9].i = out_shape_packed.d;
    specializations[11 + 10].i = out_shape_packed.c;
    specializations[11 + 11].i = out_shape_packed.cstep;

    if (use_tile)
    {
        pipeline_pooling_tile = new Pipeline(vkdev);
        pipeline_pooling_tile->set_local_size_xyz(8, 8, 1);
        pipeline_pooling_tile->create(LayerShaderType::pooling_tile, opt, specializations);
    }
    else
    {
        pipeline_pooling = new Pipeline(vkdev);
        pipeline_pooling->set_local_size_xyz(8, 8, 1);
        pipeline_pooling->create(LayerShaderType::pooling, opt, specializations);
    }

    return 0;
}

int Pooling_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_pooling;
    pipeline_pooling = 0;

    delete pipeline_pooling_tile;
    pipeline_pooling_tile = 0;

    delete pipeline_pooling_global;
    pipeline_pooling_global = 0;

    delete pipeline_pooling_global_stage1;
    pipeline_pooling_global_stage1 = 0;

    delete pipeline_pooling_global_stage2;
    pipeline_pooling_global_stage2 = 0;

    delete pipeline_pooling_adaptive;
    pipeline_pooling_adaptive = 0;

    return 0;
}

int Pooling_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (dims != 2 && dims != 3)
        return -100;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = (dims == 3) ? bottom_blob.c : 1;
    const size_t elemsize = bottom_blob.elemsize;

    if (global_pooling)
    {
        top_blob.create(channels, elemsize, 1, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        const int size = w * h;

        const bool use_two_stage = (channels < 8 && size >= 4096);

        if (!use_two_stage)
        {
            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(4);
            constants[0].i = w;
            constants[1].i = h;
            constants[2].i = channels;
            constants[3].i = bottom_blob.cstep;

            VkMat dispatcher;
            dispatcher.w = channels * 256;
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_pooling_global, bindings, constants, dispatcher);
            return 0;
        }

        const int wg = 256;
        const int unroll = 4;
        const int chunk = wg * unroll;
        const int partial_w = (size + chunk - 1) / chunk;

        VkMat partial;
        partial.create(partial_w, channels, elemsize, 1, opt.workspace_vkallocator);
        if (partial.empty())
            return -100;

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = partial;

            std::vector<vk_constant_type> constants(5);
            constants[0].i = w;
            constants[1].i = h;
            constants[2].i = channels;
            constants[3].i = bottom_blob.cstep;
            constants[4].i = partial_w;

            VkMat dispatcher;
            dispatcher.w = partial_w * 256;
            dispatcher.h = channels;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_pooling_global_stage1, bindings, constants, dispatcher);
        }

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = partial;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = partial_w;
            constants[1].i = channels;
            constants[2].i = size;

            VkMat dispatcher;
            dispatcher.w = channels * 256;
            dispatcher.h = 1;
            dispatcher.c = 1;

            cmd.record_pipeline(pipeline_pooling_global_stage2, bindings, constants, dispatcher);
        }

        return 0;
    }

    if (adaptive_pooling)
    {
        int outw = out_w == -233 ? w : out_w;
        int outh = out_h == -233 ? h : out_h;

        if (outw == w && outh == h)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, outh, channels, elemsize, 1, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.d;
        constants[4].i = (dims == 3) ? bottom_blob.c : 1;
        constants[5].i = bottom_blob.cstep;
        constants[6].i = top_blob.dims;
        constants[7].i = top_blob.w;
        constants[8].i = top_blob.h;
        constants[9].i = top_blob.d;
        constants[10].i = (dims == 3) ? top_blob.c : 1;
        constants[11].i = top_blob.cstep;

        cmd.record_pipeline(pipeline_pooling_adaptive, bindings, constants, top_blob);
        return 0;
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0 && pad_mode == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int outw, outh;
    int pl, pr, pt, pb;
    calc_output_and_pad(w, h, kernel_w, kernel_h, stride_w, stride_h, pad_left, pad_right, pad_top, pad_bottom, pad_mode, outw, outh, pl, pr, pt, pb);

    if (dims == 2)
    {
        top_blob.create(outw, outh, elemsize, 1, opt.blob_vkallocator);
    }
    else
    {
        top_blob.create(outw, outh, channels, elemsize, 1, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.d;
    constants[4].i = (dims == 3) ? bottom_blob.c : 1;
    constants[5].i = bottom_blob.cstep;
    constants[6].i = top_blob.dims;
    constants[7].i = top_blob.w;
    constants[8].i = top_blob.h;
    constants[9].i = top_blob.d;
    constants[10].i = (dims == 3) ? top_blob.c : 1;
    constants[11].i = top_blob.cstep;

    const Pipeline* pipeline = pipeline_pooling_tile ? pipeline_pooling_tile : pipeline_pooling;
    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
