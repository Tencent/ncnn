// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "lstm_vulkan.h"

#include <algorithm>

#include "layer_shader_type.h"

namespace ncnn {

LSTM_vulkan::LSTM_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;

    pipeline_lstm_copy = 0;
    pipeline_lstm_step = 0;
    pipeline_lstm_step_h = 0;
    pipeline_lstm_proj = 0;
}

int LSTM_vulkan::load_param(const ParamDict& pd)
{
    int ret = LSTM::load_param(pd);

    if (int8_scale_term)
    {
        support_vulkan = false;
    }

    return ret;
}

int LSTM_vulkan::create_pipeline(const Option& opt)
{
    if (!support_vulkan)
        return 0;

    {
        pipeline_lstm_copy = new Pipeline(vkdev);
        pipeline_lstm_copy->set_local_size_xyz(64, 1, 1);

        std::vector<vk_specialization_type> specializations;
        pipeline_lstm_copy->create(LayerShaderType::lstm_copy, opt, specializations);
    }

    {
        pipeline_lstm_step = new Pipeline(vkdev);
        pipeline_lstm_step->set_local_size_xyz(64, 1, 1);

        std::vector<vk_specialization_type> specializations;
        pipeline_lstm_step->create(LayerShaderType::lstm_step, opt, specializations);
    }

    if (num_output != hidden_size)
    {
        pipeline_lstm_step_h = new Pipeline(vkdev);
        pipeline_lstm_step_h->set_local_size_xyz(64, 1, 1);

        std::vector<vk_specialization_type> specializations_h;
        pipeline_lstm_step_h->create(LayerShaderType::lstm_step_h, opt, specializations_h);

        pipeline_lstm_proj = new Pipeline(vkdev);
        pipeline_lstm_proj->set_local_size_xyz(64, 1, 1);

        std::vector<vk_specialization_type> specializations_p;
        pipeline_lstm_proj->create(LayerShaderType::lstm_proj, opt, specializations_p);
    }

    return 0;
}

int LSTM_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_lstm_copy;
    pipeline_lstm_copy = 0;

    delete pipeline_lstm_step;
    pipeline_lstm_step = 0;

    delete pipeline_lstm_step_h;
    pipeline_lstm_step_h = 0;

    delete pipeline_lstm_proj;
    pipeline_lstm_proj = 0;

    return 0;
}

int LSTM_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (!support_vulkan)
        return 0;

    cmd.record_upload(weight_xc_data, weight_xc_data_gpu, opt);
    cmd.record_upload(bias_c_data, bias_c_data_gpu, opt);
    cmd.record_upload(weight_hc_data, weight_hc_data_gpu, opt);

    if (num_output != hidden_size)
    {
        cmd.record_upload(weight_hr_data, weight_hr_data_gpu, opt);
    }

    if (opt.lightmode)
    {
        weight_xc_data.release();
        bias_c_data.release();
        weight_hc_data.release();
        weight_hr_data.release();
    }

    return 0;
}

static inline void record_lstm_copy(const Pipeline* pipeline,
                                    VkCompute& cmd,
                                    const VkMat& src,
                                    VkMat& dst,
                                    int len,
                                    int src_offset,
                                    int dst_offset,
                                    int mode)
{
    std::vector<VkMat> bindings(2);
    bindings[0] = src;
    bindings[1] = dst;

    std::vector<vk_constant_type> constants(4);
    constants[0].i = len;
    constants[1].i = src_offset;
    constants[2].i = dst_offset;
    constants[3].i = mode;

    VkMat dispatcher;
    dispatcher.w = len;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
}

static inline void record_lstm_step(const Pipeline* pipeline,
                                    VkCompute& cmd,
                                    const VkMat& bottom_blob,
                                    const VkMat& weight_xc,
                                    const VkMat& bias_c,
                                    const VkMat& weight_hc,
                                    const VkMat& hidden_prev,
                                    const VkMat& cell_prev,
                                    VkMat& hidden_next,
                                    VkMat& cell_next,
                                    VkMat& top_blob,
                                    int size,
                                    int num_output,
                                    int hidden_size,
                                    int ti,
                                    int outw,
                                    int out_offset,
                                    int dir,
                                    int wxc_dir_stride,
                                    int whc_dir_stride,
                                    int bias_dir_stride,
                                    int bottom_step)
{
    std::vector<VkMat> bindings(9);
    bindings[0] = bottom_blob;
    bindings[1] = weight_xc;
    bindings[2] = bias_c;
    bindings[3] = weight_hc;
    bindings[4] = hidden_prev;
    bindings[5] = cell_prev;
    bindings[6] = hidden_next;
    bindings[7] = cell_next;
    bindings[8] = top_blob;

    std::vector<vk_constant_type> constants(12);
    constants[0].i = size;
    constants[1].i = num_output;
    constants[2].i = hidden_size;
    constants[3].i = ti;
    constants[4].i = outw;
    constants[5].i = out_offset;
    constants[6].i = dir;
    constants[7].i = wxc_dir_stride;
    constants[8].i = whc_dir_stride;
    constants[9].i = bias_dir_stride;
    constants[10].i = bottom_step;
    constants[11].i = 0;

    VkMat dispatcher;
    dispatcher.w = hidden_size;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
}

static inline void record_lstm_step_h(const Pipeline* pipeline,
                                      VkCompute& cmd,
                                      const VkMat& bottom_blob,
                                      const VkMat& weight_xc,
                                      const VkMat& bias_c,
                                      const VkMat& weight_hc,
                                      const VkMat& hidden_prev,
                                      const VkMat& cell_prev,
                                      VkMat& hidden_h_next,
                                      VkMat& cell_next,
                                      int size,
                                      int num_output,
                                      int hidden_size,
                                      int ti,
                                      int dir,
                                      int wxc_dir_stride,
                                      int whc_dir_stride,
                                      int bias_dir_stride,
                                      int bottom_step)
{
    std::vector<VkMat> bindings(8);
    bindings[0] = bottom_blob;
    bindings[1] = weight_xc;
    bindings[2] = bias_c;
    bindings[3] = weight_hc;
    bindings[4] = hidden_prev;
    bindings[5] = cell_prev;
    bindings[6] = hidden_h_next;
    bindings[7] = cell_next;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = size;
    constants[1].i = num_output;
    constants[2].i = hidden_size;
    constants[3].i = ti;
    constants[4].i = dir;
    constants[5].i = wxc_dir_stride;
    constants[6].i = whc_dir_stride;
    constants[7].i = bias_dir_stride;
    constants[8].i = bottom_step;
    constants[9].i = 0;

    VkMat dispatcher;
    dispatcher.w = hidden_size;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
}

static inline void record_lstm_proj(const Pipeline* pipeline,
                                    VkCompute& cmd,
                                    const VkMat& hidden_h,
                                    const VkMat& weight_hr,
                                    VkMat& hidden_next,
                                    VkMat& top_blob,
                                    int hidden_size,
                                    int num_output,
                                    int ti,
                                    int outw,
                                    int out_offset,
                                    int dir,
                                    int hr_dir_stride)
{
    std::vector<VkMat> bindings(4);
    bindings[0] = hidden_h;
    bindings[1] = weight_hr;
    bindings[2] = hidden_next;
    bindings[3] = top_blob;

    std::vector<vk_constant_type> constants(7);
    constants[0].i = hidden_size;
    constants[1].i = num_output;
    constants[2].i = ti;
    constants[3].i = outw;
    constants[4].i = out_offset;
    constants[5].i = dir;
    constants[6].i = hr_dir_stride;

    VkMat dispatcher;
    dispatcher.w = num_output;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
}

int LSTM_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    if (!support_vulkan)
        return -1;

    VkMat bottom_blob = bottom_blobs[0];

    if (bottom_blob.dims != 2)
        return -1;

    const int size = bottom_blob.w;
    const int T = bottom_blob.h;

    const int num_directions = direction == 2 ? 2 : 1;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(num_output * num_directions, T, bottom_blob.elemsize, 1, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    VkAllocator* state_vkallocator = top_blobs.size() == 3 ? opt.blob_vkallocator : opt.workspace_vkallocator;

    VkMat hidden0;
    VkMat hidden0_next;
    VkMat cell0;
    VkMat cell0_next;

    hidden0.create(num_output, 1, bottom_blob.elemsize, 1, state_vkallocator);
    hidden0_next.create(num_output, 1, bottom_blob.elemsize, 1, state_vkallocator);
    cell0.create(hidden_size, 1, bottom_blob.elemsize, 1, state_vkallocator);
    cell0_next.create(hidden_size, 1, bottom_blob.elemsize, 1, state_vkallocator);
    if (hidden0.empty() || hidden0_next.empty() || cell0.empty() || cell0_next.empty())
        return -100;

    VkMat hidden1;
    VkMat hidden1_next;
    VkMat cell1;
    VkMat cell1_next;

    if (num_directions == 2)
    {
        hidden1.create(num_output, 1, bottom_blob.elemsize, 1, state_vkallocator);
        hidden1_next.create(num_output, 1, bottom_blob.elemsize, 1, state_vkallocator);
        cell1.create(hidden_size, 1, bottom_blob.elemsize, 1, state_vkallocator);
        cell1_next.create(hidden_size, 1, bottom_blob.elemsize, 1, state_vkallocator);
        if (hidden1.empty() || hidden1_next.empty() || cell1.empty() || cell1_next.empty())
            return -100;
    }

    if (bottom_blobs.size() == 3)
    {
        VkMat hidden_in;
        VkMat cell_in;
        vkdev->convert_packing(bottom_blobs[1], hidden_in, 1, cmd, opt);
        vkdev->convert_packing(bottom_blobs[2], cell_in, 1, cmd, opt);

        record_lstm_copy(pipeline_lstm_copy, cmd, hidden_in, hidden0, num_output, 0, 0, 1);
        record_lstm_copy(pipeline_lstm_copy, cmd, cell_in, cell0, hidden_size, 0, 0, 1);

        if (num_directions == 2)
        {
            record_lstm_copy(pipeline_lstm_copy, cmd, hidden_in, hidden1, num_output, num_output, 0, 1);
            record_lstm_copy(pipeline_lstm_copy, cmd, cell_in, cell1, hidden_size, hidden_size, 0, 1);
        }
    }
    else
    {
        record_lstm_copy(pipeline_lstm_copy, cmd, bottom_blob, hidden0, num_output, 0, 0, 0);
        record_lstm_copy(pipeline_lstm_copy, cmd, bottom_blob, cell0, hidden_size, 0, 0, 0);

        if (num_directions == 2)
        {
            record_lstm_copy(pipeline_lstm_copy, cmd, bottom_blob, hidden1, num_output, 0, 0, 0);
            record_lstm_copy(pipeline_lstm_copy, cmd, bottom_blob, cell1, hidden_size, 0, 0, 0);
        }
    }

    const int wxc_dir_stride = size * (hidden_size * 4);
    const int whc_dir_stride = num_output * (hidden_size * 4);
    const int bias_dir_stride = hidden_size * 4;
    const int hr_dir_stride = hidden_size * num_output;
    const int bottom_step = size;

    const bool has_projection = (num_output != hidden_size);

    VkMat hiddenh0;
    VkMat hiddenh1;
    if (has_projection)
    {
        hiddenh0.create(hidden_size, 1, bottom_blob.elemsize, 1, opt.workspace_vkallocator);
        if (hiddenh0.empty())
            return -100;

        if (num_directions == 2)
        {
            hiddenh1.create(hidden_size, 1, bottom_blob.elemsize, 1, opt.workspace_vkallocator);
            if (hiddenh1.empty())
                return -100;
        }
    }

    auto run_sequence = [&](int dir_index, int out_offset, int reverse,
                            VkMat& hprev, VkMat& hnext,
                            VkMat& cprev, VkMat& cnext,
                            VkMat& htmp) {
        for (int t = 0; t < T; t++)
        {
            const int ti = reverse ? (T - 1 - t) : t;

            if (!has_projection)
            {
                record_lstm_step(pipeline_lstm_step,
                                 cmd,
                                 bottom_blob,
                                 weight_xc_data_gpu,
                                 bias_c_data_gpu,
                                 weight_hc_data_gpu,
                                 hprev,
                                 cprev,
                                 hnext,
                                 cnext,
                                 top_blob,
                                 size,
                                 num_output,
                                 hidden_size,
                                 ti,
                                 top_blob.w,
                                 out_offset,
                                 dir_index,
                                 wxc_dir_stride,
                                 whc_dir_stride,
                                 bias_dir_stride,
                                 bottom_step);

                std::swap(hprev, hnext);
                std::swap(cprev, cnext);
            }
            else
            {
                record_lstm_step_h(pipeline_lstm_step_h,
                                   cmd,
                                   bottom_blob,
                                   weight_xc_data_gpu,
                                   bias_c_data_gpu,
                                   weight_hc_data_gpu,
                                   hprev,
                                   cprev,
                                   htmp,
                                   cnext,
                                   size,
                                   num_output,
                                   hidden_size,
                                   ti,
                                   dir_index,
                                   wxc_dir_stride,
                                   whc_dir_stride,
                                   bias_dir_stride,
                                   bottom_step);

                record_lstm_proj(pipeline_lstm_proj,
                                 cmd,
                                 htmp,
                                 weight_hr_data_gpu,
                                 hnext,
                                 top_blob,
                                 hidden_size,
                                 num_output,
                                 ti,
                                 top_blob.w,
                                 out_offset,
                                 dir_index,
                                 hr_dir_stride);

                std::swap(hprev, hnext);
                std::swap(cprev, cnext);
            }
        }
    };

    if (direction == 0 || direction == 1)
    {
        run_sequence(0, 0, direction, hidden0, hidden0_next, cell0, cell0_next, hiddenh0);
    }
    else
    {
        run_sequence(0, 0, 0, hidden0, hidden0_next, cell0, cell0_next, hiddenh0);
        run_sequence(1, num_output, 1, hidden1, hidden1_next, cell1, cell1_next, hiddenh1);
    }

    if (top_blobs.size() == 3)
    {
        if (num_directions == 1)
        {
            top_blobs[1] = hidden0;
            top_blobs[2] = cell0;
        }
        else
        {
            VkMat& hidden_out = top_blobs[1];
            VkMat& cell_out = top_blobs[2];

            hidden_out.create(num_output, 2, bottom_blob.elemsize, 1, opt.blob_vkallocator);
            cell_out.create(hidden_size, 2, bottom_blob.elemsize, 1, opt.blob_vkallocator);
            if (hidden_out.empty() || cell_out.empty())
                return -100;

            record_lstm_copy(pipeline_lstm_copy, cmd, hidden0, hidden_out, num_output, 0, 0, 1);
            record_lstm_copy(pipeline_lstm_copy, cmd, hidden1, hidden_out, num_output, 0, num_output, 1);

            record_lstm_copy(pipeline_lstm_copy, cmd, cell0, cell_out, hidden_size, 0, 0, 1);
            record_lstm_copy(pipeline_lstm_copy, cmd, cell1, cell_out, hidden_size, 0, hidden_size, 1);
        }
    }

    return 0;
}

int LSTM_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    std::vector<VkMat> bottom_blobs(1);
    std::vector<VkMat> top_blobs(1);
    bottom_blobs[0] = bottom_blob;

    int ret = forward(bottom_blobs, top_blobs, cmd, opt);
    top_blob = top_blobs[0];
    return ret;
}

} // namespace ncnn
