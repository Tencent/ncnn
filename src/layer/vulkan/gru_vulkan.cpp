// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "gru_vulkan.h"

#include <algorithm>

#include "layer_shader_type.h"

namespace ncnn {

GRU_vulkan::GRU_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;

    pipeline_gru_step = 0;
    pipeline_gru_step_pack4 = 0;
    pipeline_gru_copy = 0;
}

int GRU_vulkan::load_param(const ParamDict& pd)
{
    int ret = GRU::load_param(pd);

    if (int8_scale_term)
    {
        support_vulkan = false;
    }

    return ret;
}

int GRU_vulkan::create_pipeline(const Option& opt)
{
    if (!support_vulkan)
        return 0;

    {
        pipeline_gru_step = new Pipeline(vkdev);
        pipeline_gru_step->set_local_size_xyz(64, 1, 1);

        std::vector<vk_specialization_type> specializations;
        pipeline_gru_step->create(LayerShaderType::gru_step, opt, specializations);
    }

    if (num_output % 4 == 0)
    {
        pipeline_gru_step_pack4 = new Pipeline(vkdev);
        pipeline_gru_step_pack4->set_local_size_xyz(64, 1, 1);

        std::vector<vk_specialization_type> specializations;
        pipeline_gru_step_pack4->create(LayerShaderType::gru_step_pack4, opt, specializations);
    }

    {
        pipeline_gru_copy = new Pipeline(vkdev);
        pipeline_gru_copy->set_local_size_xyz(64, 1, 1);

        std::vector<vk_specialization_type> specializations;
        pipeline_gru_copy->create(LayerShaderType::gru_copy, opt, specializations);
    }

    return 0;
}

int GRU_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_gru_step;
    pipeline_gru_step = 0;

    delete pipeline_gru_step_pack4;
    pipeline_gru_step_pack4 = 0;

    delete pipeline_gru_copy;
    pipeline_gru_copy = 0;

    return 0;
}

static void pack_gru_weights_bias_pack4(const Mat& weight_xc_data,
                                        const Mat& bias_c_data,
                                        const Mat& weight_hc_data,
                                        Mat& weight_xc_data_pack4,
                                        Mat& bias_c_data_pack4,
                                        Mat& weight_hc_data_pack4,
                                        int size,
                                        int num_output,
                                        int num_directions)
{
    const int num_output_pack = num_output / 4;

    weight_xc_data_pack4.create(size, num_directions * 3 * num_output_pack, (size_t)16u, 4);
    weight_hc_data_pack4.create(num_output, num_directions * 3 * num_output_pack, (size_t)16u, 4);
    bias_c_data_pack4.create(num_output_pack, num_directions * 4, (size_t)16u, 4);

    const float* wxc_ptr = weight_xc_data;
    const float* whc_ptr = weight_hc_data;
    const float* bias_ptr = bias_c_data;

    for (int dir = 0; dir < num_directions; dir++)
    {
        for (int gate = 0; gate < 3; gate++)
        {
            for (int q_pack = 0; q_pack < num_output_pack; q_pack++)
            {
                float* wxc_row = weight_xc_data_pack4.row(dir * 3 * num_output_pack + gate * num_output_pack + q_pack);
                float* whc_row = weight_hc_data_pack4.row(dir * 3 * num_output_pack + gate * num_output_pack + q_pack);

                for (int i = 0; i < size; i++)
                {
                    const int src_base = (dir * 3 * num_output + gate * num_output + q_pack * 4) * size + i;

                    wxc_row[i * 4 + 0] = wxc_ptr[src_base + 0 * size];
                    wxc_row[i * 4 + 1] = wxc_ptr[src_base + 1 * size];
                    wxc_row[i * 4 + 2] = wxc_ptr[src_base + 2 * size];
                    wxc_row[i * 4 + 3] = wxc_ptr[src_base + 3 * size];
                }

                for (int i = 0; i < num_output; i++)
                {
                    const int src_base = (dir * 3 * num_output + gate * num_output + q_pack * 4) * num_output + i;

                    whc_row[i * 4 + 0] = whc_ptr[src_base + 0 * num_output];
                    whc_row[i * 4 + 1] = whc_ptr[src_base + 1 * num_output];
                    whc_row[i * 4 + 2] = whc_ptr[src_base + 2 * num_output];
                    whc_row[i * 4 + 3] = whc_ptr[src_base + 3 * num_output];
                }
            }
        }

        for (int b = 0; b < 4; b++)
        {
            float* bias_row = bias_c_data_pack4.row(dir * 4 + b);

            for (int q_pack = 0; q_pack < num_output_pack; q_pack++)
            {
                const int q0 = q_pack * 4;
                const int src_base = dir * (num_output * 4) + b * num_output + q0;

                bias_row[q_pack * 4 + 0] = bias_ptr[src_base + 0];
                bias_row[q_pack * 4 + 1] = bias_ptr[src_base + 1];
                bias_row[q_pack * 4 + 2] = bias_ptr[src_base + 2];
                bias_row[q_pack * 4 + 3] = bias_ptr[src_base + 3];
            }
        }
    }
}

int GRU_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (!support_vulkan)
        return 0;

    cmd.record_upload(weight_xc_data, weight_xc_data_gpu, opt);
    cmd.record_upload(bias_c_data, bias_c_data_gpu, opt);
    cmd.record_upload(weight_hc_data, weight_hc_data_gpu, opt);

    if (num_output % 4 == 0)
    {
        const int size = weight_xc_data.w;
        const int num_directions = direction == 2 ? 2 : 1;

        Mat weight_xc_data_pack4;
        Mat bias_c_data_pack4;
        Mat weight_hc_data_pack4;

        pack_gru_weights_bias_pack4(weight_xc_data, bias_c_data, weight_hc_data,
                                    weight_xc_data_pack4, bias_c_data_pack4, weight_hc_data_pack4,
                                    size, num_output, num_directions);

        cmd.record_upload(weight_xc_data_pack4, weight_xc_data_gpu_pack4, opt);
        cmd.record_upload(bias_c_data_pack4, bias_c_data_gpu_pack4, opt);
        cmd.record_upload(weight_hc_data_pack4, weight_hc_data_gpu_pack4, opt);
    }

    if (opt.lightmode)
    {
        weight_xc_data.release();
        bias_c_data.release();
        weight_hc_data.release();
    }

    return 0;
}

static inline void record_gru_copy(const Pipeline* pipeline,
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

static inline void record_gru_step_pack1(const Pipeline* pipeline,
        VkCompute& cmd,
        const VkMat& bottom_blob,
        const VkMat& weight_xc,
        const VkMat& bias_c,
        const VkMat& weight_hc,
        const VkMat& hidden_prev,
        VkMat& hidden_next,
        VkMat& top_blob,
        int size,
        int num_output,
        int ti,
        int outw,
        int out_offset,
        int dir,
        int wxc_dir_stride,
        int whc_dir_stride,
        int bias_dir_stride,
        int bottom_step)
{
    std::vector<VkMat> bindings(7);
    bindings[0] = bottom_blob;
    bindings[1] = weight_xc;
    bindings[2] = bias_c;
    bindings[3] = weight_hc;
    bindings[4] = hidden_prev;
    bindings[5] = hidden_next;
    bindings[6] = top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = size;
    constants[1].i = num_output;
    constants[2].i = ti;
    constants[3].i = outw;
    constants[4].i = out_offset;
    constants[5].i = dir;
    constants[6].i = wxc_dir_stride;
    constants[7].i = whc_dir_stride;
    constants[8].i = bias_dir_stride;
    constants[9].i = bottom_step;

    VkMat dispatcher;
    dispatcher.w = num_output;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
}

static inline void record_gru_step_pack4(const Pipeline* pipeline,
        VkCompute& cmd,
        const VkMat& bottom_blob,
        const VkMat& weight_xc_pack4,
        const VkMat& bias_c_pack4,
        const VkMat& weight_hc_pack4,
        const VkMat& hidden_prev,
        VkMat& hidden_next,
        VkMat& top_blob,
        int size,
        int num_output,
        int ti,
        int outw,
        int out_offset,
        int dir)
{
    std::vector<VkMat> bindings(7);
    bindings[0] = bottom_blob;
    bindings[1] = weight_xc_pack4;
    bindings[2] = bias_c_pack4;
    bindings[3] = weight_hc_pack4;
    bindings[4] = hidden_prev;
    bindings[5] = hidden_next;
    bindings[6] = top_blob;

    std::vector<vk_constant_type> constants(6);
    constants[0].i = size;
    constants[1].i = num_output;
    constants[2].i = ti;
    constants[3].i = outw;
    constants[4].i = out_offset;
    constants[5].i = dir;

    VkMat dispatcher;
    dispatcher.w = num_output / 4;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
}

int GRU_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    if (!support_vulkan)
        return -1;

    const VkMat& bottom_blob = bottom_blobs[0];

    const int size = bottom_blob.w;
    const int timesteps = bottom_blob.h;

    const int num_directions = direction == 2 ? 2 : 1;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(num_output * num_directions, timesteps, bottom_blob.elemsize, 1, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    VkAllocator* hidden_vkallocator = top_blobs.size() == 2 ? opt.blob_vkallocator : opt.workspace_vkallocator;

    VkMat hidden0;
    VkMat hidden0_next;
    hidden0.create(num_output, 1, bottom_blob.elemsize, 1, hidden_vkallocator);
    hidden0_next.create(num_output, 1, bottom_blob.elemsize, 1, hidden_vkallocator);
    if (hidden0.empty() || hidden0_next.empty())
        return -100;

    VkMat hidden1;
    VkMat hidden1_next;
    if (num_directions == 2)
    {
        hidden1.create(num_output, 1, bottom_blob.elemsize, 1, hidden_vkallocator);
        hidden1_next.create(num_output, 1, bottom_blob.elemsize, 1, hidden_vkallocator);
        if (hidden1.empty() || hidden1_next.empty())
            return -100;
    }

    if (bottom_blobs.size() == 2)
    {
        const VkMat& hidden_in0 = bottom_blobs[1];

        if (num_directions == 1)
        {
            record_gru_copy(pipeline_gru_copy, cmd, hidden_in0, hidden0, num_output, 0, 0, 1);
        }
        else
        {
            record_gru_copy(pipeline_gru_copy, cmd, hidden_in0, hidden0, num_output, 0, 0, 1);
            record_gru_copy(pipeline_gru_copy, cmd, hidden_in0, hidden1, num_output, num_output, 0, 1);
        }
    }
    else
    {
        record_gru_copy(pipeline_gru_copy, cmd, bottom_blob, hidden0, num_output, 0, 0, 0);
        if (num_directions == 2)
        {
            record_gru_copy(pipeline_gru_copy, cmd, bottom_blob, hidden1, num_output, 0, 0, 0);
        }
    }

    const int wxc_dir_stride = size * (num_output * 3);
    const int whc_dir_stride = num_output * (num_output * 3);
    const int bias_dir_stride = num_output * 4;
    const int bottom_step = size;

    const bool use_pack4 = (num_output % 4 == 0)
                           && pipeline_gru_step_pack4
                           && !weight_xc_data_gpu_pack4.empty()
                           && !bias_c_data_gpu_pack4.empty()
                           && !weight_hc_data_gpu_pack4.empty();

    auto run_sequence = [&](int dir_index, int out_offset, int reverse, VkMat& hprev, VkMat& hnext) {
        for (int t = 0; t < timesteps; t++)
        {
            const int ti = reverse ? (timesteps - 1 - t) : t;

            if (use_pack4)
            {
                record_gru_step_pack4(pipeline_gru_step_pack4,
                                      cmd,
                                      bottom_blob,
                                      weight_xc_data_gpu_pack4,
                                      bias_c_data_gpu_pack4,
                                      weight_hc_data_gpu_pack4,
                                      hprev,
                                      hnext,
                                      top_blob,
                                      size,
                                      num_output,
                                      ti,
                                      top_blob.w,
                                      out_offset,
                                      dir_index);
            }
            else
            {
                record_gru_step_pack1(pipeline_gru_step,
                                      cmd,
                                      bottom_blob,
                                      weight_xc_data_gpu,
                                      bias_c_data_gpu,
                                      weight_hc_data_gpu,
                                      hprev,
                                      hnext,
                                      top_blob,
                                      size,
                                      num_output,
                                      ti,
                                      top_blob.w,
                                      out_offset,
                                      dir_index,
                                      wxc_dir_stride,
                                      whc_dir_stride,
                                      bias_dir_stride,
                                      bottom_step);
            }

            std::swap(hprev, hnext);
        }
    };

    if (direction == 0 || direction == 1)
    {
        run_sequence(0, 0, direction, hidden0, hidden0_next);
    }
    else
    {
        run_sequence(0, 0, 0, hidden0, hidden0_next);
        run_sequence(1, num_output, 1, hidden1, hidden1_next);
    }

    if (top_blobs.size() == 2)
    {
        if (num_directions == 1)
        {
            top_blobs[1] = hidden0;
        }
        else
        {
            VkMat& hidden_out = top_blobs[1];
            hidden_out.create(num_output, 2, bottom_blob.elemsize, 1, opt.blob_vkallocator);
            if (hidden_out.empty())
                return -100;

            record_gru_copy(pipeline_gru_copy, cmd, hidden0, hidden_out, num_output, 0, 0, 1);
            record_gru_copy(pipeline_gru_copy, cmd, hidden1, hidden_out, num_output, 0, num_output, 1);
        }
    }

    return 0;
}

int GRU_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    std::vector<VkMat> bottom_blobs(1);
    std::vector<VkMat> top_blobs(1);
    bottom_blobs[0] = bottom_blob;

    int ret = forward(bottom_blobs, top_blobs, cmd, opt);
    top_blob = top_blobs[0];
    return ret;
}

} // namespace ncnn
