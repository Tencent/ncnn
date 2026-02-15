// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "spectrogram_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

#include <math.h>

namespace ncnn {

Spectrogram_vulkan::Spectrogram_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;
    support_any_packing = true;

    n_freq = 0;

    padding = 0;
    unfold = 0;
    gemm = 0;

    pipeline_spectrogram_post = 0;
    pipeline_spectrogram_post_pack4 = 0;
}

int Spectrogram_vulkan::create_pipeline(const Option& opt)
{
    n_freq = onesided ? (n_fft / 2 + 1) : n_fft;

    padding = ncnn::create_layer_vulkan(ncnn::LayerType::Padding);
    padding->vkdev = vkdev;
    {
        ncnn::ParamDict pd;

        const int pad = center ? n_fft / 2 : 0;
        pd.set(0, 0);
        pd.set(1, 0);
        pd.set(2, pad);
        pd.set(3, pad);
        pd.set(4, pad_type);
        pd.set(5, 0.f);
        pd.set(7, 0);
        pd.set(8, 0);

        padding->load_param(pd);
        padding->load_model(ModelBinFromMatArray(0));
        padding->create_pipeline(opt);
    }

    unfold = ncnn::create_layer_vulkan(ncnn::LayerType::Unfold);
    unfold->vkdev = vkdev;
    {
        ncnn::ParamDict pd;

        pd.set(1, n_fft);
        pd.set(11, 1);
        pd.set(2, 1);
        pd.set(12, 1);
        pd.set(3, hoplen);
        pd.set(13, 1);
        pd.set(4, 0);
        pd.set(15, 0);
        pd.set(14, 0);
        pd.set(16, 0);
        pd.set(18, 0.f);

        unfold->load_param(pd);
        unfold->load_model(ModelBinFromMatArray(0));
        unfold->create_pipeline(opt);
    }

    gemm = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
    gemm->vkdev = vkdev;
    {
        ncnn::ParamDict pd;

        pd.set(0, 1.f);
        pd.set(1, 1.f);
        pd.set(2, 0);
        pd.set(3, 0);
        pd.set(4, 1);
        pd.set(5, 0);
        pd.set(6, 0);
        pd.set(7, 2 * n_freq);
        pd.set(8, 0);
        pd.set(9, n_fft);
        pd.set(14, 0);

        gemm->load_param(pd);

        Mat basis_data;
        basis_data.create(n_fft, 2 * n_freq, (size_t)4u, 1);
        if (basis_data.empty())
            return -100;

        for (int i = 0; i < n_freq; i++)
        {
            float* real_row = basis_data.row<float>(i);
            float* imag_row = basis_data.row<float>(i + n_freq);

            for (int j = 0; j < n_fft; j++)
            {
                const double angle = 2 * 3.14159265358979323846 * i * j / n_fft;
                const float w = window_data[j];

                real_row[j] = (float)cos(angle) * w;
                imag_row[j] = (float)(-sin(angle)) * w;
            }
        }

        Mat weights[1];
        weights[0] = basis_data;

        gemm->load_model(ModelBinFromMatArray(weights));
        gemm->create_pipeline(opt);

        if (opt.lightmode)
        {
            window_data.release();
        }
    }

    {
        std::vector<vk_specialization_type> specializations(1);
        specializations[0].i = power;

        pipeline_spectrogram_post = new Pipeline(vkdev);
        pipeline_spectrogram_post->set_local_size_xyz(8, 8, 1);
        pipeline_spectrogram_post->create(LayerShaderType::spectrogram_post, opt, specializations);

        pipeline_spectrogram_post_pack4 = new Pipeline(vkdev);
        pipeline_spectrogram_post_pack4->set_local_size_xyz(8, 8, 1);
        pipeline_spectrogram_post_pack4->create(LayerShaderType::spectrogram_post_pack4, opt, specializations);
    }

    return 0;
}

int Spectrogram_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    if (unfold)
    {
        unfold->destroy_pipeline(opt);
        delete unfold;
        unfold = 0;
    }

    if (gemm)
    {
        gemm->destroy_pipeline(opt);
        delete gemm;
        gemm = 0;
    }

    delete pipeline_spectrogram_post;
    pipeline_spectrogram_post = 0;

    delete pipeline_spectrogram_post_pack4;
    pipeline_spectrogram_post_pack4 = 0;

    return 0;
}

int Spectrogram_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (padding) padding->upload_model(cmd, opt);
    if (unfold) unfold->upload_model(cmd, opt);
    if (gemm) gemm->upload_model(cmd, opt);
    return 0;
}

int Spectrogram_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    VkMat x = bottom_blob;

    if (center)
    {
        VkMat xpad;
        int retp = padding->forward(x, xpad, cmd, opt);
        if (retp != 0)
            return retp;
        x = xpad;
    }

    const int size = x.w * x.elempack;
    const int frames = (size - n_fft) / hoplen + 1;
    if (frames <= 0)
        return -100;

    VkMat cols;
    {
        int retu = unfold->forward(x, cols, cmd, opt);
        if (retu != 0)
            return retu;
    }

    VkMat y;
    {
        std::vector<VkMat> inputs(1);
        inputs[0] = cols;

        std::vector<VkMat> outputs(1);
        outputs[0] = VkMat();

        int retg = gemm->forward(inputs, outputs, cmd, opt);
        if (retg != 0)
            return retg;

        vkdev->convert_packing(outputs[0], y, 1, cmd, opt);
    }

    const int out_elempack = (opt.use_packing_layout && (n_freq % 4 == 0)) ? 4 : 1;
    const size_t scalar_elemsize = bottom_blob.elemsize / bottom_blob.elempack;
    const size_t out_elemsize = scalar_elemsize * out_elempack;

    if (power == 0)
    {
        top_blob.create(2, frames, n_freq / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }
    else
    {
        top_blob.create(frames, n_freq / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    std::vector<VkMat> bindings(2);
    bindings[0] = y;
    bindings[1] = top_blob;

    if (out_elempack == 1)
    {
        const int top_row_stride = (power != 0) ? top_blob.w : (int)top_blob.cstep;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = frames;
        constants[1].i = n_freq;
        constants[2].i = y.w;
        constants[3].i = top_row_stride;

        VkMat dispatcher;
        dispatcher.w = frames;
        dispatcher.h = n_freq;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_spectrogram_post, bindings, constants, dispatcher);
    }
    else
    {
        const int top_row_stride = (power != 0) ? top_blob.w : (int)top_blob.cstep;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = frames;
        constants[1].i = n_freq;
        constants[2].i = n_freq / 4;
        constants[3].i = y.w;
        constants[4].i = top_row_stride;

        VkMat dispatcher;
        dispatcher.w = frames;
        dispatcher.h = n_freq / 4;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_spectrogram_post_pack4, bindings, constants, dispatcher);
    }

    return 0;
}

} // namespace ncnn
