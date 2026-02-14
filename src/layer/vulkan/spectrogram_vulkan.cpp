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
    support_vulkan_packing = false;
    support_any_packing = false;

    n_freq = 0;

    padding = 0;
    unfold = 0;
    gemm = 0;

    pipeline_spectrogram_post = 0;
}

int Spectrogram_vulkan::create_pipeline(const Option& opt)
{
    if (onesided)
        n_freq = n_fft / 2 + 1;
    else
        n_freq = n_fft;

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

        pd.set(1, n_fft);  // kernel_w
        pd.set(11, 1);     // kernel_h
        pd.set(2, 1);      // dilation_w
        pd.set(12, 1);     // dilation_h
        pd.set(3, hoplen); // stride_w
        pd.set(13, 1);     // stride_h
        pd.set(4, 0);      // pad_left
        pd.set(15, 0);     // pad_right
        pd.set(14, 0);     // pad_top
        pd.set(16, 0);     // pad_bottom
        pd.set(18, 0.f);   // pad_value

        unfold->load_param(pd);
        unfold->load_model(ModelBinFromMatArray(0));
        unfold->create_pipeline(opt);
    }

    gemm = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
    gemm->vkdev = vkdev;
    {
        ncnn::ParamDict pd;

        pd.set(0, 1.f);        // alpha
        pd.set(1, 1.f);        // beta
        pd.set(2, 0);          // transA
        pd.set(3, 0);          // transB
        pd.set(4, 1);          // constantA
        pd.set(5, 0);          // constantB
        pd.set(6, 0);          // constantC
        pd.set(7, 2 * n_freq); // constantM
        pd.set(8, 0);          // constantN
        pd.set(9, n_fft);      // constantK
        pd.set(14, 0);         // output_transpose

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

    const int size = x.w;
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

    const size_t elemsize = bottom_blob.elemsize;

    if (power == 0)
    {
        top_blob.create(2, frames, n_freq, elemsize, 1, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }
    else
    {
        top_blob.create(frames, n_freq, elemsize, 1, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    std::vector<VkMat> bindings(2);
    bindings[0] = y;
    bindings[1] = top_blob;

    const int top_row_stride = (power != 0) ? top_blob.w : (int)top_blob.cstep;

    std::vector<vk_constant_type> constants(4);
    constants[0].i = frames;
    constants[1].i = n_freq;
    constants[2].i = y.w;
    constants[3].i = top_row_stride;

    VkMat dispatcher;
    dispatcher.w = frames; // x = frame index
    dispatcher.h = n_freq; // y = freq index
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_spectrogram_post, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
