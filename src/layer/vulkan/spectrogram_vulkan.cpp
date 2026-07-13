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
    support_vulkan_any_packing = false;

    padding = 0;
    pipeline_spectrogram = 0;
}

int Spectrogram_vulkan::load_param(const ParamDict& pd)
{
    int ret = Spectrogram::load_param(pd);
    if (ret != 0)
        return ret;

    // stft as conv: kernel=n_fft stride=hoplen, dft basis as weight
    // row i = real basis of bin i, row freqs_onesided + i = imag basis
    const int freqs_onesided = n_fft / 2 + 1;

    float norm = 1.f;
    if (normalized == 1)
        norm = 1.f / sqrt((float)n_fft);
    if (normalized == 2)
        norm = window_data[n_fft];

    dft_weight.create(freqs_onesided * 2 * n_fft);
    {
        float* weight = dft_weight;
        for (int i = 0; i < freqs_onesided; i++)
        {
            float* re_row = weight + i * n_fft;
            float* im_row = weight + (freqs_onesided + i) * n_fft;
            for (int k = 0; k < n_fft; k++)
            {
                double angle = 2 * 3.14159265358979323846 * i * k / n_fft;
                re_row[k] = (float)(window_data[k] * cos(angle) * norm);
                im_row[k] = (float)(-window_data[k] * sin(angle) * norm);
            }
        }
    }

    return 0;
}

int Spectrogram_vulkan::create_pipeline(const Option& opt)
{
    if (center == 1)
    {
        padding = create_layer_vulkan(LayerType::Padding);
        padding->vkdev = vkdev;

        ParamDict pd;
        pd.set(0, 0);
        pd.set(1, 0);
        pd.set(2, n_fft / 2);
        pd.set(3, n_fft / 2);
        pd.set(4, pad_type);
        pd.set(5, 0.f);

        padding->load_param(pd);
        padding->create_pipeline(opt);
    }

    const int freqs_onesided = n_fft / 2 + 1;
    const int freqs = onesided ? freqs_onesided : n_fft;

    std::vector<vk_specialization_type> specializations(6);
    specializations[0].i = n_fft;
    specializations[1].i = hoplen;
    specializations[2].i = freqs_onesided;
    specializations[3].i = power;
    specializations[4].i = onesided;
    specializations[5].i = freqs;

    pipeline_spectrogram = new Pipeline(vkdev);
    pipeline_spectrogram->set_optimal_local_size_xyz(8, 8, 1);
    pipeline_spectrogram->create(LayerShaderType::spectrogram_conv, opt, specializations);

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

    delete pipeline_spectrogram;
    pipeline_spectrogram = 0;

    return 0;
}

int Spectrogram_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(dft_weight, weight_data_gpu, opt);

    dft_weight.release();

    return 0;
}

int Spectrogram_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    // center border via Padding layer (same semantics as cpu copy_make_border)
    const VkMat* input = &bottom_blob;
    VkMat bottom_blob_bordered;
    if (center == 1)
    {
        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt);
        input = &bottom_blob_bordered;
    }

    const int size = input->w;
    const int frames = (size - n_fft) / hoplen + 1;
    const int freqs_onesided = n_fft / 2 + 1;
    const int freqs = onesided ? freqs_onesided : n_fft;

    const size_t elemsize = bottom_blob.elemsize;

    if (power == 0)
    {
        top_blob.create(2, frames, freqs, elemsize, 1u, opt.blob_vkallocator);
    }
    else
    {
        top_blob.create(frames, freqs, elemsize, 1u, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = *input;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;

    std::vector<vk_constant_type> constants(2);
    constants[0].i = frames;
    constants[1].i = top_blob.cstep; // channel stride (power == 0 only)

    VkMat dispatcher;
    dispatcher.w = frames;
    dispatcher.h = freqs;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_spectrogram, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
