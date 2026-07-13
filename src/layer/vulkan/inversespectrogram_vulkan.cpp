// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "inversespectrogram_vulkan.h"

#include "layer_shader_type.h"

#include <math.h>

namespace ncnn {

InverseSpectrogram_vulkan::InverseSpectrogram_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;

    pipeline_inversespectrogram = 0;
}

int InverseSpectrogram_vulkan::load_param(const ParamDict& pd)
{
    int ret = InverseSpectrogram::load_param(pd);
    if (ret != 0)
        return ret;

    // idft basis as conv weight, 4 planes of [bin k][tap m]
    // re[m] += (sp_re[k] * cos(2*pi*k*m/n_fft) - sp_im[k] * sin(2*pi*k*m/n_fft)) / n_fft * window[m] * norm
    // im[m] += (sp_re[k] * sin(2*pi*k*m/n_fft) + sp_im[k] * cos(2*pi*k*m/n_fft)) / n_fft * window[m] * norm
    float norm = 1.f;
    if (normalized == 1)
        norm = sqrt((float)n_fft);
    if (normalized == 2)
        norm = window_data[n_fft];

    idft_weight.create(4 * n_fft * n_fft);
    {
        float* w_re_re = idft_weight;
        float* w_re_im = w_re_re + n_fft * n_fft;
        float* w_im_re = w_re_im + n_fft * n_fft;
        float* w_im_im = w_im_re + n_fft * n_fft;
        for (int k = 0; k < n_fft; k++)
        {
            for (int m = 0; m < n_fft; m++)
            {
                double angle = 2 * 3.14159265358979323846 * k * m / n_fft;
                const float wcos = (float)(window_data[m] * cos(angle) / n_fft * norm);
                const float wsin = (float)(window_data[m] * sin(angle) / n_fft * norm);
                w_re_re[k * n_fft + m] = wcos;
                w_re_im[k * n_fft + m] = -wsin;
                w_im_re[k * n_fft + m] = wsin;
                w_im_im[k * n_fft + m] = wcos;
            }
        }
    }

    window_sq.create(n_fft);
    {
        float* p = window_sq;
        for (int i = 0; i < n_fft; i++)
        {
            p[i] = window_data[i] * window_data[i];
        }
    }

    return 0;
}

int InverseSpectrogram_vulkan::create_pipeline(const Option& opt)
{
    const int pad = center == 1 ? n_fft / 2 : 0;

    std::vector<vk_specialization_type> specializations(4);
    specializations[0].i = n_fft;
    specializations[1].i = hoplen;
    specializations[2].i = returns;
    specializations[3].i = pad;

    pipeline_inversespectrogram = new Pipeline(vkdev);
    pipeline_inversespectrogram->set_optimal_local_size_xyz(64, 1, 1);
    pipeline_inversespectrogram->create(LayerShaderType::inversespectrogram_conv, opt, specializations);

    return 0;
}

int InverseSpectrogram_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_inversespectrogram;
    pipeline_inversespectrogram = 0;

    return 0;
}

int InverseSpectrogram_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(idft_weight, weight_data_gpu, opt);
    cmd.record_upload(window_sq, window_sq_gpu, opt);

    idft_weight.release();
    window_sq.release();

    return 0;
}

int InverseSpectrogram_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    // input is complex spectrogram [c=freqs, h=frames, w=2]
    const int frames = bottom_blob.h;

    const int pad = center == 1 ? n_fft / 2 : 0;
    const int outsize = (frames - 1) * hoplen + n_fft - pad * 2;

    const size_t elemsize = bottom_blob.elemsize;

    if (returns == 0)
    {
        top_blob.create(2, outsize, elemsize, 1u, opt.blob_vkallocator);
    }
    else
    {
        top_blob.create(outsize, elemsize, 1u, opt.blob_vkallocator);
    }
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;
    bindings[3] = window_sq_gpu;

    std::vector<vk_constant_type> constants(4);
    constants[0].i = frames;
    constants[1].i = outsize;
    constants[2].i = bottom_blob.c;
    constants[3].i = bottom_blob.cstep;

    VkMat dispatcher;
    dispatcher.w = outsize;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_inversespectrogram, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
