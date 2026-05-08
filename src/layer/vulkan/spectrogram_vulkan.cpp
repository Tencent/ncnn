// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "spectrogram_vulkan.h"

#include "layer_shader_type.h"
#include <math.h>

namespace ncnn {

Spectrogram_vulkan::Spectrogram_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;
    support_any_packing = true;

    n_freq = 0;

    pipeline_spectrogram_packed = 0;
}

int Spectrogram_vulkan::create_pipeline(const Option& opt)
{
    n_freq = onesided ? (n_fft / 2 + 1) : n_fft;

    int out_elempack = 1;
    if (opt.use_packing_layout && n_freq % 4 == 0)
        out_elempack = 4;

    int n_freq_packed = (n_freq + 3) / 4 * 4;

    basis_data_packed.create(n_fft, 1, n_freq_packed / 4, (size_t)4 * 4, 4);
    basis_imag_data_packed.create(n_fft, 1, n_freq_packed / 4, (size_t)4 * 4, 4);

    for (int q = 0; q < n_freq_packed; q += 4)
    {
        float* g00r = basis_data_packed.channel(q / 4);
        float* g00i = basis_imag_data_packed.channel(q / 4);

        for (int k = 0; k < n_fft; k++)
        {
            for (int i = 0; i < 4; i++)
            {
                int f = q + i;
                if (f < n_freq)
                {
                    const double angle = 2 * 3.14159265358979323846 * f * k / n_fft;
                    const float w = window_data[k];

                    g00r[0] = (float)cos(angle) * w;
                    g00i[0] = (float)(-sin(angle)) * w;
                }
                else
                {
                    g00r[0] = 0.f;
                    g00i[0] = 0.f;
                }
                g00r++;
                g00i++;
            }
        }
    }

    {
        std::vector<vk_specialization_type> specializations(4);
        specializations[0].i = power;
        specializations[1].i = n_fft;
        specializations[2].i = hoplen;
        specializations[3].i = out_elempack;

        pipeline_spectrogram_packed = new Pipeline(vkdev);
        pipeline_spectrogram_packed->set_local_size_xyz(8, 8, 1);
        pipeline_spectrogram_packed->create(LayerShaderType::spectrogram_packed, opt, specializations);
    }

    if (opt.lightmode)
    {
        window_data.release();
    }

    return 0;
}

int Spectrogram_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_spectrogram_packed;
    pipeline_spectrogram_packed = 0;

    return 0;
}

int Spectrogram_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(basis_data_packed, basis_data_gpu, opt);
    cmd.record_upload(basis_imag_data_packed, basis_imag_data_gpu, opt);

    basis_data_packed.release();
    basis_imag_data_packed.release();

    return 0;
}

int Spectrogram_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int size = bottom_blob.w * bottom_blob.elempack;
    const int pad = center ? n_fft / 2 : 0;
    const int frames = (size + 2 * pad - n_fft) / hoplen + 1;
    if (frames <= 0)
        return -100;

    int out_elempack = 1;
    if (opt.use_packing_layout && n_freq % 4 == 0)
        out_elempack = 4;

    const size_t scalar_elemsize = bottom_blob.elemsize / bottom_blob.elempack;
    const size_t out_elemsize = scalar_elemsize * out_elempack;

    int top_row_stride;
    if (power == 0)
    {
        top_blob.create(2, frames, n_freq / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
        top_row_stride = (int)top_blob.cstep;
    }
    else
    {
        top_blob.create(frames, n_freq / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
        top_row_stride = top_blob.w;
    }

    std::vector<VkMat> bindings(6);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    bindings[2] = bottom_blob;
    bindings[3] = top_blob;
    bindings[4] = basis_data_gpu;
    bindings[5] = basis_imag_data_gpu;

    std::vector<vk_constant_type> constants(6);
    constants[0].i = frames;
    constants[1].i = n_freq;
    constants[2].i = size;
    constants[3].i = top_row_stride;
    constants[4].i = center;
    constants[5].i = pad_type;

    VkMat dispatcher;
    dispatcher.w = frames;
    dispatcher.h = (n_freq + 3) / 4;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_spectrogram_packed, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
