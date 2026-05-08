// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "inversespectrogram_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

InverseSpectrogram_vulkan::InverseSpectrogram_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;
    support_any_packing = true;

    pipeline_inversespectrogram_idft = 0;
    pipeline_inversespectrogram_idft_pack4 = 0;
    pipeline_inversespectrogram_ola = 0;
    pipeline_inversespectrogram_ola_pack4 = 0;
}

int InverseSpectrogram_vulkan::create_pipeline(const Option& opt)
{
    int n_time_groups = (n_fft + 3) / 4;

    basis_cos_data_packed.create(n_fft, 1, n_time_groups, (size_t)4 * 4, 4);
    basis_sin_data_packed.create(n_fft, 1, n_time_groups, (size_t)4 * 4, 4);

    for (int q = 0; q < n_time_groups; q++)
    {
        float* g00c = basis_cos_data_packed.channel(q);
        float* g00s = basis_sin_data_packed.channel(q);

        for (int k = 0; k < n_fft; k++)
        {
            for (int i = 0; i < 4; i++)
            {
                int t = q * 4 + i;
                if (t < n_fft)
                {
                    const double angle = 2 * 3.14159265358979323846 * k * t / n_fft;
                    const float scale = window_data[t] / n_fft;

                    g00c[0] = (float)cos(angle) * scale;
                    g00s[0] = (float)sin(angle) * scale;
                }
                else
                {
                    g00c[0] = 0.f;
                    g00s[0] = 0.f;
                }
                g00c++;
                g00s++;
            }
        }
    }

    window2_data.create(n_fft, (size_t)4u, 1);
    for (int i = 0; i < n_fft; i++)
    {
        window2_data[i] = window_data[i] * window_data[i];
    }

    {
        std::vector<vk_specialization_type> specializations(3);
        specializations[0].i = n_fft;
        specializations[1].i = n_fft;
        specializations[2].i = 1;

        pipeline_inversespectrogram_idft = new Pipeline(vkdev);
        pipeline_inversespectrogram_idft->set_local_size_xyz(8, 8, 1);
        pipeline_inversespectrogram_idft->create(LayerShaderType::inversespectrogram_idft_packed, opt, specializations);
    }

    {
        std::vector<vk_specialization_type> specializations(3);
        specializations[0].i = n_fft;
        specializations[1].i = n_fft;
        specializations[2].i = 4;

        pipeline_inversespectrogram_idft_pack4 = new Pipeline(vkdev);
        pipeline_inversespectrogram_idft_pack4->set_local_size_xyz(8, 8, 1);
        pipeline_inversespectrogram_idft_pack4->create(LayerShaderType::inversespectrogram_idft_packed, opt, specializations);
    }

    {
        std::vector<vk_specialization_type> specializations(2);
        specializations[0].i = returns;
        specializations[1].i = 1;

        pipeline_inversespectrogram_ola = new Pipeline(vkdev);
        pipeline_inversespectrogram_ola->set_local_size_xyz(64, 1, 1);
        pipeline_inversespectrogram_ola->create(LayerShaderType::inversespectrogram_ola_packed, opt, specializations);
    }

    {
        std::vector<vk_specialization_type> specializations(2);
        specializations[0].i = returns;
        specializations[1].i = 4;

        pipeline_inversespectrogram_ola_pack4 = new Pipeline(vkdev);
        pipeline_inversespectrogram_ola_pack4->set_local_size_xyz(64, 1, 1);
        pipeline_inversespectrogram_ola_pack4->create(LayerShaderType::inversespectrogram_ola_packed, opt, specializations);
    }

    if (opt.lightmode)
    {
        window_data.release();
    }

    return 0;
}

int InverseSpectrogram_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_inversespectrogram_idft;
    pipeline_inversespectrogram_idft = 0;

    delete pipeline_inversespectrogram_idft_pack4;
    pipeline_inversespectrogram_idft_pack4 = 0;

    delete pipeline_inversespectrogram_ola;
    pipeline_inversespectrogram_ola = 0;

    delete pipeline_inversespectrogram_ola_pack4;
    pipeline_inversespectrogram_ola_pack4 = 0;

    window2_data.release();
    window2_data_gpu.release();

    return 0;
}

int InverseSpectrogram_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(basis_cos_data_packed, basis_cos_data_gpu, opt);
    cmd.record_upload(basis_sin_data_packed, basis_sin_data_gpu, opt);

    basis_cos_data_packed.release();
    basis_sin_data_packed.release();

    if (!window2_data.empty())
    {
        cmd.record_upload(window2_data, window2_data_gpu, opt);
    }

    return 0;
}

int InverseSpectrogram_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int frames = bottom_blob.h;
    const int freqs = bottom_blob.c * bottom_blob.elempack;

    if (frames <= 0)
        return -100;

    const int outsize = center ? (frames - 1) * hoplen + (n_fft - n_fft / 2 * 2) : (frames - 1) * hoplen + n_fft;
    if (outsize <= 0)
        return -100;

    const size_t scalar_elemsize = bottom_blob.elemsize / bottom_blob.elempack;

    VkMat yre;
    VkMat yim;
    yre.create(n_fft, frames, scalar_elemsize, 1, opt.workspace_vkallocator);
    yim.create(n_fft, frames, scalar_elemsize, 1, opt.workspace_vkallocator);
    if (yre.empty() || yim.empty())
        return -100;

    Pipeline* pipeline_idft = bottom_blob.elempack == 4 ? pipeline_inversespectrogram_idft_pack4 : pipeline_inversespectrogram_idft;

    {
        std::vector<VkMat> bindings(6);
        bindings[0] = bottom_blob;
        bindings[1] = yre;
        bindings[2] = bottom_blob;
        bindings[3] = yim;
        bindings[4] = basis_cos_data_gpu;
        bindings[5] = basis_sin_data_gpu;

        std::vector<vk_constant_type> constants(7);
        constants[0].i = frames;
        constants[1].i = n_fft;
        constants[2].i = freqs;
        constants[3].i = bottom_blob.w;
        constants[4].i = bottom_blob.h;
        constants[5].i = (int)bottom_blob.cstep;
        constants[6].i = n_fft;

        VkMat dispatcher;
        dispatcher.w = frames;
        dispatcher.h = (n_fft + 3) / 4;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_idft, bindings, constants, dispatcher);
    }

    int out_elempack = (opt.use_packing_layout && (outsize % 4 == 0)) ? 4 : 1;
    size_t out_elemsize = scalar_elemsize * out_elempack;

    if (returns == 0)
    {
        top_blob.create(2, outsize / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }
    else
    {
        top_blob.create(outsize / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    Pipeline* pipeline_ola = out_elempack == 4 ? pipeline_inversespectrogram_ola_pack4 : pipeline_inversespectrogram_ola;

    {
        std::vector<VkMat> bindings(5);
        bindings[0] = yre;
        bindings[1] = yim;
        bindings[2] = window2_data_gpu;
        bindings[3] = top_blob;
        bindings[4] = top_blob;

        std::vector<vk_constant_type> constants(6);
        constants[0].i = frames;
        constants[1].i = n_fft;
        constants[2].i = hoplen;
        constants[3].i = outsize;
        constants[4].i = center;
        constants[5].i = n_fft;

        VkMat dispatcher;
        dispatcher.w = (outsize + 3) / 4;
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_ola, bindings, constants, dispatcher);
    }

    return 0;
}

} // namespace ncnn
