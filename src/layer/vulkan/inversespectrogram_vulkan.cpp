// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "inversespectrogram_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

InverseSpectrogram_vulkan::InverseSpectrogram_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_any_packing = false;

    gemm_real = 0;
    gemm_imag = 0;

    pipeline_inversespectrogram_build_b = 0;
    pipeline_inversespectrogram_ola = 0;
}

int InverseSpectrogram_vulkan::create_pipeline(const Option& opt)
{
    const int K = 2 * n_fft;

    Mat basis_data_real;
    Mat basis_data_imag;
    basis_data_real.create(K, n_fft, (size_t)4u, 1);
    basis_data_imag.create(K, n_fft, (size_t)4u, 1);
    if (basis_data_real.empty() || basis_data_imag.empty())
        return -100;

    window2_data.create(n_fft, (size_t)4u, 1);
    if (window2_data.empty())
        return -100;

    for (int i = 0; i < n_fft; i++)
    {
        float* real_row = basis_data_real.row<float>(i);
        float* imag_row = basis_data_imag.row<float>(i);

        const float w = window_data[i];
        const float scale = w / n_fft;

        window2_data[i] = w * w;

        for (int k = 0; k < n_fft; k++)
        {
            const double angle = 2 * 3.14159265358979323846 * i * k / n_fft;

            const float c = (float)cos(angle) * scale;
            const float s = (float)sin(angle) * scale;

            real_row[k] = c;
            real_row[k + n_fft] = -s;

            imag_row[k] = s;
            imag_row[k + n_fft] = c;
        }
    }

    gemm_real = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
    gemm_real->vkdev = vkdev;
    {
        ncnn::ParamDict pd;

        pd.set(0, 1.f);   // alpha
        pd.set(1, 1.f);   // beta
        pd.set(2, 0);     // transA
        pd.set(3, 1);     // transB (B is BT layout already)
        pd.set(4, 1);     // constantA
        pd.set(5, 0);     // constantB
        pd.set(6, 0);     // constantC
        pd.set(7, n_fft); // constantM
        pd.set(8, 0);     // constantN
        pd.set(9, K);     // constantK
        pd.set(14, 0);    // output_transpose

        gemm_real->load_param(pd);

        Mat weights[1];
        weights[0] = basis_data_real;
        gemm_real->load_model(ModelBinFromMatArray(weights));
        gemm_real->create_pipeline(opt);
    }

    gemm_imag = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
    gemm_imag->vkdev = vkdev;
    {
        ncnn::ParamDict pd;

        pd.set(0, 1.f);   // alpha
        pd.set(1, 1.f);   // beta
        pd.set(2, 0);     // transA
        pd.set(3, 1);     // transB (B is BT layout already)
        pd.set(4, 1);     // constantA
        pd.set(5, 0);     // constantB
        pd.set(6, 0);     // constantC
        pd.set(7, n_fft); // constantM
        pd.set(8, 0);     // constantN
        pd.set(9, K);     // constantK
        pd.set(14, 0);    // output_transpose

        gemm_imag->load_param(pd);

        Mat weights[1];
        weights[0] = basis_data_imag;
        gemm_imag->load_model(ModelBinFromMatArray(weights));
        gemm_imag->create_pipeline(opt);
    }

    pipeline_inversespectrogram_build_b = new Pipeline(vkdev);
    pipeline_inversespectrogram_build_b->set_local_size_xyz(8, 8, 1);
    pipeline_inversespectrogram_build_b->create(LayerShaderType::inversespectrogram_build_b, opt, std::vector<vk_specialization_type>());

    pipeline_inversespectrogram_ola = new Pipeline(vkdev);
    pipeline_inversespectrogram_ola->set_local_size_xyz(256, 1, 1);
    pipeline_inversespectrogram_ola->create(LayerShaderType::inversespectrogram_ola, opt, std::vector<vk_specialization_type>());

    if (opt.lightmode)
    {
        window_data.release();
    }

    return 0;
}

int InverseSpectrogram_vulkan::destroy_pipeline(const Option& opt)
{
    if (gemm_real)
    {
        gemm_real->destroy_pipeline(opt);
        delete gemm_real;
        gemm_real = 0;
    }

    if (gemm_imag)
    {
        gemm_imag->destroy_pipeline(opt);
        delete gemm_imag;
        gemm_imag = 0;
    }

    delete pipeline_inversespectrogram_build_b;
    pipeline_inversespectrogram_build_b = 0;

    delete pipeline_inversespectrogram_ola;
    pipeline_inversespectrogram_ola = 0;

    window2_data.release();
    window2_data_gpu.release();

    return 0;
}

int InverseSpectrogram_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (gemm_real) gemm_real->upload_model(cmd, opt);
    if (gemm_imag) gemm_imag->upload_model(cmd, opt);

    if (!window2_data.empty())
    {
        cmd.record_upload(window2_data, window2_data_gpu, opt);
    }

    return 0;
}

int InverseSpectrogram_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int frames = bottom_blob.h;
    const int freqs = bottom_blob.c;

    const int freqs_onesided = n_fft / 2 + 1;
    const int onesided = (freqs == freqs_onesided) ? 1 : 0;

    if (frames <= 0)
        return -100;

    const int outsize = center ? (frames - 1) * hoplen + (n_fft & 1) : (frames - 1) * hoplen + n_fft;
    if (outsize <= 0)
        return -100;

    const int K = 2 * n_fft;

    VkMat B;
    B.create(K, frames, bottom_blob.elemsize, 1, opt.workspace_vkallocator);
    if (B.empty())
        return -100;

    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob;
        bindings[1] = B;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = frames;
        constants[1].i = freqs;
        constants[2].i = n_fft;
        constants[3].i = (int)bottom_blob.cstep;
        constants[4].i = onesided;

        VkMat dispatcher;
        dispatcher.w = n_fft;
        dispatcher.h = frames;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_inversespectrogram_build_b, bindings, constants, dispatcher);
    }

    VkMat yre4;
    VkMat yim4;
    {
        std::vector<VkMat> inputs(1);
        inputs[0] = B;

        std::vector<VkMat> outputs(1);
        outputs[0] = VkMat();

        int retr = gemm_real->forward(inputs, outputs, cmd, opt);
        if (retr != 0)
            return retr;

        yre4 = outputs[0];
    }
    {
        std::vector<VkMat> inputs(1);
        inputs[0] = B;

        std::vector<VkMat> outputs(1);
        outputs[0] = VkMat();

        int reti = gemm_imag->forward(inputs, outputs, cmd, opt);
        if (reti != 0)
            return reti;

        yim4 = outputs[0];
    }

    VkMat yre;
    VkMat yim;
    vkdev->convert_packing(yre4, yre, 1, cmd, opt);
    vkdev->convert_packing(yim4, yim, 1, cmd, opt);

    if (returns == 0)
    {
        top_blob.create(2, outsize, bottom_blob.elemsize, 1, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }
    else
    {
        top_blob.create(outsize, bottom_blob.elemsize, 1, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    {
        std::vector<VkMat> bindings(4);
        bindings[0] = yre;
        bindings[1] = yim;
        bindings[2] = window2_data_gpu;
        bindings[3] = top_blob;

        std::vector<vk_constant_type> constants(7);
        constants[0].i = frames;
        constants[1].i = n_fft;
        constants[2].i = hoplen;
        constants[3].i = outsize;
        constants[4].i = center;
        constants[5].i = returns;
        constants[6].i = yre.w;

        VkMat dispatcher;
        dispatcher.w = outsize;
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_inversespectrogram_ola, bindings, constants, dispatcher);
    }

    return 0;
}

} // namespace ncnn
