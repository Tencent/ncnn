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
    pipeline_spectrogram_packed_gemm = 0;
    pipeline_spectrogram_gemm_cm = 0;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;
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

    bool use_gemm = opt.use_sgemm_convolution
                    && n_fft >= 8
                    && n_freq >= 8;

    if (use_gemm)
    {
        use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && opt.use_fp16_arithmetic;

        if (use_cooperative_matrix)
        {
            int size = 1024;

            vkdev->info.get_optimal_cooperative_matrix_mnk(size, n_freq, n_fft, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);

            UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
            UNROLL_SG_N = std::min((n_freq + coopmat_N - 1) / coopmat_N, 2);
            UNROLL_SG_K = std::min((n_fft + coopmat_K - 1) / coopmat_K, 2);

            UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
            UNROLL_WG_N = std::min((n_freq + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

            const int blocks_n = (n_freq + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
            const int kk = (n_fft + coopmat_K - 1) / coopmat_K;

            const int weight_packed_dim0 = coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk;

            basis_data_packed.create(weight_packed_dim0, blocks_n);
            basis_imag_data_packed.create(weight_packed_dim0, blocks_n);

            for (int bn = 0; bn < blocks_n; bn++)
            {
                float* pr = basis_data_packed.row(bn);
                float* pi = basis_imag_data_packed.row(bn);

                int k = 0;
                for (; k + UNROLL_SG_K - 1 < kk; k += UNROLL_SG_K)
                {
                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zk = 0; zk < UNROLL_SG_K; zk++)
                        {
                            for (int zn = 0; zn < UNROLL_SG_N; zn++)
                            {
                                for (int i = 0; i < coopmat_K; i++)
                                {
                                    for (int j = 0; j < coopmat_N; j++)
                                    {
                                        const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                        const int gki = (k + zk) * coopmat_K + i;

                                        if (gni < n_freq && gki < n_fft)
                                        {
                                            const double angle = 2 * 3.14159265358979323846 * gni * gki / n_fft;
                                            const float w = window_data[gki];

                                            *pr++ = (float)cos(angle) * w;
                                            *pi++ = (float)(-sin(angle)) * w;
                                        }
                                        else
                                        {
                                            *pr++ = 0.f;
                                            *pi++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (; k < kk; k++)
                {
                    for (int wn = 0; wn < UNROLL_WG_N; wn++)
                    {
                        for (int zn = 0; zn < UNROLL_SG_N; zn++)
                        {
                            for (int i = 0; i < coopmat_K; i++)
                            {
                                for (int j = 0; j < coopmat_N; j++)
                                {
                                    const int gni = ((bn * UNROLL_WG_N + wn) * UNROLL_SG_N + zn) * coopmat_N + j;
                                    const int gki = k * coopmat_K + i;

                                    if (gni < n_freq && gki < n_fft)
                                    {
                                        const double angle = 2 * 3.14159265358979323846 * gni * gki / n_fft;
                                        const float w = window_data[gki];

                                        *pr++ = (float)cos(angle) * w;
                                        *pi++ = (float)(-sin(angle)) * w;
                                    }
                                    else
                                    {
                                        *pr++ = 0.f;
                                        *pi++ = 0.f;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            std::vector<vk_specialization_type> specializations(17 + 4);
            specializations[0].u32 = n_fft;
            specializations[1].u32 = hoplen;
            specializations[2].i = power;
            specializations[3].i = center;
            specializations[4].i = pad_type;
            specializations[5].u32 = coopmat_M;
            specializations[6].u32 = coopmat_N;
            specializations[7].u32 = coopmat_K;
            specializations[8].u32 = coopmat_subgroup_size;
            specializations[9].u32 = UNROLL_SG_M;
            specializations[10].u32 = UNROLL_SG_N;
            specializations[11].u32 = UNROLL_SG_K;
            specializations[12].u32 = UNROLL_WG_M;
            specializations[13].u32 = UNROLL_WG_N;
            specializations[14].u32 = size;
            specializations[15].u32 = n_freq;
            specializations[16].u32 = out_elempack;
            specializations[17 + 0].u32 = 0;
            specializations[17 + 1].u32 = 0;
            specializations[17 + 2].u32 = 0;
            specializations[17 + 3].u32 = 0;

            pipeline_spectrogram_gemm_cm = new Pipeline(vkdev);
            pipeline_spectrogram_gemm_cm->set_subgroup_size(coopmat_subgroup_size);
            pipeline_spectrogram_gemm_cm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_spectrogram_gemm_cm->create(LayerShaderType::spectrogram_gemm_cm, opt, specializations);
        }
        else
        {
            const int n_fft_packed = (n_fft + 3) / 4 * 4;
            const int n_freq_packed = (n_freq + 3) / 4 * 4;

            basis_data_packed.create(n_fft_packed / 4, n_freq_packed / 4, (size_t)4 * 4 * 4, 4 * 4);
            basis_imag_data_packed.create(n_fft_packed / 4, n_freq_packed / 4, (size_t)4 * 4 * 4, 4 * 4);

            for (int q = 0; q < n_freq_packed; q += 4)
            {
                float* g00r = basis_data_packed.row(q / 4);
                float* g00i = basis_imag_data_packed.row(q / 4);

                for (int p = 0; p < n_fft_packed; p += 4)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            const int f = q + i;
                            const int k = p + j;

                            if (f < n_freq && k < n_fft)
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
            }

            const int outh_pack4 = n_freq_packed / 4;

            std::vector<vk_specialization_type> specializations(6 + 4);
            specializations[0].i = power;
            specializations[1].i = n_fft;
            specializations[2].i = hoplen;
            specializations[3].i = out_elempack;
            specializations[4].i = center;
            specializations[5].i = pad_type;
            specializations[6 + 0].i = 0;
            specializations[6 + 1].i = n_freq;
            specializations[6 + 2].i = outh_pack4;
            specializations[6 + 3].i = 0;

            pipeline_spectrogram_packed_gemm = new Pipeline(vkdev);
            if (opt.use_shader_local_memory)
            {
                pipeline_spectrogram_packed_gemm->set_local_size_xyz(8, 8, 1);
            }
            else
            {
                pipeline_spectrogram_packed_gemm->set_local_size_xyz(16, std::min(4, outh_pack4), 1);
            }
            pipeline_spectrogram_packed_gemm->create(LayerShaderType::spectrogram_packed_gemm, opt, specializations);
        }
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

    delete pipeline_spectrogram_packed_gemm;
    pipeline_spectrogram_packed_gemm = 0;

    delete pipeline_spectrogram_gemm_cm;
    pipeline_spectrogram_gemm_cm = 0;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;

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

    if (pipeline_spectrogram_gemm_cm && use_cooperative_matrix)
    {
        std::vector<VkMat> bindings(6);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;
        bindings[2] = bottom_blob;
        bindings[3] = top_blob;
        bindings[4] = basis_data_gpu;
        bindings[5] = basis_imag_data_gpu;

        std::vector<vk_constant_type> constants(4);
        constants[0].u32 = size;
        constants[1].u32 = size;
        constants[2].u32 = frames;
        constants[3].u32 = top_row_stride;

        const int blocks_x = (frames + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
        const int blocks_y = (n_freq + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

        VkMat dispatcher;
        dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
        dispatcher.h = 1;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_spectrogram_gemm_cm, bindings, constants, dispatcher);

        return 0;
    }

    if (pipeline_spectrogram_packed_gemm)
    {
        const int n_freq_packed = (n_freq + 3) / 4 * 4;

        std::vector<VkMat> bindings(6);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;
        bindings[2] = bottom_blob;
        bindings[3] = top_blob;
        bindings[4] = basis_data_gpu;
        bindings[5] = basis_imag_data_gpu;

        std::vector<vk_constant_type> constants(7);
        constants[0].i = frames;
        constants[1].i = n_freq;
        constants[2].i = size;
        constants[3].i = top_row_stride;
        constants[4].i = center;
        constants[5].i = pad_type;
        constants[6].i = n_freq_packed / 4;

        VkMat dispatcher;
        dispatcher.w = (frames + 3) / 4;
        dispatcher.h = n_freq_packed / 4;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_spectrogram_packed_gemm, bindings, constants, dispatcher);

        return 0;
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
