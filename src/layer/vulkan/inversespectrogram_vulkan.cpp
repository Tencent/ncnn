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
    pipeline_inversespectrogram_idft_packed_gemm = 0;
    pipeline_inversespectrogram_idft_gemm_cm = 0;
    pipeline_inversespectrogram_ola = 0;
    pipeline_inversespectrogram_ola_pack4 = 0;

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
        int freqs_max = n_fft;

        bool use_gemm = opt.use_sgemm_convolution
                        && n_fft >= 8
                        && freqs_max >= 8;

        if (use_gemm)
        {
            use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && opt.use_fp16_arithmetic;

            if (use_cooperative_matrix)
            {
                int size = 1024;

                vkdev->info.get_optimal_cooperative_matrix_mnk(size, n_fft, freqs_max, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);

                UNROLL_SG_M = std::min((size + coopmat_M - 1) / coopmat_M, 2);
                UNROLL_SG_N = std::min((n_fft + coopmat_N - 1) / coopmat_N, 2);
                UNROLL_SG_K = std::min((freqs_max + coopmat_K - 1) / coopmat_K, 2);

                UNROLL_WG_M = std::min((size + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
                UNROLL_WG_N = std::min((n_fft + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

                const int blocks_n = (n_fft + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
                const int kk = (freqs_max + coopmat_K - 1) / coopmat_K;

                const int weight_packed_dim0 = coopmat_N * coopmat_K * UNROLL_SG_N * UNROLL_WG_N * kk;

                basis_cos_data_gemm_packed.create(weight_packed_dim0, blocks_n);
                basis_sin_data_gemm_packed.create(weight_packed_dim0, blocks_n);

                for (int bn = 0; bn < blocks_n; bn++)
                {
                    float* pc = basis_cos_data_gemm_packed.row(bn);
                    float* ps = basis_sin_data_gemm_packed.row(bn);

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

                                            if (gni < n_fft && gki < freqs_max)
                                            {
                                                const double angle = 2 * 3.14159265358979323846 * gki * gni / n_fft;
                                                const float scale = window_data[gni] / n_fft;

                                                *pc++ = (float)cos(angle) * scale;
                                                *ps++ = (float)sin(angle) * scale;
                                            }
                                            else
                                            {
                                                *pc++ = 0.f;
                                                *ps++ = 0.f;
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

                                        if (gni < n_fft && gki < freqs_max)
                                        {
                                            const double angle = 2 * 3.14159265358979323846 * gki * gni / n_fft;
                                            const float scale = window_data[gni] / n_fft;

                                            *pc++ = (float)cos(angle) * scale;
                                            *ps++ = (float)sin(angle) * scale;
                                        }
                                        else
                                        {
                                            *pc++ = 0.f;
                                            *ps++ = 0.f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                std::vector<vk_specialization_type> specializations(14 + 4);
                specializations[0].u32 = n_fft;
                specializations[1].u32 = freqs_max;
                specializations[2].u32 = coopmat_M;
                specializations[3].u32 = coopmat_N;
                specializations[4].u32 = coopmat_K;
                specializations[5].u32 = coopmat_subgroup_size;
                specializations[6].u32 = UNROLL_SG_M;
                specializations[7].u32 = UNROLL_SG_N;
                specializations[8].u32 = UNROLL_SG_K;
                specializations[9].u32 = UNROLL_WG_M;
                specializations[10].u32 = UNROLL_WG_N;
                specializations[11].u32 = size;
                specializations[12].u32 = n_fft;
                specializations[13].u32 = 1;
                specializations[14 + 0].u32 = 0;
                specializations[14 + 1].u32 = 0;
                specializations[14 + 2].u32 = 0;
                specializations[14 + 3].u32 = 0;

                pipeline_inversespectrogram_idft_gemm_cm = new Pipeline(vkdev);
                pipeline_inversespectrogram_idft_gemm_cm->set_subgroup_size(coopmat_subgroup_size);
                pipeline_inversespectrogram_idft_gemm_cm->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
                pipeline_inversespectrogram_idft_gemm_cm->create(LayerShaderType::inversespectrogram_idft_gemm_cm, opt, specializations);
            }
            else
            {
                const int freqs_packed = (freqs_max + 3) / 4 * 4;
                const int n_fft_packed = (n_fft + 3) / 4 * 4;

                basis_cos_data_gemm_packed.create(freqs_packed / 4, n_fft_packed / 4, (size_t)4 * 4 * 4, 4 * 4);
                basis_sin_data_gemm_packed.create(freqs_packed / 4, n_fft_packed / 4, (size_t)4 * 4 * 4, 4 * 4);

                for (int q = 0; q < n_fft_packed; q += 4)
                {
                    float* g00c = basis_cos_data_gemm_packed.row(q / 4);
                    float* g00s = basis_sin_data_gemm_packed.row(q / 4);

                    for (int p = 0; p < freqs_packed; p += 4)
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            for (int j = 0; j < 4; j++)
                            {
                                const int t = q + i;
                                const int k = p + j;

                                if (t < n_fft && k < freqs_max)
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
                }

                const int outh_pack4 = n_fft_packed / 4;

                std::vector<vk_specialization_type> specializations(4 + 4);
                specializations[0].i = n_fft;
                specializations[1].i = freqs_max;
                specializations[2].i = 1;
                specializations[3].i = 0;
                specializations[4 + 0].i = 0;
                specializations[4 + 1].i = n_fft;
                specializations[4 + 2].i = 0;
                specializations[4 + 3].i = outh_pack4;

                pipeline_inversespectrogram_idft_packed_gemm = new Pipeline(vkdev);
                if (opt.use_shader_local_memory)
                {
                    pipeline_inversespectrogram_idft_packed_gemm->set_local_size_xyz(8, 8, 1);
                }
                else
                {
                    pipeline_inversespectrogram_idft_packed_gemm->set_local_size_xyz(16, std::min(4, outh_pack4), 1);
                }
                pipeline_inversespectrogram_idft_packed_gemm->create(LayerShaderType::inversespectrogram_idft_packed_gemm, opt, specializations);
            }
        }
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

    delete pipeline_inversespectrogram_idft_packed_gemm;
    pipeline_inversespectrogram_idft_packed_gemm = 0;

    delete pipeline_inversespectrogram_idft_gemm_cm;
    pipeline_inversespectrogram_idft_gemm_cm = 0;

    delete pipeline_inversespectrogram_ola;
    pipeline_inversespectrogram_ola = 0;

    delete pipeline_inversespectrogram_ola_pack4;
    pipeline_inversespectrogram_ola_pack4 = 0;

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

    window2_data.release();
    window2_data_gpu.release();
    basis_cos_data_gemm_packed.release();
    basis_sin_data_gemm_packed.release();
    basis_cos_data_gemm_gpu.release();
    basis_sin_data_gemm_gpu.release();

    return 0;
}

int InverseSpectrogram_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(basis_cos_data_packed, basis_cos_data_gpu, opt);
    cmd.record_upload(basis_sin_data_packed, basis_sin_data_gpu, opt);

    if (!basis_cos_data_gemm_packed.empty())
    {
        cmd.record_upload(basis_cos_data_gemm_packed, basis_cos_data_gemm_gpu, opt);
        cmd.record_upload(basis_sin_data_gemm_packed, basis_sin_data_gemm_gpu, opt);
    }

    basis_cos_data_packed.release();
    basis_sin_data_packed.release();
    basis_cos_data_gemm_packed.release();
    basis_sin_data_gemm_packed.release();

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

    if (pipeline_inversespectrogram_idft_packed_gemm)
    {
        std::vector<VkMat> bindings(6);
        bindings[0] = bottom_blob;
        bindings[1] = yre;
        bindings[2] = bottom_blob;
        bindings[3] = yim;
        bindings[4] = basis_cos_data_gemm_gpu;
        bindings[5] = basis_sin_data_gemm_gpu;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = frames;
        constants[1].i = n_fft;
        constants[2].i = freqs;
        constants[3].i = bottom_blob.w;
        constants[4].i = bottom_blob.elempack;
        constants[5].i = (int)bottom_blob.cstep;
        constants[6].i = n_fft;
        constants[7].i = (n_fft + 3) / 4;

        VkMat dispatcher;
        dispatcher.w = (frames + 3) / 4;
        dispatcher.h = (n_fft + 3) / 4;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_inversespectrogram_idft_packed_gemm, bindings, constants, dispatcher);
    }
    else if (pipeline_inversespectrogram_idft_gemm_cm && use_cooperative_matrix && freqs == n_fft)
    {
        VkMat workspace_a;
        VkMat workspace_b;
        VkMat workspace_c;
        VkMat workspace_d;
        workspace_a.create(frames, n_fft, scalar_elemsize, 1, opt.workspace_vkallocator);
        workspace_b.create(frames, n_fft, scalar_elemsize, 1, opt.workspace_vkallocator);
        workspace_c.create(frames, n_fft, scalar_elemsize, 1, opt.workspace_vkallocator);
        workspace_d.create(frames, n_fft, scalar_elemsize, 1, opt.workspace_vkallocator);
        if (workspace_a.empty() || workspace_b.empty() || workspace_c.empty() || workspace_d.empty())
            return -100;

        const int blocks_x = (frames + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
        const int blocks_y = (n_fft + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);
        VkMat dispatcher;
        dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
        dispatcher.h = 1;
        dispatcher.c = 1;

        // A = sp_re × basis_cos
        {
            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob;
            bindings[1] = workspace_a;
            bindings[2] = basis_cos_data_gemm_gpu;
            bindings[3] = bindings[1];
            bindings[4] = bottom_blob;
            bindings[5] = workspace_a;

            std::vector<vk_constant_type> constants(5);
            constants[0].u32 = frames;
            constants[1].u32 = (unsigned int)bottom_blob.cstep;
            constants[2].u32 = frames;
            constants[3].u32 = frames;
            constants[4].u32 = 0;

            cmd.record_pipeline(pipeline_inversespectrogram_idft_gemm_cm, bindings, constants, dispatcher);
        }

        // B = sp_im × basis_sin (using basis_sin weight)
        {
            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob;
            bindings[1] = workspace_b;
            bindings[2] = basis_sin_data_gemm_gpu;
            bindings[3] = bindings[1];
            bindings[4] = bottom_blob;
            bindings[5] = workspace_b;

            std::vector<vk_constant_type> constants(5);
            constants[0].u32 = frames;
            constants[1].u32 = (unsigned int)bottom_blob.cstep;
            constants[2].u32 = frames;
            constants[3].u32 = frames;
            constants[4].u32 = 1;

            cmd.record_pipeline(pipeline_inversespectrogram_idft_gemm_cm, bindings, constants, dispatcher);
        }

        // C = sp_re × basis_sin
        {
            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob;
            bindings[1] = workspace_c;
            bindings[2] = basis_sin_data_gemm_gpu;
            bindings[3] = bindings[1];
            bindings[4] = bottom_blob;
            bindings[5] = workspace_c;

            std::vector<vk_constant_type> constants(5);
            constants[0].u32 = frames;
            constants[1].u32 = (unsigned int)bottom_blob.cstep;
            constants[2].u32 = frames;
            constants[3].u32 = frames;
            constants[4].u32 = 0;

            cmd.record_pipeline(pipeline_inversespectrogram_idft_gemm_cm, bindings, constants, dispatcher);
        }

        // D = sp_im × basis_cos
        {
            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob;
            bindings[1] = workspace_d;
            bindings[2] = basis_cos_data_gemm_gpu;
            bindings[3] = bindings[1];
            bindings[4] = bottom_blob;
            bindings[5] = workspace_d;

            std::vector<vk_constant_type> constants(5);
            constants[0].u32 = frames;
            constants[1].u32 = (unsigned int)bottom_blob.cstep;
            constants[2].u32 = frames;
            constants[3].u32 = frames;
            constants[4].u32 = 1;

            cmd.record_pipeline(pipeline_inversespectrogram_idft_gemm_cm, bindings, constants, dispatcher);
        }

        Mat a_cpu;
        Mat b_cpu;
        Mat c_cpu;
        Mat d_cpu;
        cmd.record_download(workspace_a, a_cpu, opt);
        cmd.record_download(workspace_b, b_cpu, opt);
        cmd.record_download(workspace_c, c_cpu, opt);
        cmd.record_download(workspace_d, d_cpu, opt);

        cmd.submit_and_wait();

        Mat a_cpu_fp32 = a_cpu;
        Mat b_cpu_fp32 = b_cpu;
        Mat c_cpu_fp32 = c_cpu;
        Mat d_cpu_fp32 = d_cpu;
        if (a_cpu.elembits() == 16)
        {
            cast_float16_to_float32(a_cpu, a_cpu_fp32, opt);
            cast_float16_to_float32(b_cpu, b_cpu_fp32, opt);
            cast_float16_to_float32(c_cpu, c_cpu_fp32, opt);
            cast_float16_to_float32(d_cpu, d_cpu_fp32, opt);
        }

        float* a_ptr = (float*)a_cpu_fp32.data;
        float* b_ptr = (float*)b_cpu_fp32.data;
        float* c_ptr = (float*)c_cpu_fp32.data;
        float* d_ptr = (float*)d_cpu_fp32.data;

        Mat re_cpu(n_fft, frames, (size_t)4);
        Mat im_cpu(n_fft, frames, (size_t)4);
        float* re_ptr = (float*)re_cpu.data;
        float* im_ptr = (float*)im_cpu.data;

        for (int j = 0; j < frames; j++)
        {
            for (int i = 0; i < n_fft; i++)
            {
                int dst_idx = j * n_fft + i;
                int src_idx = i * frames + j;
                re_ptr[dst_idx] = a_ptr[src_idx] - b_ptr[src_idx];
                im_ptr[dst_idx] = c_ptr[src_idx] + d_ptr[src_idx];
            }
        }

        VkTransfer upload_cmd(vkdev);
        upload_cmd.record_upload(re_cpu, yre, opt, false);
        upload_cmd.record_upload(im_cpu, yim, opt, false);
        upload_cmd.submit_and_wait();
    }
    else
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
