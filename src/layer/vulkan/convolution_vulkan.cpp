// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "convolution_vulkan.h"
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Convolution_vulkan)

Convolution_vulkan::Convolution_vulkan()
{
    support_vulkan = true;

    padding = 0;

    pipeline_convolution = 0;
    pipeline_convolution_1x1s1d1 = 0;
    pipeline_convolution_pack4 = 0;
    pipeline_convolution_pack4_1x1s1d1 = 0;
    pipeline_convolution_pack4_3x3s1d1_lds_8_8_2 = 0;
    winograd_padding = 0;
    winograd_crop = 0;
    pipeline_convolution_pack4_3x3s1d1_winograd23_transform_input = 0;
    pipeline_convolution_pack4_3x3s1d1_winograd23_gemm = 0;
    pipeline_convolution_pack4_3x3s1d1_winograd23_transform_output = 0;
    pipeline_convolution_pack1to4 = 0;
    pipeline_convolution_pack4to1 = 0;

    pipeline_innerproduct = 0;
    pipeline_innerproduct_pack4 = 0;
    pipeline_innerproduct_pack1to4 = 0;
    pipeline_innerproduct_pack4to1 = 0;
}

int Convolution_vulkan::create_pipeline(const Option& opt)
{
    {
        padding = ncnn::create_layer(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, pad_top);
        pd.set(1, pad_bottom);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, pad_value);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    std::vector<vk_specialization_type> specializations(10);
    specializations[0].i = kernel_w;
    specializations[1].i = kernel_h;
    specializations[2].i = dilation_w;
    specializations[3].i = dilation_h;
    specializations[4].i = stride_w;
    specializations[5].i = stride_h;
    specializations[6].i = bias_term;
    specializations[7].i = activation_type;
    specializations[8].f = activation_params.w == 1 ? activation_params[0] : 0.f;
    specializations[9].f = activation_params.w == 2 ? activation_params[1] : 0.f;

    // pack1
    if (num_input % 4 != 0 && num_output % 4 != 0)
    {
        pipeline_convolution = new Pipeline(vkdev);
        pipeline_convolution->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolution->create("convolution", opt, specializations, 4, 10);

        if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            pipeline_convolution_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution_1x1s1d1->set_optimal_local_size_xyz(-1, 1, std::max(1, num_output / 8));

            std::vector<vk_specialization_type> specializations(4);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w == 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;

            pipeline_convolution_1x1s1d1->create("convolution_1x1s1d1", opt, specializations, 4, 8);
        }
    }

    // pack4
    if (num_input % 4 == 0 && num_output % 4 == 0)
    {
        pipeline_convolution_pack4 = new Pipeline(vkdev);
        pipeline_convolution_pack4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolution_pack4->create("convolution_pack4", opt, specializations, 4, 10);

        if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            pipeline_convolution_pack4_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution_pack4_1x1s1d1->set_local_size_xyz(8, 1, std::min(8, num_output / 4));

            std::vector<vk_specialization_type> specializations(4);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w == 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;

            pipeline_convolution_pack4_1x1s1d1->create("convolution_pack4_1x1s1d1", opt, specializations, 4, 8);
        }

        if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            std::vector<vk_specialization_type> specializations(4);
            specializations[0].i = bias_term;
            specializations[1].i = activation_type;
            specializations[2].f = activation_params.w == 1 ? activation_params[0] : 0.f;
            specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;

            pipeline_convolution_pack4_3x3s1d1_lds_8_8_2 = new Pipeline(vkdev);
            pipeline_convolution_pack4_3x3s1d1_lds_8_8_2->set_local_size_xyz(8, 8, 2);
            pipeline_convolution_pack4_3x3s1d1_lds_8_8_2->create("convolution_pack4_3x3s1d1_lds_8_8_2", opt, specializations, 4, 10);

            if (num_input >= 16 && num_output >= 16)
            {
                {
                    winograd_padding = ncnn::create_layer(ncnn::LayerType::Padding);
                    winograd_padding->vkdev = vkdev;

                    ncnn::ParamDict pd;
                    pd.set(0, -233);
                    pd.set(1, -233);
                    pd.set(2, -233);
                    pd.set(3, -233);
                    pd.set(4, 0);
                    pd.set(5, 0.f);

                    winograd_padding->load_param(pd);

                    winograd_padding->create_pipeline(opt);
                }

                {
                    winograd_crop = ncnn::create_layer(ncnn::LayerType::Crop);
                    winograd_crop->vkdev = vkdev;

                    ncnn::ParamDict pd;
                    pd.set(0, -233);
                    pd.set(1, -233);
                    pd.set(2, -233);
                    pd.set(3, 0);
                    pd.set(4, 0);
                    pd.set(5, 0);

                    winograd_crop->load_param(pd);

                    winograd_crop->create_pipeline(opt);
                }
            }

            if (num_input >= 16 && num_output >= 16)
            {
                pipeline_convolution_pack4_3x3s1d1_winograd23_transform_input = new Pipeline(vkdev);
                pipeline_convolution_pack4_3x3s1d1_winograd23_transform_input->set_local_size_xyz(8, 8, 1);
                pipeline_convolution_pack4_3x3s1d1_winograd23_transform_input->create("convolution_pack4_3x3s1d1_winograd23_transform_input", opt, std::vector<vk_specialization_type>(), 2, 7);

                pipeline_convolution_pack4_3x3s1d1_winograd23_gemm = new Pipeline(vkdev);
                pipeline_convolution_pack4_3x3s1d1_winograd23_gemm->set_local_size_xyz(4, 4, 4);
                pipeline_convolution_pack4_3x3s1d1_winograd23_gemm->create("convolution_pack4_3x3s1d1_winograd23_gemm", opt, std::vector<vk_specialization_type>(), 3, 6);

                pipeline_convolution_pack4_3x3s1d1_winograd23_transform_output = new Pipeline(vkdev);
                pipeline_convolution_pack4_3x3s1d1_winograd23_transform_output->set_local_size_xyz(8, 8, 1);
                pipeline_convolution_pack4_3x3s1d1_winograd23_transform_output->create("convolution_pack4_3x3s1d1_winograd23_transform_output", opt, specializations, 3, 7);
            }
        }
    }

    // pack1to4
    if (num_input % 4 != 0 && num_output % 4 == 0)
    {
        pipeline_convolution_pack1to4 = new Pipeline(vkdev);
        pipeline_convolution_pack1to4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolution_pack1to4->create("convolution_pack1to4", opt, specializations, 4, 10);
    }

    // pack4to1
    if (num_input % 4 == 0 && num_output % 4 != 0)
    {
        pipeline_convolution_pack4to1 = new Pipeline(vkdev);
        pipeline_convolution_pack4to1->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolution_pack4to1->create("convolution_pack4to1", opt, specializations, 4, 10);
    }

    // fc
    if (kernel_w == 1 && kernel_h == 1)
    {
        std::vector<vk_specialization_type> specializations(4);
        specializations[0].i = bias_term;
        specializations[1].i = activation_type;
        specializations[2].f = activation_params.w == 1 ? activation_params[0] : 0.f;
        specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;

        // pack1
        if (num_input % 4 != 0 && num_output % 4 != 0)
        {
            pipeline_innerproduct = new Pipeline(vkdev);
            pipeline_innerproduct->set_optimal_local_size_xyz(num_output, 1, 1);
            pipeline_innerproduct->create("innerproduct", opt, specializations, 4, 10);
        }

        // pack4
        if (num_input % 4 == 0 && num_output % 4 == 0)
        {
            pipeline_innerproduct_pack4 = new Pipeline(vkdev);
            pipeline_innerproduct_pack4->set_optimal_local_size_xyz(num_output / 4, 1, 1);
            pipeline_innerproduct_pack4->create("innerproduct_pack4", opt, specializations, 4, 10);
        }

        // pack1to4
        if (num_input % 4 != 0 && num_output % 4 == 0)
        {
            pipeline_innerproduct_pack1to4 = new Pipeline(vkdev);
            pipeline_innerproduct_pack1to4->set_optimal_local_size_xyz(num_output / 4, 1, 1);
            pipeline_innerproduct_pack1to4->create("innerproduct_pack1to4", opt, specializations, 4, 10);
        }

        // pack4to1
        if (num_input % 4 == 0 && num_output % 4 != 0)
        {
            pipeline_innerproduct_pack4to1 = new Pipeline(vkdev);
            pipeline_innerproduct_pack4to1->set_optimal_local_size_xyz(num_output, 1, 1);
            pipeline_innerproduct_pack4to1->create("innerproduct_pack4to1", opt, specializations, 4, 10);
        }
    }

    return 0;
}

int Convolution_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_convolution;
    pipeline_convolution = 0;

    delete pipeline_convolution_1x1s1d1;
    pipeline_convolution_1x1s1d1 = 0;

    delete pipeline_convolution_pack4;
    pipeline_convolution_pack4 = 0;

    delete pipeline_convolution_pack4_1x1s1d1;
    pipeline_convolution_pack4_1x1s1d1 = 0;

    delete pipeline_convolution_pack4_3x3s1d1_lds_8_8_2;
    pipeline_convolution_pack4_3x3s1d1_lds_8_8_2 = 0;

    if (winograd_padding)
    {
        winograd_padding->destroy_pipeline(opt);
        delete winograd_padding;
        winograd_padding = 0;
    }

    if (winograd_crop)
    {
        winograd_crop->destroy_pipeline(opt);
        delete winograd_crop;
        winograd_crop = 0;
    }

    delete pipeline_convolution_pack4_3x3s1d1_winograd23_transform_input;
    delete pipeline_convolution_pack4_3x3s1d1_winograd23_gemm;
    delete pipeline_convolution_pack4_3x3s1d1_winograd23_transform_output;
    pipeline_convolution_pack4_3x3s1d1_winograd23_transform_input = 0;
    pipeline_convolution_pack4_3x3s1d1_winograd23_gemm = 0;
    pipeline_convolution_pack4_3x3s1d1_winograd23_transform_output = 0;

    delete pipeline_convolution_pack1to4;
    pipeline_convolution_pack1to4 = 0;

    delete pipeline_convolution_pack4to1;
    pipeline_convolution_pack4to1 = 0;

    // fc
    delete pipeline_innerproduct;
    pipeline_innerproduct = 0;

    delete pipeline_innerproduct_pack4;
    pipeline_innerproduct_pack4 = 0;

    delete pipeline_innerproduct_pack1to4;
    pipeline_innerproduct_pack1to4 = 0;

    delete pipeline_innerproduct_pack4to1;
    pipeline_innerproduct_pack4to1 = 0;

    return 0;
}

int Convolution_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    // pack1
    if (num_input % 4 != 0 && num_output % 4 != 0)
    {
        cmd.record_upload(weight_data, weight_data_gpu, opt);
    }

    // pack4
    if (num_input % 4 == 0 && num_output % 4 == 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-4b-kw-kh-inch/4a-outch/4b
        Mat weight_data_pack4;
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack4.create(maxk, num_input/4, num_output/4, (size_t)4*16, 16);

            for (int q=0; q+3<num_output; q+=4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q+1);
                const Mat k2 = weight_data_r2.channel(q+2);
                const Mat k3 = weight_data_r2.channel(q+3);

                Mat g0 = weight_data_pack4.channel(q/4);

                for (int p=0; p+3<num_input; p+=4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p+1);
                    const float* k02 = k0.row(p+2);
                    const float* k03 = k0.row(p+3);

                    const float* k10 = k1.row(p);
                    const float* k11 = k1.row(p+1);
                    const float* k12 = k1.row(p+2);
                    const float* k13 = k1.row(p+3);

                    const float* k20 = k2.row(p);
                    const float* k21 = k2.row(p+1);
                    const float* k22 = k2.row(p+2);
                    const float* k23 = k2.row(p+3);

                    const float* k30 = k3.row(p);
                    const float* k31 = k3.row(p+1);
                    const float* k32 = k3.row(p+2);
                    const float* k33 = k3.row(p+3);

                    float* g00 = g0.row(p/4);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k01[k];
                        g00[2] = k02[k];
                        g00[3] = k03[k];

                        g00[4] = k10[k];
                        g00[5] = k11[k];
                        g00[6] = k12[k];
                        g00[7] = k13[k];

                        g00[8] = k20[k];
                        g00[9] = k21[k];
                        g00[10] = k22[k];
                        g00[11] = k23[k];

                        g00[12] = k30[k];
                        g00[13] = k31[k];
                        g00[14] = k32[k];
                        g00[15] = k33[k];

                        g00 += 16;
                    }
                }
            }
        }

        cmd.record_upload(weight_data_pack4, weight_data_gpu_pack4, opt);

        bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
        if (is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
        {
            // winograd23 transform kernel
            Mat weight_data_tm;
            weight_data_tm.create(4*4, num_input, num_output);

            // G
            const float ktm[4][3] = {
                {   1.0f,     0.0f,     0.0f},
                { 1.0f/2,   1.0f/2,   1.0f/2},
                { 1.0f/2,  -1.0f/2,   1.0f/2},
                {   0.0f,     0.0f,     1.0f}
            };

            #pragma omp parallel for
            for (int p = 0; p<num_output; p++)
            {
                for (int q = 0; q<num_input; q++)
                {
                    const float* kernel0 = (const float*)weight_data + p*num_input * 9 + q * 9;
                    float* kernel_tm0 = weight_data_tm.channel(p).row(q);

                    // transform kernel
                    const float* k0 = kernel0;
                    const float* k1 = kernel0 + 3;
                    const float* k2 = kernel0 + 6;

                    // h
                    float tmp[4][3];
                    for (int i=0; i<4; i++)
                    {
                        tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                        tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                        tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                    }

                    // U
                    for (int j=0; j<4; j++)
                    {
                        float* tmpp = &tmp[j][0];

                        for (int i=0; i<4; i++)
                        {
                            kernel_tm0[j*4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }

            // src = 16-inch-outch
            // dst = 4a-4b-16-inch/4a-outch/4b
            Mat weight_data_pack4_tm;
            {
                weight_data_pack4_tm.create(16, num_input/4, num_output/4, (size_t)4*16, 16);

                for (int q=0; q+3<num_output; q+=4)
                {
                    const Mat k0 = weight_data_tm.channel(q);
                    const Mat k1 = weight_data_tm.channel(q+1);
                    const Mat k2 = weight_data_tm.channel(q+2);
                    const Mat k3 = weight_data_tm.channel(q+3);

                    Mat g0 = weight_data_pack4_tm.channel(q/4);

                    for (int p=0; p+3<num_input; p+=4)
                    {
                        const float* k00 = k0.row(p);
                        const float* k01 = k0.row(p+1);
                        const float* k02 = k0.row(p+2);
                        const float* k03 = k0.row(p+3);

                        const float* k10 = k1.row(p);
                        const float* k11 = k1.row(p+1);
                        const float* k12 = k1.row(p+2);
                        const float* k13 = k1.row(p+3);

                        const float* k20 = k2.row(p);
                        const float* k21 = k2.row(p+1);
                        const float* k22 = k2.row(p+2);
                        const float* k23 = k2.row(p+3);

                        const float* k30 = k3.row(p);
                        const float* k31 = k3.row(p+1);
                        const float* k32 = k3.row(p+2);
                        const float* k33 = k3.row(p+3);

                        float* g00 = g0.row(p/4);

                        for (int k=0; k<16; k++)
                        {
                            g00[0] = k00[k];
                            g00[1] = k01[k];
                            g00[2] = k02[k];
                            g00[3] = k03[k];

                            g00[4] = k10[k];
                            g00[5] = k11[k];
                            g00[6] = k12[k];
                            g00[7] = k13[k];

                            g00[8] = k20[k];
                            g00[9] = k21[k];
                            g00[10] = k22[k];
                            g00[11] = k23[k];

                            g00[12] = k30[k];
                            g00[13] = k31[k];
                            g00[14] = k32[k];
                            g00[15] = k33[k];

                            g00 += 16;
                        }
                    }
                }
            }

            cmd.record_upload(weight_data_pack4_tm, weight_data_gpu_pack4_tm, opt);
        }
    }

    // pack1to4
    if (num_input % 4 != 0 && num_output % 4 == 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4b-kw-kh-inch-outch/4b
        Mat weight_data_pack1to4;
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack1to4.create(maxk, num_input, num_output/4, (size_t)4*4, 4);

            for (int q=0; q+3<num_output; q+=4)
            {
                const Mat k0 = weight_data_r2.channel(q);
                const Mat k1 = weight_data_r2.channel(q+1);
                const Mat k2 = weight_data_r2.channel(q+2);
                const Mat k3 = weight_data_r2.channel(q+3);

                Mat g0 = weight_data_pack1to4.channel(q/4);

                for (int p=0; p<num_input; p++)
                {
                    const float* k00 = k0.row(p);
                    const float* k10 = k1.row(p);
                    const float* k20 = k2.row(p);
                    const float* k30 = k3.row(p);

                    float* g00 = g0.row(p);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k10[k];
                        g00[2] = k20[k];
                        g00[3] = k30[k];

                        g00 += 4;
                    }
                }
            }
        }

        cmd.record_upload(weight_data_pack1to4, weight_data_gpu_pack1to4, opt);
    }

    // pack4to1
    if (num_input % 4 == 0 && num_output % 4 != 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-kw-kh-inch/4a-outch
        Mat weight_data_pack4to1;
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            weight_data_pack4to1.create(maxk, num_input/4, num_output, (size_t)4*4, 4);

            for (int q=0; q<num_output; q++)
            {
                const Mat k0 = weight_data_r2.channel(q);
                Mat g0 = weight_data_pack4to1.channel(q);

                for (int p=0; p+3<num_input; p+=4)
                {
                    const float* k00 = k0.row(p);
                    const float* k01 = k0.row(p+1);
                    const float* k02 = k0.row(p+2);
                    const float* k03 = k0.row(p+3);

                    float* g00 = g0.row(p/4);

                    for (int k=0; k<maxk; k++)
                    {
                        g00[0] = k00[k];
                        g00[1] = k01[k];
                        g00[2] = k02[k];
                        g00[3] = k03[k];

                        g00 += 4;
                    }
                }
            }
        }

        cmd.record_upload(weight_data_pack4to1, weight_data_gpu_pack4to1, opt);
    }

    if (bias_term)
    {
        if (num_output % 4 != 0)
        {
            cmd.record_upload(bias_data, bias_data_gpu, opt);
        }

        if (num_output % 4 == 0)
        {
            Mat bias_data_pack4;
            convert_packing(bias_data, bias_data_pack4, 4);
            cmd.record_upload(bias_data_pack4, bias_data_gpu_pack4, opt);
        }
    }

    return 0;
}

int Convolution_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        int num_input = weight_data_size / num_output;
        if (bottom_blob.w * bottom_blob.elempack == num_input)
        {
            int out_elempack = num_output % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (opt.use_fp16_packed && !opt.use_fp16_storage)
            {
                if (out_elempack == 4) out_elemsize = 4*2u;
                if (out_elempack == 1) out_elemsize = 4u;
            }

            top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;

            std::vector<VkMat> bindings(4);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;
            if (elempack == 1 && out_elempack == 1)
            {
                bindings[2] = weight_data_gpu;
                bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer
            }
            else if (elempack == 4 && out_elempack == 4)
            {
                bindings[2] = weight_data_gpu_pack4;
                bindings[3] = bias_term ? bias_data_gpu_pack4 : bindings[2];// TODO use dummy buffer
            }
            else if (elempack == 1 && out_elempack == 4)
            {
                bindings[2] = weight_data_gpu_pack1to4;
                bindings[3] = bias_term ? bias_data_gpu_pack4 : bindings[2];// TODO use dummy buffer
            }
            else if (elempack == 4 && out_elempack == 1)
            {
                bindings[2] = weight_data_gpu_pack4to1;
                bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer
            }

            std::vector<vk_constant_type> constants(10);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.c;
            constants[4].i = bottom_blob.cstep;
            constants[5].i = top_blob.dims;
            constants[6].i = top_blob.w;
            constants[7].i = top_blob.h;
            constants[8].i = top_blob.c;
            constants[9].i = top_blob.cstep;

            const Pipeline* pipeline = 0;
            if (elempack == 1 && out_elempack == 1)
            {
                pipeline = pipeline_innerproduct;
            }
            else if (elempack == 4 && out_elempack == 4)
            {
                pipeline = pipeline_innerproduct_pack4;
            }
            else if (elempack == 1 && out_elempack == 4)
            {
                pipeline = pipeline_innerproduct_pack1to4;
            }
            else if (elempack == 4 && out_elempack == 1)
            {
                pipeline = pipeline_innerproduct_pack4to1;
            }

            cmd.record_pipeline(pipeline, bindings, constants, top_blob);

            return 0;
        }
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        ncnn::Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            ncnn::Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(4, (size_t)4u, 1, opt.staging_vkallocator, opt.staging_vkallocator);
            padding_param_blob.prepare_staging_buffer();
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            ncnn::Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(4, (size_t)4u, 1, opt.staging_vkallocator, opt.staging_vkallocator);
            padding_param_blob.prepare_staging_buffer();
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 4) out_elemsize = 4*2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    if (elempack == 4 && out_elempack == 4 && is_conv3x3s1d1 && channels * elempack >= 16 && num_output >= 16)
    {
        // winograd23
        int outw_bordered = (outw + 1) / 2 * 2;
        int outh_bordered = (outh + 1) / 2 * 2;

        int w_bordered = outw_bordered + 2;
        int h_bordered = outh_bordered + 2;

        int block_x = outw_bordered / 2;
        int block_y = outh_bordered / 2;

        // pad to 2n+2
        {
            ncnn::Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(4, (size_t)4u, 1, opt.staging_vkallocator, opt.staging_vkallocator);
            padding_param_blob.prepare_staging_buffer();
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = 0;
            padding_params[1] = h_bordered - bottom_blob_bordered.h;
            padding_params[2] = 0;
            padding_params[3] = w_bordered - bottom_blob_bordered.w;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob_bordered;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            winograd_padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }

        // transform input
        VkMat bottom_tm_blob;
        {
            bottom_tm_blob.create(16, block_x * block_y, channels, elemsize, elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
            if (bottom_tm_blob.empty())
                return -100;

            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob_bordered;
            bindings[1] = bottom_tm_blob;

            std::vector<vk_constant_type> constants(7);
            constants[0].i = bottom_blob_bordered.w;
            constants[1].i = bottom_blob_bordered.h;
            constants[2].i = bottom_blob_bordered.c;
            constants[3].i = bottom_blob_bordered.cstep;
            constants[4].i = bottom_tm_blob.cstep;
            constants[5].i = block_x;
            constants[6].i = block_y;

            VkMat dispatcher;
            dispatcher.w = block_x;
            dispatcher.h = block_y;
            dispatcher.c = bottom_tm_blob.c;

            cmd.record_pipeline(pipeline_convolution_pack4_3x3s1d1_winograd23_transform_input, bindings, constants, dispatcher);
        }

        // gemm
        VkMat top_tm_blob;
        {
            top_tm_blob.create(16, block_x * block_y, num_output / out_elempack, elemsize, out_elempack, opt.workspace_vkallocator, opt.staging_vkallocator);
            if (top_tm_blob.empty())
                return -100;

            std::vector<VkMat> bindings(3);
            bindings[0] = bottom_tm_blob;
            bindings[1] = top_tm_blob;
            bindings[2] = weight_data_gpu_pack4_tm;

            std::vector<vk_constant_type> constants(6);
            constants[0].i = bottom_tm_blob.c;
            constants[1].i = bottom_tm_blob.cstep;
            constants[2].i = (top_tm_blob.h + 3) / 4;
            constants[3].i = top_tm_blob.h;
            constants[4].i = top_tm_blob.c;
            constants[5].i = top_tm_blob.cstep;

            VkMat dispatcher;
            dispatcher.w = top_tm_blob.w;
            dispatcher.h = (top_tm_blob.h + 3) / 4;
            dispatcher.c = top_tm_blob.c;

            cmd.record_pipeline(pipeline_convolution_pack4_3x3s1d1_winograd23_gemm, bindings, constants, dispatcher);
        }

        // transform output
        VkMat top_blob_bordered;
        {
            top_blob_bordered.create(outw_bordered, outh_bordered, num_output / out_elempack, elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob_bordered.empty())
                return -100;

            std::vector<VkMat> bindings(3);
            bindings[0] = top_tm_blob;
            bindings[1] = top_blob_bordered;
            bindings[2] = bias_term ? bias_data_gpu_pack4 : bindings[1];

            std::vector<vk_constant_type> constants(7);
            constants[0].i = top_tm_blob.c;
            constants[1].i = top_tm_blob.cstep;
            constants[2].i = block_x;
            constants[3].i = block_y;
            constants[4].i = top_blob_bordered.w;
            constants[5].i = top_blob_bordered.h;
            constants[6].i = top_blob_bordered.cstep;

            VkMat dispatcher;
            dispatcher.w = block_x;
            dispatcher.h = block_y;
            dispatcher.c = top_blob_bordered.c;

            cmd.record_pipeline(pipeline_convolution_pack4_3x3s1d1_winograd23_transform_output, bindings, constants, dispatcher);
        }

        // crop top_blob
        {
            VkMat crop_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator, opt.staging_vkallocator);
            crop_param_blob.prepare_staging_buffer();
            int* crop_params = crop_param_blob.mapped();

            crop_params[0] = 0;
            crop_params[1] = 0;
            crop_params[2] = 0;
            crop_params[3] = outw;
            crop_params[4] = outh;
            crop_params[5] = num_output;

            std::vector<VkMat> crop_inputs(2);
            crop_inputs[0] = top_blob_bordered;
            crop_inputs[1] = crop_param_blob;

            std::vector<VkMat> crop_outputs(1);
            winograd_crop->forward(crop_inputs, crop_outputs, cmd, opt);
            top_blob = crop_outputs[0];
        }

        return 0;
    }

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;
    if (elempack == 1 && out_elempack == 1)
    {
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        bindings[2] = weight_data_gpu_pack4;
        bindings[3] = bias_term ? bias_data_gpu_pack4 : bindings[2];// TODO use dummy buffer
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        bindings[2] = weight_data_gpu_pack1to4;
        bindings[3] = bias_term ? bias_data_gpu_pack4 : bindings[2];// TODO use dummy buffer
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        bindings[2] = weight_data_gpu_pack4to1;
        bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer
    }

    // record
    if (elempack == 1 && out_elempack == 1 && kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
    {
        std::vector<vk_constant_type> constants(8);
        constants[0].i = bottom_blob_bordered.dims;
        constants[1].i = bottom_blob_bordered.cstep / 4;
        constants[2].i = bottom_blob_bordered.c;
        constants[3].i = bottom_blob_bordered.cstep / 4;
        constants[4].i = top_blob.dims;
        constants[5].i = top_blob.cstep / 4;
        constants[6].i = top_blob.c;
        constants[7].i = top_blob.cstep / 4;

        VkMat dispatcher;
        dispatcher.w = top_blob.cstep / 4;
        dispatcher.h = 1;
        dispatcher.c = top_blob.c;

        cmd.record_pipeline(pipeline_convolution_1x1s1d1, bindings, constants, dispatcher);
    }
    else if (elempack == 4 && out_elempack == 4 && kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
    {
        std::vector<vk_constant_type> constants(8);
        constants[0].i = bottom_blob_bordered.dims;
        constants[1].i = (bottom_blob_bordered.cstep + 3) / 4;
        constants[2].i = bottom_blob_bordered.c;
        constants[3].i = bottom_blob_bordered.cstep;
        constants[4].i = top_blob.dims;
        constants[5].i = (top_blob.cstep + 3) / 4;
        constants[6].i = top_blob.c;
        constants[7].i = top_blob.cstep;

        VkMat dispatcher;
        dispatcher.w = (top_blob.cstep + 3) / 4;
        dispatcher.h = 1;
        dispatcher.c = top_blob.c;

        cmd.record_pipeline(pipeline_convolution_pack4_1x1s1d1, bindings, constants, dispatcher);
    }
    else if (elempack == 4 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
    {
        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_bordered.dims;
        constants[1].i = bottom_blob_bordered.w;
        constants[2].i = bottom_blob_bordered.h;
        constants[3].i = bottom_blob_bordered.c;
        constants[4].i = bottom_blob_bordered.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        cmd.record_pipeline(pipeline_convolution_pack4_3x3s1d1_lds_8_8_2, bindings, constants, top_blob);
    }
    else
    {
        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_bordered.dims;
        constants[1].i = bottom_blob_bordered.w;
        constants[2].i = bottom_blob_bordered.h;
        constants[3].i = bottom_blob_bordered.c;
        constants[4].i = bottom_blob_bordered.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        const Pipeline* pipeline = 0;
        if (elempack == 1 && out_elempack == 1)
        {
            pipeline = pipeline_convolution;
        }
        else if (elempack == 4 && out_elempack == 4)
        {
            pipeline = pipeline_convolution_pack4;
        }
        else if (elempack == 1 && out_elempack == 4)
        {
            pipeline = pipeline_convolution_pack1to4;
        }
        else if (elempack == 4 && out_elempack == 1)
        {
            pipeline = pipeline_convolution_pack4to1;
        }

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
}

} // namespace ncnn
