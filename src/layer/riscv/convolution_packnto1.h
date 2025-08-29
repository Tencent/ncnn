// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_packnto1_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_packnto1, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    int w = bottom_blob.w;
    int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    const float* bias_data_ptr = bias_data;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

                vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);

                const float* kptr = (const float*)weight_data_packnto1.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w * packn;

                    for (int k = 0; k < maxk; k++)
                    {
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + space_ofs[k] * packn, vl);
                        vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                        _sum = __riscv_vfmacc_vv_f32m1(_sum, _val, _w, vl);

                        kptr += packn;
                    }
                }

                sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum, __riscv_vfmv_s_f_f32m1(sum, vl), vl));

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }
}
