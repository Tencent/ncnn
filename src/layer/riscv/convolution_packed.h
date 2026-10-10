// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_transform_kernel_packed_simple_rvv(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                            g00[0] = k00[k];
                            g00++;
                        }
                    }
                }
            }
        }
    }
}

static void convolution_transform_kernel_packed_rvv(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif // __riscv_vector

    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    // clang-format off
    // *INDENT-OFF*
#if __riscv_vector
    if (num_output >= packn)
    {
        if (num_input >= packn)
            weight_data_tm.create(packn * packn * maxk, num_input / packn + num_input % packn, num_output / packn + num_output % packn);
        else
            weight_data_tm.create(packn * maxk, num_input, num_output / packn + num_output % packn);
    }
    else
    {
        if (num_input >= packn)
            weight_data_tm.create(packn * maxk, num_input / packn + num_input % packn, num_output);
        else
            weight_data_tm.create(maxk, num_input, num_output);
    }
#else
    weight_data_tm.create(maxk, num_input, num_output);
#endif // __riscv_vector
    // *INDENT-ON*
    // clang-format on

    Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

    int q = 0;
#if __riscv_vector
    for (; q + (packn - 1) < num_output; q += packn)
    {
        float* g00 = weight_data_tm.channel(q / packn);

        int p = 0;
        for (; p + (packn - 1) < num_input; p += packn)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < packn; i++)
                {
                    for (int j = 0; j < packn; j++)
                    {
                        const float* k00 = weight_data_r2.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p < num_input; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < packn; j++)
                {
                    const float* k00 = weight_data_r2.channel(q + j).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __riscv_vector
    for (; q < num_output; q++)
    {
#if __riscv_vector
        float* g00 = weight_data_tm.channel(q / packn + q % packn);
#else
        float* g00 = weight_data_tm.channel(q);
#endif // __riscv_vector

        int p = 0;
#if __riscv_vector
        for (; p + (packn - 1) < num_input; p += packn)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < packn; i++)
                {
                    const float* k00 = weight_data_r2.channel(q).row(p + i);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
#endif // __riscv_vector
        for (; p < num_input; p++)
        {
            const float* k00 = weight_data_r2.channel(q).row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void convolution_packed_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif // __riscv_vector

    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

#if __riscv_vector
    const size_t N = bottom_blob.cstep * elempack;
#endif // __riscv_vector

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;

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
                space_ofs[p1] = p2 * elempack;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    const float* bias_data_ptr = bias_data;

    int remain_outch_start = 0;

#if __riscv_vector
    int nn_outch = outch / packn;
    remain_outch_start = nn_outch * packn;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = pp * packn;

        float* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);

                if (bias_data_ptr)
                {
                    _sum = __riscv_vle32_v_f32m1(bias_data_ptr + p, vl);
                }

                const float* kptr = weight_data_tm.channel(p / packn);

                int q = 0;
                if (elempack == packn)
                {
                    for (; q + (packn - 1) < inch; q += packn)
                    {
                        const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            for (int l = 0; l < packn; l++)
                            {
                                float val = r0[sok + l];
                                vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                                _sum = __riscv_vfmacc_vf_f32m1(_sum, val, _w, vl);

                                kptr += packn;
                            }
                        }
                    }
                }
                else // if (elempack == 1)
                {
                    for (; q + (packn - 1) < inch; q += packn)
                    {
                        const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            for (int l = 0; l < packn; l++)
                            {
                                float val = r0[sok + N * l];
                                vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                                _sum = __riscv_vfmacc_vf_f32m1(_sum, val, _w, vl);

                                kptr += packn;
                            }
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = r0[space_ofs[k]];
                        vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                        _sum = __riscv_vfmacc_vf_f32m1(_sum, val, _w, vl);

                        kptr += packn;
                    }
                }

                _sum = activation_ps(_sum, activation_type, activation_params, vl);

                if (out_elempack == packn)
                {
                    __riscv_vse32_v_f32m1(outptr, _sum, vl);
                    outptr += packn;
                }
                else // if (out_elempack == 1)
                {
                    __riscv_vsse32_v_f32m1(outptr, top_blob.cstep * sizeof(float), _sum, vl);
                    outptr += 1;
                }
            }
        }
    }
#endif // __riscv_vector

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
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

#if __riscv_vector
                const float* kptr = weight_data_tm.channel(p / packn + p % packn);
#else
                const float* kptr = weight_data_tm.channel(p);
#endif // __riscv_vector

                int q = 0;
#if __riscv_vector
                vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
                if (elempack == packn)
                {
                    for (; q + (packn - 1) < inch; q += packn)
                    {
                        const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(r0 + sok, vl);
                            vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                            _sum = __riscv_vfmacc_vv_f32m1(_sum, _val, _w, vl);

                            kptr += packn;
                        }
                    }
                }
                else // if (elempack == 1)
                {
                    for (; q + (packn - 1) < inch; q += packn)
                    {
                        const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            vfloat32m1_t _val = __riscv_vlse32_v_f32m1(r0 + sok, N * sizeof(float), vl);
                            vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                            _sum = __riscv_vfmacc_vv_f32m1(_sum, _val, _w, vl);

                            kptr += packn;
                        }
                    }
                }
                sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum, __riscv_vfmv_s_f_f32m1(sum, vl), vl));
#endif // __riscv_vector

                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = r0[space_ofs[k]];
                        float wt = kptr[0];
                        sum += val * wt;

                        kptr += 1;
                    }
                }

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }
}
