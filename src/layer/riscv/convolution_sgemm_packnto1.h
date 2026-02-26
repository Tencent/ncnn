// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void im2col_sgemm_packnto1_rvv(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    // Mat bottom_im2col(size, maxk, inch, 4u * packn, packn, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 4u * packn, packn, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 4u * packn, packn, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 4u * packn, packn, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 4u * packn, packn, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        int nn_size = size >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            float* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
#if C906
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = img0[l];
                        tmpptr[1] = img0[l + packn];
                        tmpptr[2] = img0[l + packn * 2];
                        tmpptr[3] = img0[l + packn * 3];
                        tmpptr[4] = img0[l + packn * 4];
                        tmpptr[5] = img0[l + packn * 5];
                        tmpptr[6] = img0[l + packn * 6];
                        tmpptr[7] = img0[l + packn * 7];
                        tmpptr += 8;
                    }

                    img0 += size * packn;
#else
                    vfloat32m1_t _val0 = __riscv_vle32_v_f32m1(img0, vl);
                    vfloat32m1_t _val1 = __riscv_vle32_v_f32m1(img0 + packn, vl);
                    vfloat32m1_t _val2 = __riscv_vle32_v_f32m1(img0 + packn * 2, vl);
                    vfloat32m1_t _val3 = __riscv_vle32_v_f32m1(img0 + packn * 3, vl);
                    vfloat32m1_t _val4 = __riscv_vle32_v_f32m1(img0 + packn * 4, vl);
                    vfloat32m1_t _val5 = __riscv_vle32_v_f32m1(img0 + packn * 5, vl);
                    vfloat32m1_t _val6 = __riscv_vle32_v_f32m1(img0 + packn * 6, vl);
                    vfloat32m1_t _val7 = __riscv_vle32_v_f32m1(img0 + packn * 7, vl);
                    __riscv_vsseg8e32_v_f32m1x8(tmpptr, __riscv_vcreate_v_f32m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                    img0 += size * packn;
                    tmpptr += packn * 8;
#endif
                }
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
#if C906
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = img0[l];
                        tmpptr[1] = img0[l + packn];
                        tmpptr[2] = img0[l + packn * 2];
                        tmpptr[3] = img0[l + packn * 3];
                        tmpptr += 4;
                    }

                    img0 += size * packn;
#else
                    vfloat32m1_t _val0 = __riscv_vle32_v_f32m1(img0, vl);
                    vfloat32m1_t _val1 = __riscv_vle32_v_f32m1(img0 + packn, vl);
                    vfloat32m1_t _val2 = __riscv_vle32_v_f32m1(img0 + packn * 2, vl);
                    vfloat32m1_t _val3 = __riscv_vle32_v_f32m1(img0 + packn * 3, vl);
                    __riscv_vsseg4e32_v_f32m1x4(tmpptr, __riscv_vcreate_v_f32m1x4(_val0, _val1, _val2, _val3), vl);

                    img0 += size * packn;
                    tmpptr += packn * 4;
#endif
                }
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
#if C906
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = img0[l];
                        tmpptr[1] = img0[l + packn];
                        tmpptr += 2;
                    }

                    img0 += size * packn;
#else
                    vfloat32m1_t _val0 = __riscv_vle32_v_f32m1(img0, vl);
                    vfloat32m1_t _val1 = __riscv_vle32_v_f32m1(img0 + packn, vl);
                    __riscv_vsseg2e32_v_f32m1x2(tmpptr, __riscv_vcreate_v_f32m1x2(_val0, _val1), vl);

                    img0 += size * packn;
                    tmpptr += packn * 2;
#endif
                }
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    vfloat32m1_t _val = __riscv_vle32_v_f32m1(img0, vl);
                    __riscv_vse32_v_f32m1(tmpptr, _val, vl);

                    img0 += size * packn;
                    tmpptr += packn;
                }
            }
        }
    }

    int nn_outch = outch / packn;
    int remain_outch_start = nn_outch * packn;

#ifdef __clang__
    // clang complains about VLA in the following loop
    float* _zero_tmp = new float[packn]();
    for (int _zero_clean_idx = 0; _zero_clean_idx < packn; _zero_clean_idx++)
    {
        _zero_tmp[_zero_clean_idx] = 0.f;
    }
#endif // __clang__
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * packn;

        float* outptr0 = top_blob.channel(p);

#ifdef __clang__
        const float* zeros = _zero_tmp;
#else
        const float zeros[packn] = {0.f};
#endif // __clang__
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
            const float* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat32m1_t _sum0 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum1 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum2 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum3 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum4 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum5 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum6 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum7 = __riscv_vle32_v_f32m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                float val0 = *tmpptr++;
                float val1 = *tmpptr++;
                float val2 = *tmpptr++;
                float val3 = *tmpptr++;
                float val4 = *tmpptr++;
                float val5 = *tmpptr++;
                float val6 = *tmpptr++;
                float val7 = *tmpptr++;
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3, _w0, vl);
                _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4, _w0, vl);
                _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5, _w0, vl);
                _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6, _w0, vl);
                _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7, _w0, vl);

                kptr0 += packn;
            }

#if C906
            __riscv_vsse32_v_f32m1(outptr0, top_blob.cstep * sizeof(float), _sum0, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 1, top_blob.cstep * sizeof(float), _sum1, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 2, top_blob.cstep * sizeof(float), _sum2, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 3, top_blob.cstep * sizeof(float), _sum3, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 4, top_blob.cstep * sizeof(float), _sum4, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 5, top_blob.cstep * sizeof(float), _sum5, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 6, top_blob.cstep * sizeof(float), _sum6, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 7, top_blob.cstep * sizeof(float), _sum7, vl);
#else
            __riscv_vssseg8e32_v_f32m1x8(outptr0, top_blob.cstep * sizeof(float), __riscv_vcreate_v_f32m1x8(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7), vl);
#endif
            outptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const float* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat32m1_t _sum0 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum1 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum2 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum3 = __riscv_vle32_v_f32m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                float val0 = *tmpptr++;
                float val1 = *tmpptr++;
                float val2 = *tmpptr++;
                float val3 = *tmpptr++;
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3, _w0, vl);

                kptr0 += packn;
            }

#if C906
            __riscv_vsse32_v_f32m1(outptr0, top_blob.cstep * sizeof(float), _sum0, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 1, top_blob.cstep * sizeof(float), _sum1, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 2, top_blob.cstep * sizeof(float), _sum2, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 3, top_blob.cstep * sizeof(float), _sum3, vl);
#else
            __riscv_vssseg4e32_v_f32m1x4(outptr0, top_blob.cstep * sizeof(float), __riscv_vcreate_v_f32m1x4(_sum0, _sum1, _sum2, _sum3), vl);
#endif
            outptr0 += 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const float* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat32m1_t _sum0 = __riscv_vle32_v_f32m1(biasptr, vl);
            vfloat32m1_t _sum1 = __riscv_vle32_v_f32m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                float val0 = *tmpptr++;
                float val1 = *tmpptr++;
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1, _w0, vl);

                kptr0 += packn;
            }

#if C906
            __riscv_vsse32_v_f32m1(outptr0, top_blob.cstep * sizeof(float), _sum0, vl);
            __riscv_vsse32_v_f32m1(outptr0 + 1, top_blob.cstep * sizeof(float), _sum1, vl);
#else
            __riscv_vssseg2e32_v_f32m1x2(outptr0, top_blob.cstep * sizeof(float), __riscv_vcreate_v_f32m1x2(_sum0, _sum1), vl);
#endif
            outptr0 += 2;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const float* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat32m1_t _sum = __riscv_vle32_v_f32m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                float val = *tmpptr++;
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum = __riscv_vfmacc_vf_f32m1(_sum, val, _w0, vl);

                kptr0 += packn;
            }

            __riscv_vsse32_v_f32m1(outptr0, top_blob.cstep * sizeof(float), _sum, vl);

            outptr0 += 1;
        }
    }
#ifdef __clang__
    delete[] _zero_tmp;
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
            const float* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;
            float sum1 = bias0;
            float sum2 = bias0;
            float sum3 = bias0;
            float sum4 = bias0;
            float sum5 = bias0;
            float sum6 = bias0;
            float sum7 = bias0;

            vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum4 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum5 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum6 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum7 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            for (int j = 0; j < nn; j++)
            {
                vfloat32m1x8_t _val = __riscv_vlseg8e32_v_f32m1x8(tmpptr, vl);
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum0 = __riscv_vfmacc_vv_f32m1(_sum0, __riscv_vget_v_f32m1x8_f32m1(_val, 0), _w0, vl);
                _sum1 = __riscv_vfmacc_vv_f32m1(_sum1, __riscv_vget_v_f32m1x8_f32m1(_val, 1), _w0, vl);
                _sum2 = __riscv_vfmacc_vv_f32m1(_sum2, __riscv_vget_v_f32m1x8_f32m1(_val, 2), _w0, vl);
                _sum3 = __riscv_vfmacc_vv_f32m1(_sum3, __riscv_vget_v_f32m1x8_f32m1(_val, 3), _w0, vl);
                _sum4 = __riscv_vfmacc_vv_f32m1(_sum4, __riscv_vget_v_f32m1x8_f32m1(_val, 4), _w0, vl);
                _sum5 = __riscv_vfmacc_vv_f32m1(_sum5, __riscv_vget_v_f32m1x8_f32m1(_val, 5), _w0, vl);
                _sum6 = __riscv_vfmacc_vv_f32m1(_sum6, __riscv_vget_v_f32m1x8_f32m1(_val, 6), _w0, vl);
                _sum7 = __riscv_vfmacc_vv_f32m1(_sum7, __riscv_vget_v_f32m1x8_f32m1(_val, 7), _w0, vl);
                tmpptr += packn * 8;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<float> ss0(packn);
            std::vector<float> ss1(packn);
            std::vector<float> ss2(packn);
            std::vector<float> ss3(packn);
            std::vector<float> ss4(packn);
            std::vector<float> ss5(packn);
            std::vector<float> ss6(packn);
            std::vector<float> ss7(packn);
            __riscv_vse32_v_f32m1((float*)ss0.data(), _sum0, vl);
            __riscv_vse32_v_f32m1((float*)ss1.data(), _sum1, vl);
            __riscv_vse32_v_f32m1((float*)ss2.data(), _sum2, vl);
            __riscv_vse32_v_f32m1((float*)ss3.data(), _sum3, vl);
            __riscv_vse32_v_f32m1((float*)ss4.data(), _sum4, vl);
            __riscv_vse32_v_f32m1((float*)ss5.data(), _sum5, vl);
            __riscv_vse32_v_f32m1((float*)ss6.data(), _sum6, vl);
            __riscv_vse32_v_f32m1((float*)ss7.data(), _sum7, vl);
            for (int i = 0; i < packn; i++)
            {
                sum0 += ss0[i];
                sum1 += ss1[i];
                sum2 += ss2[i];
                sum3 += ss3[i];
                sum4 += ss4[i];
                sum5 += ss5[i];
                sum6 += ss6[i];
                sum7 += ss7[i];
            }
#else
            sum0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum0, __riscv_vfmv_s_f_f32m1(sum0, vl), vl));
            sum1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum1, __riscv_vfmv_s_f_f32m1(sum1, vl), vl));
            sum2 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum2, __riscv_vfmv_s_f_f32m1(sum2, vl), vl));
            sum3 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum3, __riscv_vfmv_s_f_f32m1(sum3, vl), vl));
            sum4 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum4, __riscv_vfmv_s_f_f32m1(sum4, vl), vl));
            sum5 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum5, __riscv_vfmv_s_f_f32m1(sum5, vl), vl));
            sum6 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum6, __riscv_vfmv_s_f_f32m1(sum6, vl), vl));
            sum7 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum7, __riscv_vfmv_s_f_f32m1(sum7, vl), vl));
#endif

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;
            outptr0[4] = sum4;
            outptr0[5] = sum5;
            outptr0[6] = sum6;
            outptr0[7] = sum7;

            outptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const float* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;
            float sum1 = bias0;
            float sum2 = bias0;
            float sum3 = bias0;

            vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            for (int j = 0; j < nn; j++)
            {
                vfloat32m1x4_t _val = __riscv_vlseg4e32_v_f32m1x4(tmpptr, vl);
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum0 = __riscv_vfmacc_vv_f32m1(_sum0, __riscv_vget_v_f32m1x4_f32m1(_val, 0), _w0, vl);
                _sum1 = __riscv_vfmacc_vv_f32m1(_sum1, __riscv_vget_v_f32m1x4_f32m1(_val, 1), _w0, vl);
                _sum2 = __riscv_vfmacc_vv_f32m1(_sum2, __riscv_vget_v_f32m1x4_f32m1(_val, 2), _w0, vl);
                _sum3 = __riscv_vfmacc_vv_f32m1(_sum3, __riscv_vget_v_f32m1x4_f32m1(_val, 3), _w0, vl);
                tmpptr += packn * 4;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<float> ss0(packn);
            std::vector<float> ss1(packn);
            std::vector<float> ss2(packn);
            std::vector<float> ss3(packn);
            __riscv_vse32_v_f32m1((float*)ss0.data(), _sum0, vl);
            __riscv_vse32_v_f32m1((float*)ss1.data(), _sum1, vl);
            __riscv_vse32_v_f32m1((float*)ss2.data(), _sum2, vl);
            __riscv_vse32_v_f32m1((float*)ss3.data(), _sum3, vl);
            for (int i = 0; i < packn; i++)
            {
                sum0 += ss0[i];
                sum1 += ss1[i];
                sum2 += ss2[i];
                sum3 += ss3[i];
            }
#else
            sum0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum0, __riscv_vfmv_s_f_f32m1(sum0, vl), vl));
            sum1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum1, __riscv_vfmv_s_f_f32m1(sum1, vl), vl));
            sum2 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum2, __riscv_vfmv_s_f_f32m1(sum2, vl), vl));
            sum3 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum3, __riscv_vfmv_s_f_f32m1(sum3, vl), vl));
#endif

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;

            outptr0 += 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const float* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;
            float sum1 = bias0;

            vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            for (int j = 0; j < nn; j++)
            {
                vfloat32m1x2_t _val = __riscv_vlseg2e32_v_f32m1x2(tmpptr, vl);
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum0 = __riscv_vfmacc_vv_f32m1(_sum0, __riscv_vget_v_f32m1x2_f32m1(_val, 0), _w0, vl);
                _sum1 = __riscv_vfmacc_vv_f32m1(_sum1, __riscv_vget_v_f32m1x2_f32m1(_val, 1), _w0, vl);
                tmpptr += packn * 2;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<float> ss0(packn);
            std::vector<float> ss1(packn);
            __riscv_vse32_v_f32m1((float*)ss0.data(), _sum0, vl);
            __riscv_vse32_v_f32m1((float*)ss1.data(), _sum1, vl);
            for (int i = 0; i < packn; i++)
            {
                sum0 += ss0[i];
                sum1 += ss1[i];
            }
#else
            sum0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum0, __riscv_vfmv_s_f_f32m1(sum0, vl), vl));
            sum1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum1, __riscv_vfmv_s_f_f32m1(sum1, vl), vl));
#endif

            outptr0[0] = sum0;
            outptr0[1] = sum1;

            outptr0 += 2;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const float* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;

            vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            for (int j = 0; j < nn; j++)
            {
                vfloat32m1_t _val0 = __riscv_vle32_v_f32m1(tmpptr, vl);
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum0 = __riscv_vfmacc_vv_f32m1(_sum0, _val0, _w0, vl);
                tmpptr += packn;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<float> ss0(packn);
            __riscv_vse32_v_f32m1((float*)ss0.data(), _sum0, vl);
            for (int i = 0; i < packn; i++)
            {
                sum0 += ss0[i];
            }
#else
            sum0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(_sum0, __riscv_vfmv_s_f_f32m1(sum0, vl), vl));
#endif

            outptr0[0] = sum0;

            outptr0 += 1;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_packnto1_rvv(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int packn = csrr_vlenb() / 4;

    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(packn * packn * maxk, inch / packn, outch / packn + outch % packn);

    int q = 0;
    for (; q + (packn - 1) < outch; q += packn)
    {
        float* g00 = kernel_tm.channel(q / packn);

        for (int p = 0; p + (packn - 1) < inch; p += packn)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < packn; i++)
                {
                    for (int j = 0; j < packn; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; q < outch; q++)
    {
        const Mat k0 = kernel.channel(q);

        float* g00 = kernel_tm.channel(q / packn + q % packn);

        for (int p = 0; p + (packn - 1) < inch; p += packn)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < packn; j++)
                {
                    const float* k00 = k0.row(p + j);

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_packnto1_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 4u * packn, packn, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * packn;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const float* sptr = img.row(dilation_h * u) + dilation_w * v * packn;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr, vl);
                            __riscv_vse32_v_f32m1(ptr, _val, vl);

                            sptr += stride_w * packn;
                            ptr += packn;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_packnto1_rvv(bottom_im2col, top_blob, kernel, _bias, opt);
}
