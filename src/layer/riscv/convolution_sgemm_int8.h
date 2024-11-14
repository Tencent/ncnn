// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void im2col_sgemm_int8_rvv(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
#if __riscv_vector
    int packn = csrr_vlenb();
    size_t vl = vsetvl_e8m1(packn);
#else
    int packn = 4;
    size_t vl = 4;
#endif

    // Mat bottom_im2col(size, maxk, inch, 1u, 1, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    // permute
    Mat tmp;
    if (size >= packn)
        tmp.create(packn * maxk, inch, size / packn + size % packn, 1u, 1, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 1u, 1, opt.workspace_allocator);
    {
        int nn_size = size / packn;
        int remain_size_start = nn_size * packn;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * packn;

            int8_t* tmpptr = tmp.channel(i / packn);

            for (int q = 0; q < inch; q++)
            {
                const int8_t* img0 = (const int8_t*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
#if __riscv_vector
                    vse8_v_i8m1(tmpptr, vle8_v_i8m1(img0, vl), vl);
#else
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];
                    tmpptr[2] = img0[2];
                    tmpptr[3] = img0[3];
#endif
                    img0 += size;
                    tmpptr += packn;
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            int8_t* tmpptr = tmp.channel(i / packn + i % packn);

            for (int q = 0; q < inch; q++)
            {
                const int8_t* img0 = (const int8_t*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    img0 += size;
                    tmpptr += 1;
                }
            }
        }
    }


#if __riscv_vector
    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);
        int* outptr2 = top_blob.channel(p + 2);
        int* outptr3 = top_blob.channel(p + 3);
        int* outptr4 = top_blob.channel(p + 4);
        int* outptr5 = top_blob.channel(p + 5);
        int* outptr6 = top_blob.channel(p + 6);
        int* outptr7 = top_blob.channel(p + 7);

        int i = 0;
        for (; i + (packn - 1) < size; i += packn)
        {
            const int8_t* tmpptr = tmp.channel(i / packn);
            const int8_t* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            vint32m4_t _sum0_32 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum1_32 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum2_32 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum3_32 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum4_32 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum5_32 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum6_32 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum7_32 = vmv_v_x_i32m4(0, vl);

            for (int q = 0; q < nn; q++)
            {
                vint16m2_t _val = vwcvt_x_x_v_i16m2(vle8_v_i8m1(tmpptr, vl), vl);
                _sum0_32 = vwmacc_vx_i32m4(_sum0_32, kptr[0], _val, vl);
                _sum1_32 = vwmacc_vx_i32m4(_sum1_32, kptr[1], _val, vl);
                _sum2_32 = vwmacc_vx_i32m4(_sum2_32, kptr[2], _val, vl);
                _sum3_32 = vwmacc_vx_i32m4(_sum3_32, kptr[3], _val, vl);
                _sum4_32 = vwmacc_vx_i32m4(_sum4_32, kptr[4], _val, vl);
                _sum5_32 = vwmacc_vx_i32m4(_sum5_32, kptr[5], _val, vl);
                _sum6_32 = vwmacc_vx_i32m4(_sum6_32, kptr[6], _val, vl);
                _sum7_32 = vwmacc_vx_i32m4(_sum7_32, kptr[7], _val, vl);
                tmpptr += packn;
                kptr += 8;
            }

            vse32_v_i32m4(outptr0, _sum0_32, vl);
            vse32_v_i32m4(outptr1, _sum1_32, vl);
            vse32_v_i32m4(outptr2, _sum2_32, vl);
            vse32_v_i32m4(outptr3, _sum3_32, vl);
            vse32_v_i32m4(outptr4, _sum4_32, vl);
            vse32_v_i32m4(outptr5, _sum5_32, vl);
            vse32_v_i32m4(outptr6, _sum6_32, vl);
            vse32_v_i32m4(outptr7, _sum7_32, vl);

            outptr0 += packn;
            outptr1 += packn;
            outptr2 += packn;
            outptr3 += packn;
            outptr4 += packn;
            outptr5 += packn;
            outptr6 += packn;
            outptr7 += packn;
        }
        for (; i < size; i++)
        {
            const int8_t* tmpptr = tmp.channel(i / packn + i % packn);
            const int8_t* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;
            int sum4 = 0;
            int sum5 = 0;
            int sum6 = 0;
            int sum7 = 0;

            for (int q = 0; q < nn; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                sum2 += tmpptr[0] * kptr[2];
                sum3 += tmpptr[0] * kptr[3];
                sum4 += tmpptr[0] * kptr[4];
                sum5 += tmpptr[0] * kptr[5];
                sum6 += tmpptr[0] * kptr[6];
                sum7 += tmpptr[0] * kptr[7];
                tmpptr++;
                kptr += 8;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;
            outptr4[0] = sum4;
            outptr5[0] = sum5;
            outptr6[0] = sum6;
            outptr7[0] = sum7;

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
            outptr4++;
            outptr5++;
            outptr6++;
            outptr7++;
        }
    }

    nn_outch = (outch - remain_outch_start) >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);
        int* outptr2 = top_blob.channel(p + 2);
        int* outptr3 = top_blob.channel(p + 3);

        int i = 0;
        for (; i + (packn - 1) < size; i += packn)
        {
            const int8_t* tmpptr = tmp.channel(i / packn);
            const int8_t* kptr = kernel.channel(p / 8 + (p % 8) / 4);

            int nn = inch * maxk; // inch always > 0

            vint32m4_t _sum0_32 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum1_32 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum2_32 = vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum3_32 = vmv_v_x_i32m4(0, vl);

            for (int q = 0; q < nn; q++)
            {
                vint16m2_t _val = vwcvt_x_x_v_i16m2(vle8_v_i8m1(tmpptr, vl), vl);
                _sum0_32 = vwmacc_vx_i32m4(_sum0_32, kptr[0], _val, vl);
                _sum1_32 = vwmacc_vx_i32m4(_sum1_32, kptr[1], _val, vl);
                _sum2_32 = vwmacc_vx_i32m4(_sum2_32, kptr[2], _val, vl);
                _sum3_32 = vwmacc_vx_i32m4(_sum3_32, kptr[3], _val, vl);

                tmpptr += packn;
                kptr += 4;
            }

            vse32_v_i32m4(outptr0, _sum0_32, vl);
            vse32_v_i32m4(outptr1, _sum1_32, vl);
            vse32_v_i32m4(outptr2, _sum2_32, vl);
            vse32_v_i32m4(outptr3, _sum3_32, vl);

            outptr0 += packn;
            outptr1 += packn;
            outptr2 += packn;
            outptr3 += packn;
        }
        for (; i < size; i++)
        {
            const int8_t* tmpptr = tmp.channel(i / packn + i % packn);
            const int8_t* kptr = kernel.channel(p / 8 + (p % 8) / 4);

            int nn = inch * maxk; // inch always > 0

            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            for (int q = 0; q < nn; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                sum2 += tmpptr[0] * kptr[2];
                sum3 += tmpptr[0] * kptr[3];
                tmpptr++;
                kptr += 4;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
        }
    }

    remain_outch_start += nn_outch << 2;
#else
    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            const int8_t* tmpptr = tmp.channel(i / 4);
            const int8_t* kptr = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            int sum00 = 0;
            int sum01 = 0;
            int sum02 = 0;
            int sum03 = 0;
            int sum10 = 0;
            int sum11 = 0;
            int sum12 = 0;
            int sum13 = 0;

            for (int q = 0; q < nn; q++)
            {
                int8_t k0 = kptr[0];
                int8_t k1 = kptr[1];
                sum00 += tmpptr[0] * k0;
                sum01 += tmpptr[1] * k0;
                sum02 += tmpptr[2] * k0;
                sum03 += tmpptr[3] * k0;
                sum10 += tmpptr[0] * k1;
                sum11 += tmpptr[1] * k1;
                sum12 += tmpptr[2] * k1;
                sum13 += tmpptr[3] * k1;
                tmpptr += 4;
                kptr += 2;
            }

            outptr0[0] = sum00;
            outptr0[1] = sum01;
            outptr0[2] = sum02;
            outptr0[3] = sum03;
            outptr1[0] = sum10;
            outptr1[1] = sum11;
            outptr1[2] = sum12;
            outptr1[3] = sum13;

            outptr0 += 4;
            outptr1 += 4;
        }
        for (; i < size; i++)
        {
            const int8_t* tmpptr = tmp.channel(i / 4 + i % 4);
            const int8_t* kptr = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            int sum0 = 0;
            int sum1 = 0;

            for (int q = 0; q < nn; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                tmpptr++;
                kptr += 2;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;

            outptr0++;
            outptr1++;
        }
    }
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr0 = top_blob.channel(p);

        int i = 0;
        for (; i + (packn - 1) < size; i += packn)
        {
            const int8_t* tmpptr = tmp.channel(i / packn);
#if __riscv_vector
            const int8_t* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const int8_t* kptr = kernel.channel(p / 2 + p % 2);
#endif

            int nn = inch * maxk; // inch always > 0

#if __riscv_vector
            vint32m4_t _sum0_32 = vmv_v_x_i32m4(0, vl);
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;
#endif

            for (int q = 0; q < nn; q++)
            {
#if __riscv_vector
                vint16m2_t _val = vwcvt_x_x_v_i16m2(vle8_v_i8m1(tmpptr, vl), vl);
                _sum0_32 = vwmacc_vx_i32m4(_sum0_32, kptr[0], _val, vl);
#else
                int8_t k0 = kptr[0];
                sum0 += tmpptr[0] * k0;
                sum1 += tmpptr[1] * k0;
                sum2 += tmpptr[2] * k0;
                sum3 += tmpptr[3] * k0;
#endif
                tmpptr += packn;
                kptr++;
            }

#if __riscv_vector
            vse32_v_i32m4(outptr0, _sum0_32, vl);
#else
            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;
#endif
            outptr0 += packn;
        }
        for (; i < size; i++)
        {
            const int8_t* tmpptr = tmp.channel(i / packn + i % packn);
#if __riscv_vector
            const int8_t* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const int8_t* kptr = kernel.channel(p / 2 + p % 2);
#endif

            int nn = inch * maxk; // inch always > 0

            int sum0 = 0;

            for (int q = 0; q < nn; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_int8_rvv(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-maxk-inch-outch/8b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __riscv_vector
    kernel_tm.create(8 * maxk, inch, outch / 8 + (outch % 8) / 4 + outch % 4, (size_t)1u);
#else
    kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2, (size_t)1u);
#endif
    int q = 0;
#if __riscv_vector
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);
        const Mat k4 = kernel.channel(q + 4);
        const Mat k5 = kernel.channel(q + 5);
        const Mat k6 = kernel.channel(q + 6);
        const Mat k7 = kernel.channel(q + 7);

        int8_t* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p < inch; p++)
        {
            const int8_t* k00 = (const int8_t*)k0.row(p);
            const int8_t* k10 = (const int8_t*)k1.row(p);
            const int8_t* k20 = (const int8_t*)k2.row(p);
            const int8_t* k30 = (const int8_t*)k3.row(p);
            const int8_t* k40 = (const int8_t*)k4.row(p);
            const int8_t* k50 = (const int8_t*)k5.row(p);
            const int8_t* k60 = (const int8_t*)k6.row(p);
            const int8_t* k70 = (const int8_t*)k7.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = (int8_t)k00[k];
                g00[1] = (int8_t)k10[k];
                g00[2] = (int8_t)k20[k];
                g00[3] = (int8_t)k30[k];
                g00[4] = (int8_t)k40[k];
                g00[5] = (int8_t)k50[k];
                g00[6] = (int8_t)k60[k];
                g00[7] = (int8_t)k70[k];

                g00 += 8;
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);

        int8_t* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);

        for (int p = 0; p < inch; p++)
        {
            const int8_t* k00 = (const int8_t*)k0.row(p);
            const int8_t* k10 = (const int8_t*)k1.row(p);
            const int8_t* k20 = (const int8_t*)k2.row(p);
            const int8_t* k30 = (const int8_t*)k3.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = (int8_t)k00[k];
                g00[1] = (int8_t)k10[k];
                g00[2] = (int8_t)k20[k];
                g00[3] = (int8_t)k30[k];

                g00 += 4;
            }
        }
    }
#else
    for (; q + 1 < outch; q += 2)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);

        int8_t* g00 = kernel_tm.channel(q / 2);

        for (int p = 0; p < inch; p++)
        {
            const int8_t* k00 = (const int8_t*)k0.row(p);
            const int8_t* k10 = (const int8_t*)k1.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];

                g00 += 2;
            }
        }
    }
#endif
    for (; q < outch; q++)
    {
        const Mat k0 = kernel.channel(q);

#if __riscv_vector
        int8_t* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + q % 4);
#else
        int8_t* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        for (int p = 0; p < inch; p++)
        {
            const int8_t* k00 = (const int8_t*)k0.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = (int8_t)k00[k];

                g00 += 1;
            }
        }
    }
}

static void convolution_im2col_sgemm_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 1u, 1, opt.workspace_allocator);
    {
        const int gap = w * stride_h - outw * stride_w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            int8_t* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const int8_t* sptr = (const int8_t*)img.row(dilation_h * u) + dilation_w * v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += stride_w;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_int8_rvv(bottom_im2col, top_blob, kernel, opt);
}
