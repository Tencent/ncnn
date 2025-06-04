// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void im2col_sgemm_int8_lsx(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 8u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    // permute
    Mat tmp;
#if __loongarch_sx
    if (inch >= 4)
    {
        if (size >= 2)
            tmp.create(2 * maxk, inch / 4 + inch % 4, size / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 4 + inch % 4, size, 4u, 4, opt.workspace_allocator);
    }
    else
#endif // __loongarch_sx
    {
        if (size >= 2)
            tmp.create(2 * maxk, inch, size / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else
            tmp.create(maxk, inch, size, 1u, 1, opt.workspace_allocator);
    }
    {
        int remain_size_start = 0;
        int nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            signed char* tmpptr = tmp.channel(i / 2);

            int q = 0;
#if __loongarch_sx
            for (; q + 3 < inch; q += 4)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img0[1];
                    tmpptr[5] = img1[1];
                    tmpptr[6] = img2[1];
                    tmpptr[7] = img3[1];
                    tmpptr += 8;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                }
            }
#endif // __loongarch_sx
            for (; q < inch; q++)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];

                    tmpptr += 2;

                    img0 += size;
                }
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            signed char* tmpptr = tmp.channel(i / 2 + i % 2);

            int q = 0;
#if __loongarch_sx
            for (; q + 3 < inch; q += 4)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr += 4;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                }
            }
#endif // __loongarch_sx
            for (; q < inch; q++)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];

                    tmpptr += 1;

                    img0 += size;
                }
            }
        }
    }

#if __loongarch_sx
    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);
        int* outptr2 = top_blob.channel(p + 2);
        int* outptr3 = top_blob.channel(p + 3);

        int i = 0;
        for (; i + 1 < size; i += 2)
        {
            const signed char* tmpptr = tmp.channel(i / 2);
            const signed char* kptr = kernel.channel(p / 4);

            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            __m128i _sum00 = __lsx_vreplgr2vr_w(0);
            __m128i _sum10 = __lsx_vreplgr2vr_w(0);

            if (nn4 > 0)
            {
                __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                __m128i _sum02 = __lsx_vreplgr2vr_w(0);
                __m128i _sum03 = __lsx_vreplgr2vr_w(0);
                __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                __m128i _sum12 = __lsx_vreplgr2vr_w(0);
                __m128i _sum13 = __lsx_vreplgr2vr_w(0);

                int j = 0;
                for (; j < nn4; j++)
                {
                    __m128i _val = __lsx_vld(tmpptr, 0);
                    __m128i _val01 = __lsx_vilvl_b(__lsx_vslti_b(_val, 0), _val);

                    __m128i _val0 = __lsx_vilvl_d(_val01, _val01);
                    __m128i _val1 = __lsx_vilvh_d(_val01, _val01);

                    __m128i _w01 = __lsx_vld(kptr, 0);
                    __m128i _extw01 = __lsx_vslti_b(_w01, 0);
                    __m128i _w0 = __lsx_vilvl_b(_extw01, _w01);
                    __m128i _w1 = __lsx_vilvh_b(_extw01, _w01);

                    __m128i _s00 = __lsx_vmul_h(_val0, _w0);
                    __m128i _s01 = __lsx_vmul_h(_val0, _w1);
                    __m128i _s10 = __lsx_vmul_h(_val1, _w0);
                    __m128i _s11 = __lsx_vmul_h(_val1, _w1);

                    __m128i _exts00 = __lsx_vslti_h(_s00, 0);
                    __m128i _exts01 = __lsx_vslti_h(_s01, 0);
                    __m128i _exts10 = __lsx_vslti_h(_s10, 0);
                    __m128i _exts11 = __lsx_vslti_h(_s11, 0);
                    __m128i _s00l = __lsx_vilvl_h(_exts00, _s00);
                    __m128i _s00h = __lsx_vilvh_h(_exts00, _s00);
                    __m128i _s01l = __lsx_vilvl_h(_exts01, _s01);
                    __m128i _s01h = __lsx_vilvh_h(_exts01, _s01);
                    __m128i _s10l = __lsx_vilvl_h(_exts10, _s10);
                    __m128i _s10h = __lsx_vilvh_h(_exts10, _s10);
                    __m128i _s11l = __lsx_vilvl_h(_exts11, _s11);
                    __m128i _s11h = __lsx_vilvh_h(_exts11, _s11);

                    _sum00 = __lsx_vadd_w(_sum00, _s00l);
                    _sum01 = __lsx_vadd_w(_sum01, _s00h);
                    _sum02 = __lsx_vadd_w(_sum02, _s01l);
                    _sum03 = __lsx_vadd_w(_sum03, _s01h);
                    _sum10 = __lsx_vadd_w(_sum10, _s10l);
                    _sum11 = __lsx_vadd_w(_sum11, _s10h);
                    _sum12 = __lsx_vadd_w(_sum12, _s11l);
                    _sum13 = __lsx_vadd_w(_sum13, _s11h);

                    tmpptr += 8;
                    kptr += 16;
                }

                // transpose 4x4
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = __lsx_vilvl_w(_sum01, _sum00);
                    _tmp1 = __lsx_vilvl_w(_sum03, _sum02);
                    _tmp2 = __lsx_vilvh_w(_sum01, _sum00);
                    _tmp3 = __lsx_vilvh_w(_sum03, _sum02);
                    _sum00 = __lsx_vilvl_d(_tmp1, _tmp0);
                    _sum01 = __lsx_vilvh_d(_tmp1, _tmp0);
                    _sum02 = __lsx_vilvl_d(_tmp3, _tmp2);
                    _sum03 = __lsx_vilvh_d(_tmp3, _tmp2);
                }
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = __lsx_vilvl_w(_sum11, _sum10);
                    _tmp1 = __lsx_vilvl_w(_sum13, _sum12);
                    _tmp2 = __lsx_vilvh_w(_sum11, _sum10);
                    _tmp3 = __lsx_vilvh_w(_sum13, _sum12);
                    _sum10 = __lsx_vilvl_d(_tmp1, _tmp0);
                    _sum11 = __lsx_vilvh_d(_tmp1, _tmp0);
                    _sum12 = __lsx_vilvl_d(_tmp3, _tmp2);
                    _sum13 = __lsx_vilvh_d(_tmp3, _tmp2);
                }

                _sum00 = __lsx_vadd_w(_sum00, _sum01);
                _sum02 = __lsx_vadd_w(_sum02, _sum03);
                _sum10 = __lsx_vadd_w(_sum10, _sum11);
                _sum12 = __lsx_vadd_w(_sum12, _sum13);

                _sum00 = __lsx_vadd_w(_sum00, _sum02);
                _sum10 = __lsx_vadd_w(_sum10, _sum12);
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                __m128i _val0 = __lsx_vreplgr2vr_h(tmpptr[0]);
                __m128i _val1 = __lsx_vreplgr2vr_h(tmpptr[1]);
                __m128i _val = __lsx_vilvl_d(_val1, _val0);

                __m128i _w = __lsx_vld(kptr, 0);
                __m128i _w16 = __lsx_vilvl_b(__lsx_vslti_b(_w, 0), _w);

                _w16 = __lsx_vilvl_d(_w16, _w16);

                __m128i _s0 = __lsx_vmul_h(_val, _w16);
                __m128i _exts0 = __lsx_vslti_h(_s0, 0);
                __m128i _s0l = __lsx_vilvl_h(_exts0, _s0);
                __m128i _s0h = __lsx_vilvh_h(_exts0, _s0);

                _sum00 = __lsx_vadd_w(_sum00, _s0l);
                _sum10 = __lsx_vadd_w(_sum10, _s0h);

                tmpptr += 2;
                kptr += 4;
            }

            int sum[8];
            __lsx_vst(_sum00, sum, 0);
            __lsx_vst(_sum10, sum + 4, 0);

            outptr0[0] = sum[0];
            outptr1[0] = sum[1];
            outptr2[0] = sum[2];
            outptr3[0] = sum[3];
            outptr0[1] = sum[4];
            outptr1[1] = sum[5];
            outptr2[1] = sum[6];
            outptr3[1] = sum[7];
            outptr0 += 2;
            outptr1 += 2;
            outptr2 += 2;
            outptr3 += 2;
        }
        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
            const signed char* kptr = kernel.channel(p / 4);

            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            __m128i _sum0 = __lsx_vreplgr2vr_w(0);

            if (nn4 > 0)
            {
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                __m128i _sum2 = __lsx_vreplgr2vr_w(0);
                __m128i _sum3 = __lsx_vreplgr2vr_w(0);

                int j = 0;
                for (; j < nn4; j++)
                {
                    __m128i _val = __lsx_vld(tmpptr, 0);
                    __m128i _val16 = __lsx_vilvl_b(__lsx_vslti_b(_val, 0), _val);

                    _val16 = __lsx_vilvl_d(_val16, _val16);

                    __m128i _w01 = __lsx_vld(kptr, 0);
                    __m128i _extw01 = __lsx_vslti_b(_w01, 0);
                    __m128i _w0 = __lsx_vilvl_b(_extw01, _w01);
                    __m128i _w1 = __lsx_vilvh_b(_extw01, _w01);

                    __m128i _s0 = __lsx_vmul_h(_val16, _w0);
                    __m128i _s1 = __lsx_vmul_h(_val16, _w1);

                    __m128i _exts0 = __lsx_vslti_h(_s0, 0);
                    __m128i _exts1 = __lsx_vslti_h(_s1, 0);
                    __m128i _s0l = __lsx_vilvl_h(_exts0, _s0);
                    __m128i _s0h = __lsx_vilvh_h(_exts0, _s0);
                    __m128i _s1l = __lsx_vilvl_h(_exts1, _s1);
                    __m128i _s1h = __lsx_vilvh_h(_exts1, _s1);

                    _sum0 = __lsx_vadd_w(_sum0, _s0l);
                    _sum1 = __lsx_vadd_w(_sum1, _s0h);
                    _sum2 = __lsx_vadd_w(_sum2, _s1l);
                    _sum3 = __lsx_vadd_w(_sum3, _s1h);

                    tmpptr += 4;
                    kptr += 16;
                }

                // transpose 4x4
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = __lsx_vilvl_w(_sum1, _sum0);
                    _tmp1 = __lsx_vilvl_w(_sum3, _sum2);
                    _tmp2 = __lsx_vilvh_w(_sum1, _sum0);
                    _tmp3 = __lsx_vilvh_w(_sum3, _sum2);
                    _sum0 = __lsx_vilvl_d(_tmp1, _tmp0);
                    _sum1 = __lsx_vilvh_d(_tmp1, _tmp0);
                    _sum2 = __lsx_vilvl_d(_tmp3, _tmp2);
                    _sum3 = __lsx_vilvh_d(_tmp3, _tmp2);
                }

                _sum0 = __lsx_vadd_w(_sum0, _sum1);
                _sum2 = __lsx_vadd_w(_sum2, _sum3);
                _sum0 = __lsx_vadd_w(_sum0, _sum2);
            }
            int j = 0;
            for (; j < nn1; j++)
            {
                __m128i _val = __lsx_vreplgr2vr_h(tmpptr[0]);

                __m128i _w = __lsx_vld(kptr, 0);
                __m128i _w16 = __lsx_vilvl_b(__lsx_vslti_b(_w, 0), _w);

                __m128i _s0 = __lsx_vmul_h(_val, _w16);
                __m128i _s032 = __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0);

                _sum0 = __lsx_vadd_w(_sum0, _s032);

                tmpptr += 1;
                kptr += 4;
            }

            int sum[4];
            __lsx_vst(_sum0, sum, 0);

            outptr0[0] = sum[0];
            outptr1[0] = sum[1];
            outptr2[0] = sum[2];
            outptr3[0] = sum[3];
            outptr0 += 1;
            outptr1 += 1;
            outptr2 += 1;
            outptr3 += 1;
        }
    }
#else // __loongarch_sx
    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);

        int i = 0;
        for (; i + 1 < size; i += 2)
        {
            const signed char* tmpptr = tmp.channel(i / 2);
            const signed char* kptr = kernel.channel(p / 2);

            int sum00 = 0;
            int sum01 = 0;
            int sum10 = 0;
            int sum11 = 0;

            int nn1 = inch * maxk;

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val0 = tmpptr[0];
                signed char val1 = tmpptr[1];
                signed char w0 = kptr[0];
                signed char w1 = kptr[1];

                sum00 += val0 * w0;
                sum01 += val1 * w0;
                sum10 += val0 * w1;
                sum11 += val1 * w1;

                tmpptr += 2;
                kptr += 2;
            }

            outptr0[0] = sum00;
            outptr0[1] = sum01;
            outptr1[0] = sum10;
            outptr1[1] = sum11;
            outptr0 += 2;
            outptr1 += 2;
        }
        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
            const signed char* kptr = kernel.channel(p / 2);

            int sum00 = 0;
            int sum10 = 0;

            int nn1 = inch * maxk;

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val0 = tmpptr[0];
                signed char w0 = kptr[0];
                signed char w1 = kptr[1];

                sum00 += val0 * w0;
                sum10 += val0 * w1;

                tmpptr += 1;
                kptr += 2;
            }

            outptr0[0] = sum00;
            outptr1[0] = sum10;
            outptr0 += 1;
            outptr1 += 1;
        }
    }
#endif // __loongarch_sx

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr0 = top_blob.channel(p);

        int i = 0;
        for (; i + 1 < size; i += 2)
        {
            const signed char* tmpptr = tmp.channel(i / 2);
#if __loongarch_sx
            const signed char* kptr = kernel.channel(p / 4 + p % 4);
#else
            const signed char* kptr = kernel.channel(p / 2 + p % 2);
#endif

            int sum0 = 0;
            int sum1 = 0;

#if __loongarch_sx
            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            if (nn4 > 0)
            {
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);

                int j = 0;
                for (; j < nn4; j++)
                {
                    __m128i _val = __lsx_vld(tmpptr, 0);
                    __m128i _val16 = __lsx_vilvl_b(__lsx_vslti_b(_val, 0), _val);

                    __m128i _w = __lsx_vld(kptr, 0);
                    __m128i _w16 = __lsx_vilvl_b(__lsx_vslti_b(_w, 0), _w);

                    _w16 = __lsx_vilvl_d(_w16, _w16);

                    __m128i _s0 = __lsx_vmul_h(_val16, _w16);
                    __m128i _exts0 = __lsx_vslti_h(_s0, 0);
                    __m128i _s0l = __lsx_vilvl_h(_exts0, _s0);
                    __m128i _s0h = __lsx_vilvh_h(_exts0, _s0);

                    _sum0 = __lsx_vadd_w(_sum0, _s0l);
                    _sum1 = __lsx_vadd_w(_sum1, _s0h);

                    tmpptr += 8;
                    kptr += 4;
                }

                sum0 = __lsx_reduce_add_w(_sum0);
                sum1 = __lsx_reduce_add_w(_sum1);
            }
#else
            int nn1 = inch * maxk;
#endif // __loongarch_sx

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val0 = tmpptr[0];
                signed char val1 = tmpptr[1];
                signed char w = kptr[0];

                sum0 += val0 * w;
                sum1 += val1 * w;

                tmpptr += 2;
                kptr += 1;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0 += 2;
        }
        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
#if __loongarch_sx
            const signed char* kptr = kernel.channel(p / 4 + p % 4);
#else
            const signed char* kptr = kernel.channel(p / 2 + p % 2);
#endif

            int sum = 0;

#if __loongarch_sx
            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            if (nn4 > 0)
            {
                __m128i _sum = __lsx_vreplgr2vr_w(0);

                int j = 0;
                for (; j < nn4; j++)
                {
                    __m128i _val = __lsx_vld(tmpptr, 0);
                    __m128i _val16 = __lsx_vilvl_b(__lsx_vslti_b(_val, 0), _val);

                    __m128i _w = __lsx_vld(kptr, 0);
                    __m128i _w16 = __lsx_vilvl_b(__lsx_vslti_b(_w, 0), _w);

                    __m128i _s0 = __lsx_vmul_h(_val16, _w16);
                    __m128i _s032 = __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0);

                    _sum = __lsx_vadd_w(_sum, _s032);

                    tmpptr += 4;
                    kptr += 4;
                }

                sum = __lsx_reduce_add_w(_sum);
            }
#else
            int nn1 = inch * maxk;
#endif // __loongarch_sx

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val = tmpptr[0];
                signed char w = kptr[0];

                sum += val * w;

                tmpptr += 1;
                kptr += 1;
            }

            outptr0[0] = sum;
            outptr0 += 1;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_int8_lsx(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4a-4b-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __loongarch_sx
    if (outch >= 4)
    {
        if (inch >= 4)
            kernel_tm.create(16 * maxk, inch / 4 + inch % 4, outch / 4 + outch % 4, (size_t)1u);
        else
            kernel_tm.create(4 * maxk, inch, outch / 4 + outch % 4, (size_t)1u);
    }
#else
    if (outch >= 2)
    {
        kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2, (size_t)1u);
    }
#endif // __loongarch_sx
    else
    {
#if __loongarch_sx
        if (inch >= 4)
            kernel_tm.create(4 * maxk, inch / 4 + inch % 4, outch, (size_t)1u);
        else
#endif // __loongarch_sx
        {
            kernel_tm.create(1 * maxk, inch, outch, (size_t)1u);
        }
    }

    int q = 0;
#if __loongarch_sx
    for (; q + 3 < outch; q += 4)
    {
        signed char* g00 = kernel_tm.channel(q / 4);

        int p = 0;
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const signed char* k00 = kernel.channel(q + i).row<const signed char>(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#else  // __loongarch_sx
    for (; q + 1 < outch; q += 2)
    {
        signed char* g00 = kernel_tm.channel(q / 2);

        int p = 0;
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 2; i++)
                {
                    const signed char* k00 = kernel.channel(q + i).row<const signed char>(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __loongarch_sx
    for (; q < outch; q++)
    {
#if __loongarch_sx
        signed char* g00 = kernel_tm.channel(q / 4 + q % 4);
#else
        signed char* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __loongarch_sx
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 4; j++)
                {
                    const signed char* k00 = kernel.channel(q).row<const signed char>(p + j);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
#endif // __loongarch_sx
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k00 = kernel.channel(q).row<const signed char>(p);
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void convolution_im2col_sgemm_int8_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
            signed char* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const signed char* sptr = img.row<const signed char>(dilation_h * u) + dilation_w * v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 3 < outw; j += 4)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[stride_w];
                            ptr[2] = sptr[stride_w * 2];
                            ptr[3] = sptr[stride_w * 3];

                            sptr += stride_w * 4;
                            ptr += 4;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[stride_w];

                            sptr += stride_w * 2;
                            ptr += 2;
                        }
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

    im2col_sgemm_int8_lsx(bottom_im2col, top_blob, kernel, opt);
}
