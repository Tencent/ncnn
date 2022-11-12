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

static void im2col_sgemm_pack8to4_int8_lsx(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 8u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    // permute
    Mat tmp;
    if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 8u, 8, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 8u, 8, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        int nn_size = size >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            int64_t* tmpptr = tmp.channel(i / 2);

            for (int q = 0; q < inch; q++)
            {
                const int64_t* img0 = (const int64_t*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    __m128i _v = __lsx_vld(img0, 0);
                    __lsx_vst(_v, tmpptr, 0);
                    tmpptr += 2;
                    img0 += size;
                }
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            int64_t* tmpptr = tmp.channel(i / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                const int64_t* img0 = (const int64_t*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr += 1;
                    img0 += size;
                }
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        int* outptr0 = top_blob.channel(p);

        int i = 0;
        for (; i + 1 < size; i += 2)
        {
            const signed char* tmpptr = tmp.channel(i / 2);
            const signed char* kptr = kernel.channel(p);

            int nn = inch * maxk; // inch always > 0

            __m128i _sum00 = __lsx_vreplgr2vr_w(0);
            __m128i _sum01 = __lsx_vreplgr2vr_w(0);
            __m128i _sum02 = __lsx_vreplgr2vr_w(0);
            __m128i _sum03 = __lsx_vreplgr2vr_w(0);
            __m128i _sum10 = __lsx_vreplgr2vr_w(0);
            __m128i _sum11 = __lsx_vreplgr2vr_w(0);
            __m128i _sum12 = __lsx_vreplgr2vr_w(0);
            __m128i _sum13 = __lsx_vreplgr2vr_w(0);

            int j = 0;
            for (; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 64);
                __builtin_prefetch(kptr + 128);
                __m128i _val01 = __lsx_vld(tmpptr, 0);
                __m128i _extval01 = __lsx_vslti_b(_val01, 0);
                __m128i _val0 = __lsx_vilvl_b(_extval01, _val01);
                __m128i _val1 = __lsx_vilvh_b(_extval01, _val01);

                __m128i _w01 = __lsx_vld(kptr, 0);
                __m128i _w23 = __lsx_vld(kptr + 16, 0);
                __m128i _extw01 = __lsx_vslti_b(_w01, 0);
                __m128i _extw23 = __lsx_vslti_b(_w23, 0);
                __m128i _w0 = __lsx_vilvl_b(_extw01, _w01);
                __m128i _w1 = __lsx_vilvh_b(_extw01, _w01);
                __m128i _w2 = __lsx_vilvl_b(_extw23, _w23);
                __m128i _w3 = __lsx_vilvh_b(_extw23, _w23);

                __m128i _s00 = __lsx_vmul_h(_val0, _w0);
                __m128i _s01 = __lsx_vmul_h(_val0, _w1);
                __m128i _s02 = __lsx_vmul_h(_val0, _w2);
                __m128i _s03 = __lsx_vmul_h(_val0, _w3);
                __m128i _s10 = __lsx_vmul_h(_val1, _w0);
                __m128i _s11 = __lsx_vmul_h(_val1, _w1);
                __m128i _s12 = __lsx_vmul_h(_val1, _w2);
                __m128i _s13 = __lsx_vmul_h(_val1, _w3);

                _sum00 = __lsx_vadd_w(_sum00, __lsx_vhaddw_w_h(_s00, _s00));
                _sum01 = __lsx_vadd_w(_sum01, __lsx_vhaddw_w_h(_s01, _s01));
                _sum02 = __lsx_vadd_w(_sum02, __lsx_vhaddw_w_h(_s02, _s02));
                _sum03 = __lsx_vadd_w(_sum03, __lsx_vhaddw_w_h(_s03, _s03));
                _sum10 = __lsx_vadd_w(_sum10, __lsx_vhaddw_w_h(_s10, _s10));
                _sum11 = __lsx_vadd_w(_sum11, __lsx_vhaddw_w_h(_s11, _s11));
                _sum12 = __lsx_vadd_w(_sum12, __lsx_vhaddw_w_h(_s12, _s12));
                _sum13 = __lsx_vadd_w(_sum13, __lsx_vhaddw_w_h(_s13, _s13));

                tmpptr += 16;
                kptr += 32;
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

            __lsx_vst(_sum00, outptr0, 0);
            __lsx_vst(_sum10, outptr0 + 4, 0);
            outptr0 += 8;
        }
        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
            const signed char* kptr = kernel.channel(p);

            int nn = inch * maxk; // inch always > 0

            __m128i _sum0 = __lsx_vreplgr2vr_w(0);
            __m128i _sum1 = __lsx_vreplgr2vr_w(0);
            __m128i _sum2 = __lsx_vreplgr2vr_w(0);
            __m128i _sum3 = __lsx_vreplgr2vr_w(0);

            int j = 0;
            for (; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 32);
                __builtin_prefetch(kptr + 128);
                __m128i _val = __lsx_vld(tmpptr, 0);
                __m128i _val16 = __lsx_vilvl_b(__lsx_vslti_b(_val, 0), _val);

                __m128i _w01 = __lsx_vld(kptr, 0);
                __m128i _w23 = __lsx_vld(kptr + 16, 0);
                __m128i _extw01 = __lsx_vslti_b(_w01, 0);
                __m128i _extw23 = __lsx_vslti_b(_w23, 0);
                __m128i _w0 = __lsx_vilvl_b(_extw01, _w01);
                __m128i _w1 = __lsx_vilvh_b(_extw01, _w01);
                __m128i _w2 = __lsx_vilvl_b(_extw23, _w23);
                __m128i _w3 = __lsx_vilvh_b(_extw23, _w23);

                __m128i _s0 = __lsx_vmul_h(_val16, _w0);
                __m128i _s1 = __lsx_vmul_h(_val16, _w1);
                __m128i _s2 = __lsx_vmul_h(_val16, _w2);
                __m128i _s3 = __lsx_vmul_h(_val16, _w3);

                _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
                _sum2 = __lsx_vadd_w(_sum2, __lsx_vhaddw_w_h(_s2, _s2));
                _sum3 = __lsx_vadd_w(_sum3, __lsx_vhaddw_w_h(_s3, _s3));

                tmpptr += 8;
                kptr += 32;
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

            __lsx_vst(_sum0, outptr0, 0);
            outptr0 += 4;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack8to4_int8_lsx(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8a-4b-maxk-inch/8a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(32 * maxk, inch / 8, outch / 4, (size_t)1u);

    for (int q = 0; q + 3 < outch; q += 4)
    {
        signed char* g00 = kernel_tm.channel(q / 4);

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_pack8to4_int8_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 8u, 8, opt.workspace_allocator);
    {
        const int gap = w * stride_h - outw * stride_w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            int64_t* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const int64_t* sptr = img.row<const int64_t>(dilation_h * u) + dilation_w * v;

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

    im2col_sgemm_pack8to4_int8_lsx(bottom_im2col, top_blob, kernel, opt);
}
