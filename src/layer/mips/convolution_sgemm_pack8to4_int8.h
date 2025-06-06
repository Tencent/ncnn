// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void im2col_sgemm_pack8to4_int8_msa(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
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
                    v16i8 _v = __msa_ld_b(img0, 0);
                    __msa_st_b(_v, tmpptr, 0);
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

            v4i32 _sum00 = __msa_fill_w(0);
            v4i32 _sum01 = __msa_fill_w(0);
            v4i32 _sum02 = __msa_fill_w(0);
            v4i32 _sum03 = __msa_fill_w(0);
            v4i32 _sum10 = __msa_fill_w(0);
            v4i32 _sum11 = __msa_fill_w(0);
            v4i32 _sum12 = __msa_fill_w(0);
            v4i32 _sum13 = __msa_fill_w(0);

            int j = 0;
            for (; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 64);
                __builtin_prefetch(kptr + 128);
                v16i8 _val01 = __msa_ld_b(tmpptr, 0);
                v16i8 _extval01 = __msa_clti_s_b(_val01, 0);
                v8i16 _val0 = (v8i16)__msa_ilvr_b(_extval01, _val01);
                v8i16 _val1 = (v8i16)__msa_ilvl_b(_extval01, _val01);

                v16i8 _w01 = __msa_ld_b(kptr, 0);
                v16i8 _w23 = __msa_ld_b(kptr + 16, 0);
                v16i8 _extw01 = __msa_clti_s_b(_w01, 0);
                v16i8 _extw23 = __msa_clti_s_b(_w23, 0);
                v8i16 _w0 = (v8i16)__msa_ilvr_b(_extw01, _w01);
                v8i16 _w1 = (v8i16)__msa_ilvl_b(_extw01, _w01);
                v8i16 _w2 = (v8i16)__msa_ilvr_b(_extw23, _w23);
                v8i16 _w3 = (v8i16)__msa_ilvl_b(_extw23, _w23);

                v8i16 _s00 = __msa_mulv_h(_val0, _w0);
                v8i16 _s01 = __msa_mulv_h(_val0, _w1);
                v8i16 _s02 = __msa_mulv_h(_val0, _w2);
                v8i16 _s03 = __msa_mulv_h(_val0, _w3);
                v8i16 _s10 = __msa_mulv_h(_val1, _w0);
                v8i16 _s11 = __msa_mulv_h(_val1, _w1);
                v8i16 _s12 = __msa_mulv_h(_val1, _w2);
                v8i16 _s13 = __msa_mulv_h(_val1, _w3);

                _sum00 = __msa_addv_w(_sum00, __msa_hadd_s_w(_s00, _s00));
                _sum01 = __msa_addv_w(_sum01, __msa_hadd_s_w(_s01, _s01));
                _sum02 = __msa_addv_w(_sum02, __msa_hadd_s_w(_s02, _s02));
                _sum03 = __msa_addv_w(_sum03, __msa_hadd_s_w(_s03, _s03));
                _sum10 = __msa_addv_w(_sum10, __msa_hadd_s_w(_s10, _s10));
                _sum11 = __msa_addv_w(_sum11, __msa_hadd_s_w(_s11, _s11));
                _sum12 = __msa_addv_w(_sum12, __msa_hadd_s_w(_s12, _s12));
                _sum13 = __msa_addv_w(_sum13, __msa_hadd_s_w(_s13, _s13));

                tmpptr += 16;
                kptr += 32;
            }

            // transpose 4x4
            {
                v4i32 _tmp0, _tmp1, _tmp2, _tmp3;
                _tmp0 = __msa_ilvr_w(_sum01, _sum00);
                _tmp1 = __msa_ilvr_w(_sum03, _sum02);
                _tmp2 = __msa_ilvl_w(_sum01, _sum00);
                _tmp3 = __msa_ilvl_w(_sum03, _sum02);
                _sum00 = (v4i32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp0);
                _sum01 = (v4i32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp0);
                _sum02 = (v4i32)__msa_ilvr_d((v2i64)_tmp3, (v2i64)_tmp2);
                _sum03 = (v4i32)__msa_ilvl_d((v2i64)_tmp3, (v2i64)_tmp2);
            }
            {
                v4i32 _tmp0, _tmp1, _tmp2, _tmp3;
                _tmp0 = __msa_ilvr_w(_sum11, _sum10);
                _tmp1 = __msa_ilvr_w(_sum13, _sum12);
                _tmp2 = __msa_ilvl_w(_sum11, _sum10);
                _tmp3 = __msa_ilvl_w(_sum13, _sum12);
                _sum10 = (v4i32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp0);
                _sum11 = (v4i32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp0);
                _sum12 = (v4i32)__msa_ilvr_d((v2i64)_tmp3, (v2i64)_tmp2);
                _sum13 = (v4i32)__msa_ilvl_d((v2i64)_tmp3, (v2i64)_tmp2);
            }

            _sum00 = __msa_addv_w(_sum00, _sum01);
            _sum02 = __msa_addv_w(_sum02, _sum03);
            _sum10 = __msa_addv_w(_sum10, _sum11);
            _sum12 = __msa_addv_w(_sum12, _sum13);

            _sum00 = __msa_addv_w(_sum00, _sum02);
            _sum10 = __msa_addv_w(_sum10, _sum12);

            __msa_st_w(_sum00, outptr0, 0);
            __msa_st_w(_sum10, outptr0 + 4, 0);
            outptr0 += 8;
        }
        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
            const signed char* kptr = kernel.channel(p);

            int nn = inch * maxk; // inch always > 0

            v4i32 _sum0 = __msa_fill_w(0);
            v4i32 _sum1 = __msa_fill_w(0);
            v4i32 _sum2 = __msa_fill_w(0);
            v4i32 _sum3 = __msa_fill_w(0);

            int j = 0;
            for (; j < nn; j++)
            {
                __builtin_prefetch(tmpptr + 32);
                __builtin_prefetch(kptr + 128);
                v16i8 _val = __msa_ld_b(tmpptr, 0);
                v8i16 _val16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_val, 0), _val);

                v16i8 _w01 = __msa_ld_b(kptr, 0);
                v16i8 _w23 = __msa_ld_b(kptr + 16, 0);
                v16i8 _extw01 = __msa_clti_s_b(_w01, 0);
                v16i8 _extw23 = __msa_clti_s_b(_w23, 0);
                v8i16 _w0 = (v8i16)__msa_ilvr_b(_extw01, _w01);
                v8i16 _w1 = (v8i16)__msa_ilvl_b(_extw01, _w01);
                v8i16 _w2 = (v8i16)__msa_ilvr_b(_extw23, _w23);
                v8i16 _w3 = (v8i16)__msa_ilvl_b(_extw23, _w23);

                v8i16 _s0 = __msa_mulv_h(_val16, _w0);
                v8i16 _s1 = __msa_mulv_h(_val16, _w1);
                v8i16 _s2 = __msa_mulv_h(_val16, _w2);
                v8i16 _s3 = __msa_mulv_h(_val16, _w3);

                _sum0 = __msa_addv_w(_sum0, __msa_hadd_s_w(_s0, _s0));
                _sum1 = __msa_addv_w(_sum1, __msa_hadd_s_w(_s1, _s1));
                _sum2 = __msa_addv_w(_sum2, __msa_hadd_s_w(_s2, _s2));
                _sum3 = __msa_addv_w(_sum3, __msa_hadd_s_w(_s3, _s3));

                tmpptr += 8;
                kptr += 32;
            }

            // transpose 4x4
            {
                v4i32 _tmp0, _tmp1, _tmp2, _tmp3;
                _tmp0 = __msa_ilvr_w(_sum1, _sum0);
                _tmp1 = __msa_ilvr_w(_sum3, _sum2);
                _tmp2 = __msa_ilvl_w(_sum1, _sum0);
                _tmp3 = __msa_ilvl_w(_sum3, _sum2);
                _sum0 = (v4i32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp0);
                _sum1 = (v4i32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp0);
                _sum2 = (v4i32)__msa_ilvr_d((v2i64)_tmp3, (v2i64)_tmp2);
                _sum3 = (v4i32)__msa_ilvl_d((v2i64)_tmp3, (v2i64)_tmp2);
            }

            _sum0 = __msa_addv_w(_sum0, _sum1);
            _sum2 = __msa_addv_w(_sum2, _sum3);

            _sum0 = __msa_addv_w(_sum0, _sum2);

            __msa_st_w(_sum0, outptr0, 0);
            outptr0 += 4;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack8to4_int8_msa(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
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

static void convolution_im2col_sgemm_pack8to4_int8_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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

    im2col_sgemm_pack8to4_int8_msa(bottom_im2col, top_blob, kernel, opt);
}
