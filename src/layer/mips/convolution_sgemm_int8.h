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

#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
void im2col_sgemm_int8_loongson_mmi(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt);
void convolution_im2col_sgemm_transform_kernel_int8_loongson_mmi(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);
#endif

static void im2col_sgemm_int8_msa(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
    {
        im2col_sgemm_int8_loongson_mmi(bottom_im2col, top_blob, kernel, opt);
        return;
    }
#endif

    // Mat bottom_im2col(size, maxk, inch, 8u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    // permute
    Mat tmp;
#if __mips_msa || __mips_loongson_mmi
    if (inch >= 4)
    {
        if (size >= 2)
            tmp.create(2 * maxk, inch / 4 + inch % 4, size / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 4 + inch % 4, size, 4u, 4, opt.workspace_allocator);
    }
    else
#endif // __mips_msa || __mips_loongson_mmi
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
#if __mips_msa || __mips_loongson_mmi
            for (; q + 3 < inch; q += 4)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;

                for (int k = 0; k < maxk; k++)
                {
#if __mips_loongson_mmi
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img0[1];
                    tmpptr[3] = img1[1];
                    tmpptr[4] = img2[0];
                    tmpptr[5] = img3[0];
                    tmpptr[6] = img2[1];
                    tmpptr[7] = img3[1];
#else  // __mips_loongson_mmi
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img0[1];
                    tmpptr[5] = img1[1];
                    tmpptr[6] = img2[1];
                    tmpptr[7] = img3[1];
#endif // __mips_loongson_mmi
                    tmpptr += 8;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                }
            }
#endif // __mips_msa || __mips_loongson_mmi
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
#if __mips_msa || __mips_loongson_mmi
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
#endif // __mips_msa || __mips_loongson_mmi
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

#if __mips_msa
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

            v4i32 _sum00 = __msa_fill_w(0);
            v4i32 _sum10 = __msa_fill_w(0);

            if (nn4 > 0)
            {
                v4i32 _sum01 = __msa_fill_w(0);
                v4i32 _sum02 = __msa_fill_w(0);
                v4i32 _sum03 = __msa_fill_w(0);
                v4i32 _sum11 = __msa_fill_w(0);
                v4i32 _sum12 = __msa_fill_w(0);
                v4i32 _sum13 = __msa_fill_w(0);

                int j = 0;
                for (; j < nn4; j++)
                {
                    v16i8 _val = __msa_ld_b(tmpptr, 0);
                    v8i16 _val01 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_val, 0), _val);

                    v8i16 _val0 = (v8i16)__msa_ilvr_d((v2i64)_val01, (v2i64)_val01);
                    v8i16 _val1 = (v8i16)__msa_ilvl_d((v2i64)_val01, (v2i64)_val01);

                    v16i8 _w01 = __msa_ld_b(kptr, 0);
                    v16i8 _extw01 = __msa_clti_s_b(_w01, 0);
                    v8i16 _w0 = (v8i16)__msa_ilvr_b(_extw01, _w01);
                    v8i16 _w1 = (v8i16)__msa_ilvl_b(_extw01, _w01);

                    v8i16 _s00 = __msa_mulv_h(_val0, _w0);
                    v8i16 _s01 = __msa_mulv_h(_val0, _w1);
                    v8i16 _s10 = __msa_mulv_h(_val1, _w0);
                    v8i16 _s11 = __msa_mulv_h(_val1, _w1);

                    v8i16 _exts00 = __msa_clti_s_h(_s00, 0);
                    v8i16 _exts01 = __msa_clti_s_h(_s01, 0);
                    v8i16 _exts10 = __msa_clti_s_h(_s10, 0);
                    v8i16 _exts11 = __msa_clti_s_h(_s11, 0);
                    v4i32 _s00l = (v4i32)__msa_ilvr_h(_exts00, _s00);
                    v4i32 _s00h = (v4i32)__msa_ilvl_h(_exts00, _s00);
                    v4i32 _s01l = (v4i32)__msa_ilvr_h(_exts01, _s01);
                    v4i32 _s01h = (v4i32)__msa_ilvl_h(_exts01, _s01);
                    v4i32 _s10l = (v4i32)__msa_ilvr_h(_exts10, _s10);
                    v4i32 _s10h = (v4i32)__msa_ilvl_h(_exts10, _s10);
                    v4i32 _s11l = (v4i32)__msa_ilvr_h(_exts11, _s11);
                    v4i32 _s11h = (v4i32)__msa_ilvl_h(_exts11, _s11);

                    _sum00 = __msa_addv_w(_sum00, _s00l);
                    _sum01 = __msa_addv_w(_sum01, _s00h);
                    _sum02 = __msa_addv_w(_sum02, _s01l);
                    _sum03 = __msa_addv_w(_sum03, _s01h);
                    _sum10 = __msa_addv_w(_sum10, _s10l);
                    _sum11 = __msa_addv_w(_sum11, _s10h);
                    _sum12 = __msa_addv_w(_sum12, _s11l);
                    _sum13 = __msa_addv_w(_sum13, _s11h);

                    tmpptr += 8;
                    kptr += 16;
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
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                v8i16 _val0 = __msa_fill_h(tmpptr[0]);
                v8i16 _val1 = __msa_fill_h(tmpptr[1]);
                v8i16 _val = (v8i16)__msa_ilvr_d((v2i64)_val1, (v2i64)_val0);

                v16i8 _w = __msa_ld_b(kptr, 0);
                v8i16 _w16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_w, 0), _w);

                _w16 = (v8i16)__msa_ilvr_d((v2i64)_w16, (v2i64)_w16);

                v8i16 _s0 = __msa_mulv_h(_val, _w16);
                v8i16 _exts0 = __msa_clti_s_h(_s0, 0);
                v4i32 _s0l = (v4i32)__msa_ilvr_h(_exts0, _s0);
                v4i32 _s0h = (v4i32)__msa_ilvl_h(_exts0, _s0);

                _sum00 = __msa_addv_w(_sum00, _s0l);
                _sum10 = __msa_addv_w(_sum10, _s0h);

                tmpptr += 2;
                kptr += 4;
            }

            int sum[8];
            __msa_st_w(_sum00, sum, 0);
            __msa_st_w(_sum10, sum + 4, 0);

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

            v4i32 _sum0 = __msa_fill_w(0);

            if (nn4 > 0)
            {
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);

                int j = 0;
                for (; j < nn4; j++)
                {
                    v16i8 _val = __msa_ld_b(tmpptr, 0);
                    v8i16 _val16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_val, 0), _val);

                    _val16 = (v8i16)__msa_ilvr_d((v2i64)_val16, (v2i64)_val16);

                    v16i8 _w01 = __msa_ld_b(kptr, 0);
                    v16i8 _extw01 = __msa_clti_s_b(_w01, 0);
                    v8i16 _w0 = (v8i16)__msa_ilvr_b(_extw01, _w01);
                    v8i16 _w1 = (v8i16)__msa_ilvl_b(_extw01, _w01);

                    v8i16 _s0 = __msa_mulv_h(_val16, _w0);
                    v8i16 _s1 = __msa_mulv_h(_val16, _w1);

                    v8i16 _exts0 = __msa_clti_s_h(_s0, 0);
                    v8i16 _exts1 = __msa_clti_s_h(_s1, 0);
                    v4i32 _s0l = (v4i32)__msa_ilvr_h(_exts0, _s0);
                    v4i32 _s0h = (v4i32)__msa_ilvl_h(_exts0, _s0);
                    v4i32 _s1l = (v4i32)__msa_ilvr_h(_exts1, _s1);
                    v4i32 _s1h = (v4i32)__msa_ilvl_h(_exts1, _s1);

                    _sum0 = __msa_addv_w(_sum0, _s0l);
                    _sum1 = __msa_addv_w(_sum1, _s0h);
                    _sum2 = __msa_addv_w(_sum2, _s1l);
                    _sum3 = __msa_addv_w(_sum3, _s1h);

                    tmpptr += 4;
                    kptr += 16;
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
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                v8i16 _val = __msa_fill_h(tmpptr[0]);

                v16i8 _w = __msa_ld_b(kptr, 0);
                v8i16 _w16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_w, 0), _w);

                v8i16 _s0 = __msa_mulv_h(_val, _w16);
                v4i32 _s032 = (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0);

                _sum0 = __msa_addv_w(_sum0, _s032);

                tmpptr += 1;
                kptr += 4;
            }

            int sum[4];
            __msa_st_w(_sum0, sum, 0);

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
#else // __mips_msa
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

#if __mips_loongson_mmi
            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            if (nn4 > 0)
            {
                int32x2_t _sum0 = __mmi_pzerow_s();
                int32x2_t _sum1 = __mmi_pzerow_s();

                double temp0 = 0;
                double temp1 = 0;
                double temp2 = 0;
                double temp3 = 0;
                double temp4 = 0;
                double temp5 = 0;
                double temp6 = 0;
                double temp7 = 0;

                uint64_t flag_0x44 = 0x44;
                uint64_t flag_0xee = 0xee;

                int j = 0;
                for (; j < nn4; j++)
                {
                    asm volatile(
                        "ld         $0, 32(%0)      \n" // __builtin_prefetch(tmpptr + 32);
                        "ld         $0, 32(%1)      \n" // __builtin_prefetch(kptr + 32);

                        "ldc1       %4, 0(%0)       \n" // int8x8_t _v = __mmi_pldb_s(tmpptr);
                        "ldc1       %6, 0(%1)       \n" // int8x8_t _k = __mmi_pldb_s(kptr);

                        "mtc1       $0, %8          \n" // int8x8_t _zero = __mmi_pzerob_s();
                        "pcmpgtb    %5, %8, %4      \n" // int8x8_t _extv = __mmi_pcmpgtb_s(_zero, _v);
                        "pcmpgtb    %7, %8, %6      \n" // int8x8_t _extk = __mmi_pcmpgtb_s(_zero, _k);

                        "punpcklbh  %8, %4, %5      \n" // int16x4_t _v0 = (int16x4_t)__mmi_punpcklbh_s(_v, _extv);
                        "punpckhbh  %9, %4, %5      \n" // int16x4_t _v1 = (int16x4_t)__mmi_punpckhbh_s(_v, _extv);
                        "punpcklbh  %10, %6, %7     \n" // int16x4_t _k0 = (int16x4_t)__mmi_punpcklbh_s(_k, _extk);
                        "punpckhbh  %11, %6, %7     \n" // int16x4_t _k1 = (int16x4_t)__mmi_punpckhbh_s(_k, _extk);

                        "pshufh     %4, %10, %24    \n" // int16x4_t _k0202 = __mmi_pshufh_s(_k0, 0x44);
                        "pshufh     %5, %10, %25    \n" // int16x4_t _k1313 = __mmi_pshufh_s(_k0, 0xee);
                        "pshufh     %6, %11, %24    \n" // int16x4_t _k4646 = __mmi_pshufh_s(_k1, 0x44);
                        "pshufh     %7, %11, %25    \n" // int16x4_t _k5757 = __mmi_pshufh_s(_k1, 0xee);

                        "pmaddhw    %4, %8, %4      \n" // int32x2_t _s0x = __mmi_pmaddhw(_v0, _k0202);
                        "pmaddhw    %5, %8, %5      \n" // int32x2_t _s1x = __mmi_pmaddhw(_v0, _k1313);
                        "pmaddhw    %6, %9, %6      \n" // int32x2_t _s0y = __mmi_pmaddhw(_v1, _k4646);
                        "pmaddhw    %7, %9, %7      \n" // int32x2_t _s1y = __mmi_pmaddhw(_v1, _k5757);

                        "paddw      %2, %2, %4      \n" // _sum0 = __mmi_paddw_s(_sum0, _s0x);
                        "paddw      %3, %3, %5      \n" // _sum1 = __mmi_paddw_s(_sum1, _s1x);
                        "paddw      %2, %2, %6      \n" // _sum0 = __mmi_paddw_s(_sum0, _s0y);
                        "paddw      %3, %3, %7      \n" // _sum1 = __mmi_paddw_s(_sum1, _s1y);

                        : "=r"(tmpptr), // %0
                        "=r"(kptr),   // %1
                        "=f"(_sum0),  // %2
                        "=f"(_sum1),  // %3
                        "=f"(temp0),  // %4
                        "=f"(temp1),  // %5
                        "=f"(temp2),  // %6
                        "=f"(temp3),  // %7
                        "=f"(temp4),  // %8
                        "=f"(temp5),  // %9
                        "=f"(temp6),  // %10
                        "=f"(temp7)   // %11
                        : "0"(tmpptr),
                        "1"(kptr),
                        "2"(_sum0),
                        "3"(_sum1),
                        "4"(temp0),
                        "5"(temp1),
                        "6"(temp2),
                        "7"(temp3),
                        "8"(temp4),
                        "9"(temp5),
                        "10"(temp6),
                        "11"(temp7),
                        "f"(flag_0x44), // %24
                        "f"(flag_0xee)  // %25
                        : "memory");

                    tmpptr += 8;
                    kptr += 8;
                }

                int sum[4];
                __mmi_pstw_s(sum, _sum0);
                __mmi_pstw_s(sum + 2, _sum1);

                sum00 = sum[0];
                sum01 = sum[1];
                sum10 = sum[2];
                sum11 = sum[3];
            }
#else  // __mips_loongson_mmi
            int nn1 = inch * maxk;
#endif // __mips_loongson_mmi

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

#if __mips_loongson_mmi
            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            if (nn4 > 0)
            {
                int32x2_t _sum01 = __mmi_pzerow_s();

                double temp0 = 0;
                double temp1 = 0;
                double temp2 = 0;
                double temp3 = 0;
                double temp4 = 0;
                double temp5 = 0;

                uint64_t flag_0x44 = 0x44;
                uint64_t flag_0xee = 0xee;

                int j = 0;
                for (; j < nn4; j++)
                {
                    asm volatile(
                        "ld         $0, 16(%0)      \n" // __builtin_prefetch(tmpptr + 16);
                        "ld         $0, 32(%1)      \n" // __builtin_prefetch(kptr + 32);

                        "ldc1       %3, 0(%0)       \n" // int8x8_t _v = __mmi_pldb_s(tmpptr);
                        "ldc1       %5, 0(%1)       \n" // int8x8_t _k = __mmi_pldb_s(kptr);

                        "mtc1       $0, %7          \n" // int8x8_t _zero = __mmi_pzerob_s();
                        "pcmpgtb    %4, %7, %3      \n" // int8x8_t _extv = __mmi_pcmpgtb_s(_zero, _v);
                        "pcmpgtb    %6, %7, %5      \n" // int8x8_t _extk = __mmi_pcmpgtb_s(_zero, _k);

                        "punpcklbh  %4, %3, %4      \n" // int16x4_t _v0 = (int16x4_t)__mmi_punpcklbh_s(_v, _extv);
                        "punpcklbh  %7, %5, %6      \n" // int16x4_t _k0 = (int16x4_t)__mmi_punpcklbh_s(_k, _extk);
                        "punpckhbh  %8, %5, %6      \n" // int16x4_t _k1 = (int16x4_t)__mmi_punpckhbh_s(_k, _extk);

                        "pshufh     %3, %4, %18     \n" // int16x4_t _v0101 = __mmi_pshufh_s(_v0, 0x44);
                        "pshufh     %4, %4, %19     \n" // int16x4_t _v2323 = __mmi_pshufh_s(_v0, 0xee);

                        "pmaddhw    %3, %3, %7      \n" // int32x2_t _s01x = __mmi_pmaddhw(_v0101, _k0);
                        "pmaddhw    %4, %4, %8      \n" // int32x2_t _s01y = __mmi_pmaddhw(_v2323, _k1);

                        "paddw      %2, %2, %3      \n" // _sum01 = __mmi_paddw_s(_sum01, _s01x);
                        "paddw      %2, %2, %4      \n" // _sum01 = __mmi_paddw_s(_sum01, _s01y);

                        : "=r"(tmpptr), // %0
                        "=r"(kptr),   // %1
                        "=f"(_sum01), // %2
                        "=f"(temp0),  // %3
                        "=f"(temp1),  // %4
                        "=f"(temp2),  // %5
                        "=f"(temp3),  // %6
                        "=f"(temp4),  // %7
                        "=f"(temp5)   // %8
                        : "0"(tmpptr),
                        "1"(kptr),
                        "2"(_sum01),
                        "3"(temp0),
                        "4"(temp1),
                        "5"(temp2),
                        "6"(temp3),
                        "7"(temp4),
                        "8"(temp5),
                        "f"(flag_0x44), // %18
                        "f"(flag_0xee)  // %19
                        : "memory");

                    tmpptr += 4;
                    kptr += 8;
                }

                int sum[2];
                __mmi_pstw_s(sum, _sum01);

                sum00 = sum[0];
                sum10 = sum[1];
            }
#else  // __mips_loongson_mmi
            int nn1 = inch * maxk;
#endif // __mips_loongson_mmi

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
#endif // __mips_msa

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr0 = top_blob.channel(p);

        int i = 0;
        for (; i + 1 < size; i += 2)
        {
            const signed char* tmpptr = tmp.channel(i / 2);
#if __mips_msa
            const signed char* kptr = kernel.channel(p / 4 + p % 4);
#else
            const signed char* kptr = kernel.channel(p / 2 + p % 2);
#endif

            int sum0 = 0;
            int sum1 = 0;

#if __mips_msa
            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            if (nn4 > 0)
            {
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);

                int j = 0;
                for (; j < nn4; j++)
                {
                    v16i8 _val = __msa_ld_b(tmpptr, 0);
                    v8i16 _val16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_val, 0), _val);

                    v16i8 _w = __msa_ld_b(kptr, 0);
                    v8i16 _w16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_w, 0), _w);

                    _w16 = (v8i16)__msa_ilvr_d((v2i64)_w16, (v2i64)_w16);

                    v8i16 _s0 = __msa_mulv_h(_val16, _w16);
                    v8i16 _exts0 = __msa_clti_s_h(_s0, 0);
                    v4i32 _s0l = (v4i32)__msa_ilvr_h(_exts0, _s0);
                    v4i32 _s0h = (v4i32)__msa_ilvl_h(_exts0, _s0);

                    _sum0 = __msa_addv_w(_sum0, _s0l);
                    _sum1 = __msa_addv_w(_sum1, _s0h);

                    tmpptr += 8;
                    kptr += 4;
                }

                sum0 = _sum0[0] + _sum0[1] + _sum0[2] + _sum0[3];
                sum1 = _sum1[0] + _sum1[1] + _sum1[2] + _sum1[3];
            }
#elif __mips_loongson_mmi
            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            if (nn4 > 0)
            {
                int32x2_t _sum01 = __mmi_pzerow_s();

                double temp0 = 0;
                double temp1 = 0;
                double temp2 = 0;
                double temp3 = 0;
                double temp4 = 0;
                double temp5 = 0;

                uint64_t flag_0x44 = 0x44;
                uint64_t flag_0xee = 0xee;

                int j = 0;
                for (; j < nn4; j++)
                {
                    asm volatile(
                        "ld         $0, 32(%0)      \n" // __builtin_prefetch(tmpptr + 32);
                        "ld         $0, 16(%1)      \n" // __builtin_prefetch(kptr + 16);

                        "ldc1       %3, 0(%0)       \n" // int8x8_t _v = __mmi_pldb_s(tmpptr);
                        "ldc1       %5, 0(%1)       \n" // int8x8_t _k = __mmi_pldb_s(kptr);

                        "mtc1       $0, %6          \n" // int8x8_t _zero = __mmi_pzerob_s();
                        "pcmpgtb    %4, %6, %3      \n" // int8x8_t _extv = __mmi_pcmpgtb_s(_zero, _v);
                        "pcmpgtb    %6, %6, %5      \n" // int8x8_t _extk = __mmi_pcmpgtb_s(_zero, _k);

                        "punpcklbh  %7, %3, %4      \n" // int16x4_t _v0 = (int16x4_t)__mmi_punpcklbh_s(_v, _extv);
                        "punpckhbh  %8, %3, %4      \n" // int16x4_t _v1 = (int16x4_t)__mmi_punpckhbh_s(_v, _extv);
                        "punpcklbh  %5, %5, %6      \n" // int16x4_t _k0 = (int16x4_t)__mmi_punpcklbh_s(_k, _extk);

                        "pshufh     %3, %5, %18     \n" // int16x4_t _k0202 = __mmi_pshufh_s(_k0, 0x44);
                        "pshufh     %4, %5, %19     \n" // int16x4_t _k1313 = __mmi_pshufh_s(_k0, 0xee);

                        "pmaddhw    %3, %7, %3      \n" // int32x2_t _s01x = __mmi_pmaddhw(_v0, _k0101);
                        "pmaddhw    %4, %8, %4      \n" // int32x2_t _s01y = __mmi_pmaddhw(_v1, _k2323);

                        "paddw      %2, %2, %3      \n" // _sum01 = __mmi_paddw_s(_sum01, _s01x);
                        "paddw      %2, %2, %4      \n" // _sum01 = __mmi_paddw_s(_sum01, _s01y);

                        : "=r"(tmpptr), // %0
                        "=r"(kptr),   // %1
                        "=f"(_sum01), // %2
                        "=f"(temp0),  // %3
                        "=f"(temp1),  // %4
                        "=f"(temp2),  // %5
                        "=f"(temp3),  // %6
                        "=f"(temp4),  // %7
                        "=f"(temp5)   // %8
                        : "0"(tmpptr),
                        "1"(kptr),
                        "2"(_sum01),
                        "3"(temp0),
                        "4"(temp1),
                        "5"(temp2),
                        "6"(temp3),
                        "7"(temp4),
                        "8"(temp5),
                        "f"(flag_0x44), // %18
                        "f"(flag_0xee)  // %19
                        : "memory");

                    tmpptr += 8;
                    kptr += 4;
                }

                int sum[2];
                __mmi_pstw_s(sum, _sum01);

                sum0 = sum[0];
                sum1 = sum[1];
            }
#else  // __mips_loongson_mmi
            int nn1 = inch * maxk;
#endif // __mips_msa || __mips_loongson_mmi

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
#if __mips_msa
            const signed char* kptr = kernel.channel(p / 4 + p % 4);
#else
            const signed char* kptr = kernel.channel(p / 2 + p % 2);
#endif

            int sum = 0;

#if __mips_msa
            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            if (nn4 > 0)
            {
                v4i32 _sum = __msa_fill_w(0);

                int j = 0;
                for (; j < nn4; j++)
                {
                    v16i8 _val = __msa_ld_b(tmpptr, 0);
                    v8i16 _val16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_val, 0), _val);

                    v16i8 _w = __msa_ld_b(kptr, 0);
                    v8i16 _w16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_w, 0), _w);

                    v8i16 _s0 = __msa_mulv_h(_val16, _w16);
                    v4i32 _s032 = (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0);

                    _sum = __msa_addv_w(_sum, _s032);

                    tmpptr += 4;
                    kptr += 4;
                }

                sum = _sum[0] + _sum[1] + _sum[2] + _sum[3];
            }
#elif __mips_loongson_mmi
            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            if (nn4 > 0)
            {
                int32x2_t _sum0 = __mmi_pzerow_s();

                double temp0 = 0;
                double temp1 = 0;
                double temp2 = 0;
                double temp3 = 0;

                int j = 0;
                for (; j < nn4; j++)
                {
                    asm volatile(
                        "ld         $0, 16(%0)      \n" // __builtin_prefetch(tmpptr + 16);
                        "ld         $0, 16(%1)      \n" // __builtin_prefetch(kptr + 16);

                        "ldc1       %3, 0(%0)       \n" // int8x8_t _v = __mmi_pldb_s(tmpptr);
                        "ldc1       %5, 0(%1)       \n" // int8x8_t _k = __mmi_pldb_s(kptr);

                        "mtc1       $0, %6          \n" // int8x8_t _zero = __mmi_pzerob_s();
                        "pcmpgtb    %4, %6, %3      \n" // int8x8_t _extv = __mmi_pcmpgtb_s(_zero, _v);
                        "pcmpgtb    %6, %6, %5      \n" // int8x8_t _extk = __mmi_pcmpgtb_s(_zero, _k);

                        "punpcklbh  %3, %3, %4      \n" // int16x4_t _v0 = (int16x4_t)__mmi_punpcklbh_s(_v, _extv);
                        "punpcklbh  %5, %5, %6      \n" // int16x4_t _k0 = (int16x4_t)__mmi_punpcklbh_s(_k, _extk);

                        "pmaddhw    %3, %3, %5      \n" // int32x2_t _s0x = __mmi_pmaddhw(_v0, _k0);
                        "paddw      %2, %2, %3      \n" // _sum0 = __mmi_paddw_s(_sum0, _s0x);

                        : "=r"(tmpptr), // %0
                        "=r"(kptr),   // %1
                        "=f"(_sum0),  // %2
                        "=f"(temp0),  // %3
                        "=f"(temp1),  // %4
                        "=f"(temp2),  // %5
                        "=f"(temp3)   // %6
                        : "0"(tmpptr),
                        "1"(kptr),
                        "2"(_sum0),
                        "3"(temp0),
                        "4"(temp1),
                        "5"(temp2),
                        "6"(temp3)
                        : "memory");

                    tmpptr += 4;
                    kptr += 4;
                }

                int tmp[2];
                __mmi_pstw_s(tmp, _sum0);

                sum = tmp[0] + tmp[1];
            }
#else  // __mips_loongson_mmi
            int nn1 = inch * maxk;
#endif // __mips_msa

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

static void convolution_im2col_sgemm_transform_kernel_int8_msa(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
    {
        convolution_im2col_sgemm_transform_kernel_int8_loongson_mmi(_kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
        return;
    }
#endif

    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4a-4b-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __mips_msa
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
#if __mips_loongson_mmi
        if (inch >= 4)
            kernel_tm.create(8 * maxk, inch / 4 + inch % 4, outch / 2 + outch % 2, (size_t)1u);
        else
#endif // __mips_loongson_mmi
        {
            kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2, (size_t)1u);
        }
    }
#endif // __mips_msa
    else
    {
#if __mips_msa || __mips_loongson_mmi
        if (inch >= 4)
            kernel_tm.create(4 * maxk, inch / 4 + inch % 4, outch, (size_t)1u);
        else
#endif // __mips_msa || __mips_loongson_mmi
        {
            kernel_tm.create(1 * maxk, inch, outch, (size_t)1u);
        }
    }

    int q = 0;
#if __mips_msa
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
#else // __mips_msa
    for (; q + 1 < outch; q += 2)
    {
        signed char* g00 = kernel_tm.channel(q / 2);

        int p = 0;
#if __mips_loongson_mmi
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k00 = kernel.channel(q).row<const signed char>(p);
                const signed char* k01 = kernel.channel(q).row<const signed char>(p + 1);
                const signed char* k02 = kernel.channel(q).row<const signed char>(p + 2);
                const signed char* k03 = kernel.channel(q).row<const signed char>(p + 3);
                const signed char* k10 = kernel.channel(q + 1).row<const signed char>(p);
                const signed char* k11 = kernel.channel(q + 1).row<const signed char>(p + 1);
                const signed char* k12 = kernel.channel(q + 1).row<const signed char>(p + 2);
                const signed char* k13 = kernel.channel(q + 1).row<const signed char>(p + 3);

                g00[0] = k00[k];
                g00[1] = k01[k];
                g00[2] = k10[k];
                g00[3] = k11[k];
                g00[4] = k02[k];
                g00[5] = k03[k];
                g00[6] = k12[k];
                g00[7] = k13[k];

                g00 += 8;
            }
        }
#endif // __mips_loongson_mmi
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
#endif // __mips_msa
    for (; q < outch; q++)
    {
#if __mips_msa
        signed char* g00 = kernel_tm.channel(q / 4 + q % 4);
#else
        signed char* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __mips_msa || __mips_loongson_mmi
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
#endif // __mips_msa || __mips_loongson_mmi
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

static void convolution_im2col_sgemm_int8_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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

    im2col_sgemm_int8_msa(bottom_im2col, top_blob, kernel, opt);
}
