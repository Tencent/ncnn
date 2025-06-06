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
void convolution_winograd_dot_int8_loongson_mmi(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt);
#endif

static void convolution_winograd_dot_int8_msa(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_MMI && !__mips_msa && !__mips_loongson_mmi
    if (ncnn::cpu_support_loongson_mmi())
    {
        convolution_winograd_dot_int8_loongson_mmi(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
        return;
    }
#endif

    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 2u, 1, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
#if __mips_msa || __mips_loongson_mmi
    if (inch >= 4)
    {
        if (tiles >= 2)
            bottom_blob_tm2.create(inch / 4 + inch % 4, tiles / 2 + tiles % 2, batch, 16u, 8, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(inch / 4 + inch % 4, tiles, batch, 8u, 4, opt.workspace_allocator);
    }
    else
#endif // __mips_msa || __mips_loongson_mmi
    {
        if (tiles >= 2)
            bottom_blob_tm2.create(inch, tiles / 2 + tiles % 2, batch, 4u, 2, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(inch, tiles, batch, 2u, 1, opt.workspace_allocator);
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
        for (; i + 1 < tiles; i += 2)
        {
            short* tmpptr = tm2.row<short>(i / 2);

            const short* r0 = (const short*)bottom_blob_tm + r * tiles + i;

            int q = 0;
#if __mips_msa || __mips_loongson_mmi
            const short* r1 = (const short*)bottom_blob_tm.channel(1) + r * tiles + i;
            const short* r2 = (const short*)bottom_blob_tm.channel(2) + r * tiles + i;
            const short* r3 = (const short*)bottom_blob_tm.channel(3) + r * tiles + i;
            for (; q + 3 < inch; q += 4)
            {
#if __mips_loongson_mmi
                tmpptr[0] = r0[0];
                tmpptr[1] = r1[0];
                tmpptr[2] = r0[1];
                tmpptr[3] = r1[1];
                tmpptr[4] = r2[0];
                tmpptr[5] = r3[0];
                tmpptr[6] = r2[1];
                tmpptr[7] = r3[1];
#else  // __mips_loongson_mmi
                tmpptr[0] = r0[0];
                tmpptr[1] = r1[0];
                tmpptr[2] = r2[0];
                tmpptr[3] = r3[0];
                tmpptr[4] = r0[1];
                tmpptr[5] = r1[1];
                tmpptr[6] = r2[1];
                tmpptr[7] = r3[1];
#endif // __mips_loongson_mmi
                r0 += bottom_blob_tm.cstep * 4;
                r1 += bottom_blob_tm.cstep * 4;
                r2 += bottom_blob_tm.cstep * 4;
                r3 += bottom_blob_tm.cstep * 4;
                tmpptr += 8;
            }
#endif // __mips_msa || __mips_loongson_mmi
            for (; q < inch; q++)
            {
                tmpptr[0] = r0[0];
                tmpptr[1] = r0[1];
                r0 += bottom_blob_tm.cstep;
                tmpptr += 2;
            }
        }
        for (; i < tiles; i++)
        {
            short* tmpptr = tm2.row<short>(i / 2 + i % 2);

            const short* r0 = (const short*)bottom_blob_tm + r * tiles + i;

            int q = 0;
#if __mips_msa || __mips_loongson_mmi
            const short* r1 = (const short*)bottom_blob_tm.channel(1) + r * tiles + i;
            const short* r2 = (const short*)bottom_blob_tm.channel(2) + r * tiles + i;
            const short* r3 = (const short*)bottom_blob_tm.channel(3) + r * tiles + i;
            for (; q + 3 < inch; q += 4)
            {
                tmpptr[0] = r0[0];
                tmpptr[1] = r1[0];
                tmpptr[2] = r2[0];
                tmpptr[3] = r3[0];
                r0 += bottom_blob_tm.cstep * 4;
                r1 += bottom_blob_tm.cstep * 4;
                r2 += bottom_blob_tm.cstep * 4;
                r3 += bottom_blob_tm.cstep * 4;
                tmpptr += 4;
            }
#endif // __mips_msa || __mips_loongson_mmi
            for (; q < inch; q++)
            {
                tmpptr[0] = r0[0];
                r0 += bottom_blob_tm.cstep;
                tmpptr += 1;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 4u, 1, opt.workspace_allocator);

#if __mips_msa
    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        int* output0_tm = top_blob_tm.channel(p);
        int* output1_tm = top_blob_tm.channel(p + 1);
        int* output2_tm = top_blob_tm.channel(p + 2);
        int* output3_tm = top_blob_tm.channel(p + 3);

        const Mat kernel0_tm = kernel_tm.channel(p / 4);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 1 < tiles; i += 2)
            {
                const short* r0 = bb2.row<const short>(i / 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int nn4 = inch / 4;
                int nn1 = inch % 4;

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
                        v8i16 _val01 = __msa_ld_h(r0, 0);

                        v8i16 _val0 = (v8i16)__msa_ilvr_d((v2i64)_val01, (v2i64)_val01);
                        v8i16 _val1 = (v8i16)__msa_ilvl_d((v2i64)_val01, (v2i64)_val01);

                        v8i16 _w0 = __msa_ld_h(k0, 0);
                        v8i16 _w1 = __msa_ld_h(k0 + 8, 0);

                        v8i16 _extval0 = __msa_clti_s_h(_val0, 0);
                        v8i16 _extval1 = __msa_clti_s_h(_val1, 0);
                        v8i16 _extw0 = __msa_clti_s_h(_w0, 0);
                        v8i16 _extw1 = __msa_clti_s_h(_w1, 0);

                        v4i32 _val0l = (v4i32)__msa_ilvr_h(_extval0, _val0);
                        v4i32 _val0h = (v4i32)__msa_ilvl_h(_extval0, _val0);
                        v4i32 _val1l = (v4i32)__msa_ilvr_h(_extval1, _val1);
                        v4i32 _val1h = (v4i32)__msa_ilvl_h(_extval1, _val1);

                        v4i32 _w0l = (v4i32)__msa_ilvr_h(_extw0, _w0);
                        v4i32 _w0h = (v4i32)__msa_ilvl_h(_extw0, _w0);
                        v4i32 _w1l = (v4i32)__msa_ilvr_h(_extw1, _w1);
                        v4i32 _w1h = (v4i32)__msa_ilvl_h(_extw1, _w1);

                        _sum00 = __msa_maddv_w(_sum00, _val0l, _w0l);
                        _sum01 = __msa_maddv_w(_sum01, _val0h, _w0h);
                        _sum02 = __msa_maddv_w(_sum02, _val0l, _w1l);
                        _sum03 = __msa_maddv_w(_sum03, _val0h, _w1h);
                        _sum10 = __msa_maddv_w(_sum10, _val1l, _w0l);
                        _sum11 = __msa_maddv_w(_sum11, _val1h, _w0h);
                        _sum12 = __msa_maddv_w(_sum12, _val1l, _w1l);
                        _sum13 = __msa_maddv_w(_sum13, _val1h, _w1h);

                        r0 += 8;
                        k0 += 16;
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

                for (int j = 0; j < nn1; j++)
                {
                    v8i16 _val0 = __msa_fill_h(r0[0]);
                    v8i16 _val1 = __msa_fill_h(r0[1]);
                    v8i16 _val = (v8i16)__msa_ilvr_d((v2i64)_val1, (v2i64)_val0);

                    v8i16 _w16 = __msa_ld_h(k0, 0);

                    _w16 = (v8i16)__msa_ilvr_d((v2i64)_w16, (v2i64)_w16);

                    v8i16 _extval = __msa_clti_s_h(_val, 0);
                    v8i16 _extw16 = __msa_clti_s_h(_w16, 0);

                    v4i32 _vall = (v4i32)__msa_ilvr_h(_extval, _val);
                    v4i32 _valh = (v4i32)__msa_ilvl_h(_extval, _val);
                    v4i32 _w0l = (v4i32)__msa_ilvr_h(_extw16, _w16);
                    v4i32 _w0h = (v4i32)__msa_ilvl_h(_extw16, _w16);

                    _sum00 = __msa_maddv_w(_sum00, _vall, _w0l);
                    _sum10 = __msa_maddv_w(_sum10, _valh, _w0h);

                    r0 += 2;
                    k0 += 4;
                }

                int sum[8];
                __msa_st_w(_sum00, sum, 0);
                __msa_st_w(_sum10, sum + 4, 0);

                output0_tm[0] = sum[0];
                output1_tm[0] = sum[1];
                output2_tm[0] = sum[2];
                output3_tm[0] = sum[3];
                output0_tm[1] = sum[4];
                output1_tm[1] = sum[5];
                output2_tm[1] = sum[6];
                output3_tm[1] = sum[7];
                output0_tm += 2;
                output1_tm += 2;
                output2_tm += 2;
                output3_tm += 2;
            }
            for (; i < tiles; i++)
            {
                const short* r0 = bb2.row<const short>(i / 2 + i % 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int nn4 = inch / 4;
                int nn1 = inch % 4;

                v4i32 _sum0 = __msa_fill_w(0);

                if (nn4 > 0)
                {
                    v4i32 _sum1 = __msa_fill_w(0);
                    v4i32 _sum2 = __msa_fill_w(0);
                    v4i32 _sum3 = __msa_fill_w(0);

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        v8i16 _val16 = __msa_ld_h(r0, 0);

                        _val16 = (v8i16)__msa_ilvr_d((v2i64)_val16, (v2i64)_val16);

                        v8i16 _w0 = __msa_ld_h(k0, 0);
                        v8i16 _w1 = __msa_ld_h(k0 + 8, 0);

                        v8i16 _extval16 = __msa_clti_s_h(_val16, 0);
                        v8i16 _extw0 = __msa_clti_s_h(_w0, 0);
                        v8i16 _extw1 = __msa_clti_s_h(_w1, 0);

                        v4i32 _val0l = (v4i32)__msa_ilvr_h(_extval16, _val16);
                        v4i32 _val0h = (v4i32)__msa_ilvl_h(_extval16, _val16);

                        v4i32 _w0l = (v4i32)__msa_ilvr_h(_extw0, _w0);
                        v4i32 _w0h = (v4i32)__msa_ilvl_h(_extw0, _w0);
                        v4i32 _w1l = (v4i32)__msa_ilvr_h(_extw1, _w1);
                        v4i32 _w1h = (v4i32)__msa_ilvl_h(_extw1, _w1);

                        _sum0 = __msa_maddv_w(_sum0, _val0l, _w0l);
                        _sum1 = __msa_maddv_w(_sum1, _val0h, _w0h);
                        _sum2 = __msa_maddv_w(_sum2, _val0l, _w1l);
                        _sum3 = __msa_maddv_w(_sum3, _val0h, _w1h);

                        r0 += 4;
                        k0 += 16;
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

                for (int j = 0; j < nn1; j++)
                {
                    v4i32 _val = __msa_fill_w(r0[0]);
                    v8i16 _w16 = __msa_ld_h(k0, 0);

                    v8i16 _extw16 = __msa_clti_s_h(_w16, 0);
                    v4i32 _w0l = (v4i32)__msa_ilvr_h(_extw16, _w16);

                    _sum0 = __msa_maddv_w(_sum0, _val, _w0l);

                    r0 += 1;
                    k0 += 4;
                }

                int sum[4];
                __msa_st_w(_sum0, sum, 0);

                output0_tm[0] = sum[0];
                output1_tm[0] = sum[1];
                output2_tm[0] = sum[2];
                output3_tm[0] = sum[3];
                output0_tm += 1;
                output1_tm += 1;
                output2_tm += 1;
                output3_tm += 1;
            }
        }
    }
#else // __mips_msa
    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        int* output0_tm = top_blob_tm.channel(p);
        int* output1_tm = top_blob_tm.channel(p + 1);

        const Mat kernel0_tm = kernel_tm.channel(p / 2);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 1 < tiles; i += 2)
            {
                const short* r0 = bb2.row<const short>(i / 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;

#if __mips_loongson_mmi
                int nn4 = inch / 4;
                int nn1 = inch % 4;

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

                    uint64_t flag_0x44 = 0x44;
                    uint64_t flag_0xee = 0xee;

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        asm volatile(
                            "ld         $0, 64(%0)      \n" // __builtin_prefetch(r0 + 32);
                            "ld         $0, 64(%1)      \n" // __builtin_prefetch(k0 + 32);

                            "ldc1       %4, 0(%0)       \n" // int16x4_t _v0 = __mmi_pldh_s(r0);
                            "ldc1       %5, 8(%0)       \n" // int16x4_t _v1 = __mmi_pldh_s(r0 + 4);
                            "ldc1       %6, 0(%1)       \n" // int16x4_t _k0 = __mmi_pldh_s(k0);
                            "ldc1       %7, 8(%1)       \n" // int16x4_t _k1 = __mmi_pldh_s(k0 + 4);

                            "pshufh     %8, %6, %20     \n" // int16x4_t _k0202 = __mmi_pshufh_s(_k0, 0x44);
                            "pshufh     %9, %6, %21     \n" // int16x4_t _k1313 = __mmi_pshufh_s(_k0, 0xee);
                            "pshufh     %6, %7, %20     \n" // int16x4_t _k4646 = __mmi_pshufh_s(_k1, 0x44);
                            "pshufh     %7, %7, %21     \n" // int16x4_t _k5757 = __mmi_pshufh_s(_k1, 0xee);

                            "pmaddhw    %8, %4, %8      \n" // int32x2_t _s0x = __mmi_pmaddhw(_v0, _k0202);
                            "pmaddhw    %9, %4, %9      \n" // int32x2_t _s1x = __mmi_pmaddhw(_v0, _k1313);
                            "pmaddhw    %6, %5, %6      \n" // int32x2_t _s0y = __mmi_pmaddhw(_v1, _k4646);
                            "pmaddhw    %7, %5, %7      \n" // int32x2_t _s1y = __mmi_pmaddhw(_v1, _k5757);

                            "paddw      %2, %2, %8      \n" // _sum0 = __mmi_paddw_s(_sum0, _s0x);
                            "paddw      %3, %3, %9      \n" // _sum1 = __mmi_paddw_s(_sum1, _s1x);
                            "paddw      %2, %2, %6      \n" // _sum0 = __mmi_paddw_s(_sum0, _s0y);
                            "paddw      %3, %3, %7      \n" // _sum1 = __mmi_paddw_s(_sum1, _s1y);

                            : "=r"(r0),    // %0
                            "=r"(k0),    // %1
                            "=f"(_sum0), // %2
                            "=f"(_sum1), // %3
                            "=f"(temp0), // %4
                            "=f"(temp1), // %5
                            "=f"(temp2), // %6
                            "=f"(temp3), // %7
                            "=f"(temp4), // %8
                            "=f"(temp5)  // %9
                            : "0"(r0),
                            "1"(k0),
                            "2"(_sum0),
                            "3"(_sum1),
                            "4"(temp0),
                            "5"(temp1),
                            "6"(temp2),
                            "7"(temp3),
                            "8"(temp4),
                            "9"(temp5),
                            "f"(flag_0x44), // %20
                            "f"(flag_0xee)  // %21
                            : "memory");

                        r0 += 8;
                        k0 += 8;
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
                int nn1 = inch;
#endif // __mips_loongson_mmi

                for (int j = 0; j < nn1; j++)
                {
                    signed short val0 = r0[0];
                    signed short val1 = r0[1];
                    signed short w0 = k0[0];
                    signed short w1 = k0[1];

                    sum00 += val0 * w0;
                    sum01 += val1 * w0;
                    sum10 += val0 * w1;
                    sum11 += val1 * w1;

                    r0 += 2;
                    k0 += 2;
                }

                output0_tm[0] = sum00;
                output0_tm[1] = sum01;
                output1_tm[0] = sum10;
                output1_tm[1] = sum11;
                output0_tm += 2;
                output1_tm += 2;
            }
            for (; i < tiles; i++)
            {
                const short* r0 = bb2.row<const short>(i / 2 + i % 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum0 = 0;
                int sum1 = 0;

#if __mips_loongson_mmi
                int nn4 = inch / 4;
                int nn1 = inch % 4;

                if (nn4 > 0)
                {
                    int32x2_t _sum01 = __mmi_pzerow_s();

                    double temp0 = 0;
                    double temp1 = 0;
                    double temp2 = 0;
                    double temp3 = 0;
                    double temp4 = 0;

                    uint64_t flag_0x44 = 0x44;
                    uint64_t flag_0xee = 0xee;

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        asm volatile(
                            "ld         $0, 32(%0)      \n" // __builtin_prefetch(r0 + 16);
                            "ld         $0, 64(%1)      \n" // __builtin_prefetch(k0 + 32);

                            "ldc1       %4, 0(%0)       \n" // int16x4_t _v0 = __mmi_pldh_s(r0);
                            "ldc1       %5, 0(%1)       \n" // int16x4_t _k0 = __mmi_pldh_s(k0);
                            "ldc1       %6, 8(%1)       \n" // int16x4_t _k1 = __mmi_pldh_s(k0 + 4);

                            "pshufh     %7, %4, %16     \n" // int16x4_t _v0101 = __mmi_pshufh_s(_v0, 0x44);
                            "pshufh     %4, %4, %17     \n" // int16x4_t _v2323 = __mmi_pshufh_s(_v0, 0xee);

                            "pmaddhw    %5, %7, %5      \n" // int32x2_t _s01x = __mmi_pmaddhw(_v0101, _k0);
                            "pmaddhw    %6, %4, %6      \n" // int32x2_t _s01y = __mmi_pmaddhw(_v2323, _k1);

                            "paddw      %2, %2, %5      \n" // _sum01 = __mmi_paddw_s(_sum01, _s01x);
                            "paddw      %2, %2, %6      \n" // _sum01 = __mmi_paddw_s(_sum01, _s01y);

                            : "=r"(r0),     // %0
                            "=r"(k0),     // %1
                            "=f"(_sum01), // %2
                            "=f"(temp0),  // %3
                            "=f"(temp1),  // %4
                            "=f"(temp2),  // %5
                            "=f"(temp3),  // %6
                            "=f"(temp4)   // %7
                            : "0"(r0),
                            "1"(k0),
                            "2"(_sum01),
                            "3"(temp0),
                            "4"(temp1),
                            "5"(temp2),
                            "6"(temp3),
                            "7"(temp4),
                            "f"(flag_0x44), // %16
                            "f"(flag_0xee)  // %17
                            : "memory");

                        r0 += 4;
                        k0 += 8;
                    }

                    int sum[2];
                    __mmi_pstw_s(sum, _sum01);

                    sum0 = sum[0];
                    sum1 = sum[1];
                }
#else  // __mips_loongson_mmi
                int nn1 = inch;
#endif // __mips_loongson_mmi

                for (int j = 0; j < nn1; j++)
                {
                    signed short val0 = r0[0];
                    signed short w0 = k0[0];
                    signed short w1 = k0[1];

                    sum0 += val0 * w0;
                    sum1 += val0 * w1;

                    r0 += 1;
                    k0 += 2;
                }

                output0_tm[0] = sum0;
                output1_tm[0] = sum1;
                output0_tm += 1;
                output1_tm += 1;
            }
        }
    }
#endif // __mips_msa

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* output0_tm = top_blob_tm.channel(p);

#if __mips_msa
        const Mat kernel0_tm = kernel_tm.channel(p / 4 + p % 4);
#else
        const Mat kernel0_tm = kernel_tm.channel(p / 2 + p % 2);
#endif

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 1 < tiles; i += 2)
            {
                const short* r0 = bb2.row<const short>(i / 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum0 = 0;
                int sum1 = 0;

#if __mips_msa
                int nn4 = inch / 4;
                int nn1 = inch % 4;

                if (nn4 > 0)
                {
                    v4i32 _sum0 = __msa_fill_w(0);
                    v4i32 _sum1 = __msa_fill_w(0);

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        v8i16 _val16 = __msa_ld_h(r0, 0);

                        v8i16 _w16 = __msa_ld_h(k0, 0);

                        _w16 = (v8i16)__msa_ilvr_d((v2i64)_w16, (v2i64)_w16);

                        v8i16 _extval16 = __msa_clti_s_h(_val16, 0);
                        v8i16 _extw16 = __msa_clti_s_h(_w16, 0);

                        v4i32 _val0l = (v4i32)__msa_ilvr_h(_extval16, _val16);
                        v4i32 _val0h = (v4i32)__msa_ilvl_h(_extval16, _val16);

                        v4i32 _w0l = (v4i32)__msa_ilvr_h(_extw16, _w16);
                        v4i32 _w0h = (v4i32)__msa_ilvl_h(_extw16, _w16);

                        _sum0 = __msa_maddv_w(_sum0, _val0l, _w0l);
                        _sum1 = __msa_maddv_w(_sum1, _val0h, _w0h);

                        r0 += 8;
                        k0 += 4;
                    }

                    sum0 = _sum0[0] + _sum0[1] + _sum0[2] + _sum0[3];
                    sum1 = _sum1[0] + _sum1[1] + _sum1[2] + _sum1[3];
                }
#elif __mips_loongson_mmi
                int nn4 = inch / 4;
                int nn1 = inch % 4;

                if (nn4 > 0)
                {
                    int32x2_t _sum01 = __mmi_pzerow_s();

                    double temp0 = 0;
                    double temp1 = 0;
                    double temp2 = 0;
                    double temp3 = 0;
                    double temp4 = 0;

                    uint64_t flag_0x44 = 0x44;
                    uint64_t flag_0xee = 0xee;

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        asm volatile(
                            "ld         $0, 64(%0)      \n" // __builtin_prefetch(r0 + 32);
                            "ld         $0, 32(%1)      \n" // __builtin_prefetch(k0 + 16);

                            "ldc1       %4, 0(%0)       \n" // int16x4_t _v0 = __mmi_pldh_s(r0);
                            "ldc1       %5, 8(%0)       \n" // int16x4_t _v1 = __mmi_pldh_s(r0 + 4);
                            "ldc1       %6, 0(%1)       \n" // int16x4_t _k0 = __mmi_pldh_s(k0);

                            "pshufh     %7, %6, %16     \n" // int16x4_t _k0202 = __mmi_pshufh_s(_k0, 0x44);
                            "pshufh     %6, %6, %17     \n" // int16x4_t _k1313 = __mmi_pshufh_s(_k0, 0xee);

                            "pmaddhw    %4, %4, %7      \n" // int32x2_t _s01x = __mmi_pmaddhw(_v0, _k0101);
                            "pmaddhw    %5, %5, %6      \n" // int32x2_t _s01y = __mmi_pmaddhw(_v1, _k2323);

                            "paddw      %2, %2, %4      \n" // _sum01 = __mmi_paddw_s(_sum01, _s01x);
                            "paddw      %2, %2, %5      \n" // _sum01 = __mmi_paddw_s(_sum01, _s01y);

                            : "=r"(r0),     // %0
                            "=r"(k0),     // %1
                            "=f"(_sum01), // %2
                            "=f"(temp0),  // %3
                            "=f"(temp1),  // %4
                            "=f"(temp2),  // %5
                            "=f"(temp3),  // %6
                            "=f"(temp4)   // %7
                            : "0"(r0),
                            "1"(k0),
                            "2"(_sum01),
                            "3"(temp0),
                            "4"(temp1),
                            "5"(temp2),
                            "6"(temp3),
                            "7"(temp4),
                            "f"(flag_0x44), // %16
                            "f"(flag_0xee)  // %17
                            : "memory");

                        r0 += 8;
                        k0 += 4;
                    }

                    int sum[2];
                    __mmi_pstw_s(sum, _sum01);

                    sum0 = sum[0];
                    sum1 = sum[1];
                }
#else  // __mips_loongson_mmi
                int nn1 = inch;
#endif // __mips_msa || __mips_loongson_mmi

                for (int q = 0; q < nn1; q++)
                {
                    signed short val0 = r0[0];
                    signed short val1 = r0[1];
                    signed short w = k0[0];

                    sum0 += val0 * w;
                    sum1 += val1 * w;

                    k0 += 1;
                    r0 += 2;
                }

                output0_tm[0] = sum0;
                output0_tm[1] = sum1;
                output0_tm += 2;
            }
            for (; i < tiles; i++)
            {
                const short* r0 = bb2.row<const short>(i / 2 + i % 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum = 0;

#if __mips_msa
                int nn4 = inch / 4;
                int nn1 = inch % 4;

                if (nn4 > 0)
                {
                    v4i32 _sum = __msa_fill_w(0);

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        v8i16 _val16 = __msa_ld_h(r0, 0);
                        v8i16 _w16 = __msa_ld_h(k0, 0);

                        v8i16 _extval16 = __msa_clti_s_h(_val16, 0);
                        v8i16 _extw16 = __msa_clti_s_h(_w16, 0);

                        v4i32 _val0l = (v4i32)__msa_ilvr_h(_extval16, _val16);
                        v4i32 _w0l = (v4i32)__msa_ilvr_h(_extw16, _w16);

                        _sum = __msa_maddv_w(_sum, _val0l, _w0l);

                        r0 += 4;
                        k0 += 4;
                    }

                    sum = _sum[0] + _sum[1] + _sum[2] + _sum[3];
                }
#elif __mips_loongson_mmi
                int nn4 = inch / 4;
                int nn1 = inch % 4;

                if (nn4 > 0)
                {
                    int32x2_t _sum0 = __mmi_pzerow_s();

                    double temp0 = 0;
                    double temp1 = 0;

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        asm volatile(
                            "ld         $0, 32(%0)      \n" // __builtin_prefetch(r0 + 16);
                            "ld         $0, 32(%1)      \n" // __builtin_prefetch(k0 + 16);

                            "ldc1       %3, 0(%0)       \n" // int16x4_t _v0 = __mmi_pldh_s(r0);
                            "ldc1       %4, 0(%1)       \n" // int16x4_t _k0 = __mmi_pldh_s(k0);

                            "pmaddhw    %3, %3, %4      \n" // int32x2_t _s0x = __mmi_pmaddhw(_v0, _k0);
                            "paddw      %2, %2, %3      \n" // _sum0 = __mmi_paddw_s(_sum0, _s0x);

                            : "=r"(r0),    // %0
                            "=r"(k0),    // %1
                            "=f"(_sum0), // %2
                            "=f"(temp0), // %3
                            "=f"(temp1)  // %4
                            : "0"(r0),
                            "1"(k0),
                            "2"(_sum0),
                            "3"(temp0),
                            "4"(temp1)
                            : "memory");

                        r0 += 4;
                        k0 += 4;
                    }

                    int tmp[2];
                    __mmi_pstw_s(tmp, _sum0);

                    sum = tmp[0] + tmp[1];
                }
#else  // __mips_loongson_mmi
                int nn1 = inch;
#endif // __mips_msa || __mips_loongson_mmi

                for (int q = 0; q < nn1; q++)
                {
                    signed short val = r0[0];
                    signed short w = k0[0];

                    sum += val * w;

                    k0 += 1;
                    r0 += 1;
                }

                output0_tm[0] = sum;
                output0_tm++;
            }
        }
    }
}
