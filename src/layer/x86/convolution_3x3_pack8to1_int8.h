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

#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
void conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse_avx512vnni(const Mat& kernel, Mat& kernel_tm_pack8to1, int inch, int outch, const Option& opt);
void conv3x3s1_winograd43_pack8to1_int8_sse_avx512vnni(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
void conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse_avxvnni(const Mat& kernel, Mat& kernel_tm_pack8to1, int inch, int outch, const Option& opt);
void conv3x3s1_winograd43_pack8to1_int8_sse_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse_avx2(const Mat& kernel, Mat& kernel_tm_pack8to1, int inch, int outch, const Option& opt);
void conv3x3s1_winograd43_pack8to1_int8_sse_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
void conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse_xop(const Mat& kernel, Mat& kernel_tm_pack8to1, int inch, int outch, const Option& opt);
void conv3x3s1_winograd43_pack8to1_int8_sse_xop(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt);
#endif
#endif

static void conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse(const Mat& kernel, Mat& kernel_tm_pack8to1, int inch, int outch, const Option& opt)
{
#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse_avx512vnni(kernel, kernel_tm_pack8to1, inch, outch, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse_avxvnni(kernel, kernel_tm_pack8to1, inch, outch, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse_avx2(kernel, kernel_tm_pack8to1, inch, outch, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
    if (ncnn::cpu_support_x86_xop())
    {
        conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse_xop(kernel, kernel_tm_pack8to1, inch, outch, opt);
        return;
    }
#endif
#endif

    // winograd43 transform kernel
    Mat kernel_tm(6 * 6, inch, outch, (size_t)2u);

    const short ktm[6][3] = {
        {6, 0, 0},
        {-4, -4, -4},
        {-4, 4, -4},
        {1, 2, 4},
        {1, -2, 4},
        {0, 0, 6}
    };

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const signed char* kernel0 = (const signed char*)kernel + p * inch * 9 + q * 9;
            short* kernel_tm0 = kernel_tm.channel(p).row<short>(q);

            // transform kernel
            const signed char* k0 = kernel0;
            const signed char* k1 = kernel0 + 3;
            const signed char* k2 = kernel0 + 6;

            // h
            short tmp[6][3];
            for (int i = 0; i < 6; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 6; j++)
            {
                short* tmpp = &tmp[j][0];

                for (int i = 0; i < 6; i++)
                {
                    kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 36-inch-outch
    // dst = 4b-8a-inch/8a-36-outch/4b
    kernel_tm_pack8to1.create(8 * inch / 8, 36, outch / 4 + outch % 4, (size_t)2u * 4, 4);

    int p = 0;
    for (; p + 3 < outch; p += 4)
    {
        Mat g0 = kernel_tm_pack8to1.channel(p / 4);

        for (int k = 0; k < 36; k++)
        {
            short* g00 = g0.row<short>(k);

            for (int q = 0; q + 7 < inch; q += 8)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const short* k00 = kernel_tm.channel(p + i).row<const short>(q + j);
                        g00[0] = k00[k];
                        g00 += 1;
                    }
                }
            }
        }
    }
    for (; p < outch; p++)
    {
        const Mat k0 = kernel_tm.channel(p);

        Mat g0 = kernel_tm_pack8to1.channel(p / 4 + p % 4);

        for (int k = 0; k < 36; k++)
        {
            short* g00 = g0.row<short>(k);

            for (int q = 0; q + 7 < inch; q += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0.row<const short>(q + i)[k];

                    g00 += 1;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_pack8to1_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Option& opt)
{
#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        conv3x3s1_winograd43_pack8to1_int8_sse_avx512vnni(bottom_blob, top_blob, kernel_tm, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        conv3x3s1_winograd43_pack8to1_int8_sse_avxvnni(bottom_blob, top_blob, kernel_tm, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        conv3x3s1_winograd43_pack8to1_int8_sse_avx2(bottom_blob, top_blob, kernel_tm, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
    if (ncnn::cpu_support_x86_xop())
    {
        conv3x3s1_winograd43_pack8to1_int8_sse_xop(bottom_blob, top_blob, kernel_tm, opt);
        return;
    }
#endif
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    //     size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 4n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 3) / 4 * 4;
    outh = (outh + 3) / 4 * 4;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, BORDER_CONSTANT, 0.f, opt);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        const int tiles = w_tm / 6 * h_tm / 6;

        bottom_blob_tm.create(tiles, 36, inch, 2u * elempack, elempack, opt.workspace_allocator);

        // const float itm[4][4] = {
        //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
        //     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
        //     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
        //     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
        // };

        // 0 =  4 * r00 - 5 * r02 + r04
        // 1 = -4 * (r01 + r02) + r04 + r03
        // 2 =  4 * (r01 - r02) + r04 - r03
        // 3 = -2 * (r01 - r03) + r04 - r02
        // 4 =  2 * (r01 - r03) + r04 - r02
        // 5 =  4 * r01 - 5 * r03 + r05

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            short tmp[6][6][8];

            // tile
            for (int i = 0; i < h_tm / 6; i++)
            {
                for (int j = 0; j < w_tm / 6; j++)
                {
                    const signed char* r0 = img0.row<const signed char>(i * 4) + (j * 4) * 8;

                    for (int m = 0; m < 6; m++)
                    {
                        // TODO use _mm_cvtepi8_epi16 on sse4.1
                        __m128i _r00_01 = _mm_loadu_si128((const __m128i*)r0);
                        __m128i _r02_03 = _mm_loadu_si128((const __m128i*)(r0 + 16));
                        __m128i _r04_05 = _mm_loadu_si128((const __m128i*)(r0 + 32));
                        __m128i _extr0001 = _mm_cmpgt_epi8(_mm_setzero_si128(), _r00_01);
                        __m128i _extr0203 = _mm_cmpgt_epi8(_mm_setzero_si128(), _r02_03);
                        __m128i _extr0405 = _mm_cmpgt_epi8(_mm_setzero_si128(), _r04_05);
                        __m128i _r00 = _mm_unpacklo_epi8(_r00_01, _extr0001);
                        __m128i _r01 = _mm_unpackhi_epi8(_r00_01, _extr0001);
                        __m128i _r02 = _mm_unpacklo_epi8(_r02_03, _extr0203);
                        __m128i _r03 = _mm_unpackhi_epi8(_r02_03, _extr0203);
                        __m128i _r04 = _mm_unpacklo_epi8(_r04_05, _extr0405);
                        __m128i _r05 = _mm_unpackhi_epi8(_r04_05, _extr0405);

                        __m128i _v5 = _mm_set1_epi16(5);

                        __m128i _tmp0m = _mm_sub_epi16(_mm_add_epi16(_mm_slli_epi16(_r00, 2), _r04), _mm_mullo_epi16(_r02, _v5));
                        __m128i _tmp1m = _mm_sub_epi16(_mm_add_epi16(_r04, _r03), _mm_slli_epi16(_mm_add_epi16(_r01, _r02), 2));
                        __m128i _tmp2m = _mm_add_epi16(_mm_sub_epi16(_r04, _r03), _mm_slli_epi16(_mm_sub_epi16(_r01, _r02), 2));
                        __m128i _tmp3m = _mm_sub_epi16(_mm_sub_epi16(_r04, _r02), _mm_slli_epi16(_mm_sub_epi16(_r01, _r03), 1));
                        __m128i _tmp4m = _mm_add_epi16(_mm_sub_epi16(_r04, _r02), _mm_slli_epi16(_mm_sub_epi16(_r01, _r03), 1));
                        __m128i _tmp5m = _mm_sub_epi16(_mm_add_epi16(_mm_slli_epi16(_r01, 2), _r05), _mm_mullo_epi16(_r03, _v5));

                        _mm_storeu_si128((__m128i*)tmp[0][m], _tmp0m);
                        _mm_storeu_si128((__m128i*)tmp[1][m], _tmp1m);
                        _mm_storeu_si128((__m128i*)tmp[2][m], _tmp2m);
                        _mm_storeu_si128((__m128i*)tmp[3][m], _tmp3m);
                        _mm_storeu_si128((__m128i*)tmp[4][m], _tmp4m);
                        _mm_storeu_si128((__m128i*)tmp[5][m], _tmp5m);

                        r0 += w * 8;
                    }

                    short* r0_tm_0 = (short*)img0_tm + (i * w_tm / 6 + j) * 8;
                    short* r0_tm_1 = r0_tm_0 + tiles * 8;
                    short* r0_tm_2 = r0_tm_0 + tiles * 16;
                    short* r0_tm_3 = r0_tm_0 + tiles * 24;
                    short* r0_tm_4 = r0_tm_0 + tiles * 32;
                    short* r0_tm_5 = r0_tm_0 + tiles * 40;

                    for (int m = 0; m < 6; m++)
                    {
                        __m128i _tmp00 = _mm_loadu_si128((const __m128i*)tmp[m][0]);
                        __m128i _tmp01 = _mm_loadu_si128((const __m128i*)tmp[m][1]);
                        __m128i _tmp02 = _mm_loadu_si128((const __m128i*)tmp[m][2]);
                        __m128i _tmp03 = _mm_loadu_si128((const __m128i*)tmp[m][3]);
                        __m128i _tmp04 = _mm_loadu_si128((const __m128i*)tmp[m][4]);
                        __m128i _tmp05 = _mm_loadu_si128((const __m128i*)tmp[m][5]);

                        __m128i _v5 = _mm_set1_epi16(5);

                        __m128i _r0tm0 = _mm_sub_epi16(_mm_add_epi16(_mm_slli_epi16(_tmp00, 2), _tmp04), _mm_mullo_epi16(_tmp02, _v5));
                        __m128i _r0tm1 = _mm_sub_epi16(_mm_add_epi16(_tmp04, _tmp03), _mm_slli_epi16(_mm_add_epi16(_tmp01, _tmp02), 2));
                        __m128i _r0tm2 = _mm_add_epi16(_mm_sub_epi16(_tmp04, _tmp03), _mm_slli_epi16(_mm_sub_epi16(_tmp01, _tmp02), 2));
                        __m128i _r0tm3 = _mm_sub_epi16(_mm_sub_epi16(_tmp04, _tmp02), _mm_slli_epi16(_mm_sub_epi16(_tmp01, _tmp03), 1));
                        __m128i _r0tm4 = _mm_add_epi16(_mm_sub_epi16(_tmp04, _tmp02), _mm_slli_epi16(_mm_sub_epi16(_tmp01, _tmp03), 1));
                        __m128i _r0tm5 = _mm_sub_epi16(_mm_add_epi16(_mm_slli_epi16(_tmp01, 2), _tmp05), _mm_mullo_epi16(_tmp03, _v5));

                        _mm_storeu_si128((__m128i*)r0_tm_0, _r0tm0);
                        _mm_storeu_si128((__m128i*)r0_tm_1, _r0tm1);
                        _mm_storeu_si128((__m128i*)r0_tm_2, _r0tm2);
                        _mm_storeu_si128((__m128i*)r0_tm_3, _r0tm3);
                        _mm_storeu_si128((__m128i*)r0_tm_4, _r0tm4);
                        _mm_storeu_si128((__m128i*)r0_tm_5, _r0tm5);

                        r0_tm_0 += tiles * 48;
                        r0_tm_1 += tiles * 48;
                        r0_tm_2 += tiles * 48;
                        r0_tm_3 += tiles * 48;
                        r0_tm_4 += tiles * 48;
                        r0_tm_5 += tiles * 48;
                    }
                }
            }
        }
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        const int tiles = h_tm / 6 * w_tm / 6;

        // permute
        //         bottom_blob_tm.create(tiles, 36, inch, elemsize, elempack, opt.workspace_allocator);
        Mat bottom_blob_tm2;
#if __AVX2__
        if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 36, 2u * elempack, elempack, opt.workspace_allocator);
#else
        if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 36, 2u * elempack, elempack, opt.workspace_allocator);
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 36; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
#if __AVX2__
            for (; i + 3 < tiles; i += 4)
            {
                short* tmpptr = tm2.row<short>(i / 4);

                const short* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m256i _r0 = _mm256_loadu_si256((const __m256i*)r0);
                    __m256i _r1 = _mm256_loadu_si256((const __m256i*)(r0 + 16));
                    _mm256_storeu_si256((__m256i*)tmpptr, _r0);
                    _mm256_storeu_si256((__m256i*)(tmpptr + 16), _r1);
                    r0 += bottom_blob_tm.cstep * 8;
                    tmpptr += 32;
                }
            }
#endif
            for (; i + 1 < tiles; i += 2)
            {
#if __AVX2__
                short* tmpptr = tm2.row<short>(i / 4 + (i % 4) / 2);
#else
                short* tmpptr = tm2.row<short>(i / 2);
#endif

                const short* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m128i _r0 = _mm_loadu_si128((const __m128i*)r0);
                    __m128i _r1 = _mm_loadu_si128((const __m128i*)(r0 + 8));
                    _mm_storeu_si128((__m128i*)tmpptr, _r0);
                    _mm_storeu_si128((__m128i*)(tmpptr + 8), _r1);
                    r0 += bottom_blob_tm.cstep * 8;
                    tmpptr += 16;
                }
            }
            for (; i < tiles; i++)
            {
#if __AVX2__
                short* tmpptr = tm2.row<short>(i / 4 + (i % 4) / 2 + i % 2);
#else
                short* tmpptr = tm2.row<short>(i / 2 + i % 2);
#endif

                const short* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m128i _r0 = _mm_loadu_si128((const __m128i*)r0);
                    _mm_storeu_si128((__m128i*)tmpptr, _r0);
                    r0 += bottom_blob_tm.cstep * 8;
                    tmpptr += 8;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 36, outch, 4u, 1, opt.workspace_allocator);

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 4;

            int* output0_tm = top_blob_tm.channel(p);
            int* output1_tm = top_blob_tm.channel(p + 1);
            int* output2_tm = top_blob_tm.channel(p + 2);
            int* output3_tm = top_blob_tm.channel(p + 3);

            const Mat kernel0_tm = kernel_tm.channel(p / 4);

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __AVX2__
                for (; i + 3 < tiles; i += 4)
                {
                    const short* r0 = bb2.row<const short>(i / 4);
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

                    __m256i _sum00_11 = _mm256_setzero_si256();
                    __m256i _sum10_01 = _mm256_setzero_si256();
                    __m256i _sum02_13 = _mm256_setzero_si256();
                    __m256i _sum12_03 = _mm256_setzero_si256();

                    __m256i _sum04_15 = _mm256_setzero_si256();
                    __m256i _sum14_05 = _mm256_setzero_si256();
                    __m256i _sum06_17 = _mm256_setzero_si256();
                    __m256i _sum16_07 = _mm256_setzero_si256();

                    for (int j = 0; j < nn; j++)
                    {
                        // 0 1 2 3 4 5 6 7 8 9 a b c d e f
                        __m256i _val01 = _mm256_loadu_si256((const __m256i*)r0);

                        __m256i _w01 = _mm256_loadu_si256((const __m256i*)k0);
                        __m256i _w23 = _mm256_loadu_si256((const __m256i*)(k0 + 16));

                        __m256i _val10 = _mm256_permute4x64_epi64(_val01, 78);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum00_11 = _mm256_dpwssd_epi32(_sum00_11, _val01, _w01);
                        _sum10_01 = _mm256_dpwssd_epi32(_sum10_01, _val10, _w01);
                        _sum02_13 = _mm256_dpwssd_epi32(_sum02_13, _val01, _w23);
                        _sum12_03 = _mm256_dpwssd_epi32(_sum12_03, _val10, _w23);
#else
                        _sum00_11 = _mm256_add_epi32(_sum00_11, _mm256_madd_epi16(_val01, _w01));
                        _sum10_01 = _mm256_add_epi32(_sum10_01, _mm256_madd_epi16(_val10, _w01));
                        _sum02_13 = _mm256_add_epi32(_sum02_13, _mm256_madd_epi16(_val01, _w23));
                        _sum12_03 = _mm256_add_epi32(_sum12_03, _mm256_madd_epi16(_val10, _w23));
#endif

                        __m256i _val23 = _mm256_loadu_si256((const __m256i*)(r0 + 16));

                        __m256i _val32 = _mm256_permute4x64_epi64(_val23, 78);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum04_15 = _mm256_dpwssd_epi32(_sum04_15, _val23, _w01);
                        _sum14_05 = _mm256_dpwssd_epi32(_sum14_05, _val32, _w01);
                        _sum06_17 = _mm256_dpwssd_epi32(_sum06_17, _val23, _w23);
                        _sum16_07 = _mm256_dpwssd_epi32(_sum16_07, _val32, _w23);
#else
                        _sum04_15 = _mm256_add_epi32(_sum04_15, _mm256_madd_epi16(_val23, _w01));
                        _sum14_05 = _mm256_add_epi32(_sum14_05, _mm256_madd_epi16(_val32, _w01));
                        _sum06_17 = _mm256_add_epi32(_sum06_17, _mm256_madd_epi16(_val23, _w23));
                        _sum16_07 = _mm256_add_epi32(_sum16_07, _mm256_madd_epi16(_val32, _w23));
#endif

                        r0 += 32;
                        k0 += 32;
                    }

                    // transpose 4x8
                    {
                        __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm256_unpacklo_epi32(_sum00_11, _sum10_01);
                        _tmp1 = _mm256_unpacklo_epi32(_sum02_13, _sum12_03);
                        _tmp2 = _mm256_unpackhi_epi32(_sum00_11, _sum10_01);
                        _tmp3 = _mm256_unpackhi_epi32(_sum02_13, _sum12_03);
                        _sum00_11 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                        _sum10_01 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                        _sum02_13 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                        _sum12_03 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                    }
                    {
                        __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm256_unpacklo_epi32(_sum04_15, _sum14_05);
                        _tmp1 = _mm256_unpacklo_epi32(_sum06_17, _sum16_07);
                        _tmp2 = _mm256_unpackhi_epi32(_sum04_15, _sum14_05);
                        _tmp3 = _mm256_unpackhi_epi32(_sum06_17, _sum16_07);
                        _sum04_15 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                        _sum14_05 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                        _sum06_17 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                        _sum16_07 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                    }

                    _sum00_11 = _mm256_add_epi32(_sum00_11, _sum10_01);
                    _sum02_13 = _mm256_add_epi32(_sum02_13, _sum12_03);
                    _sum00_11 = _mm256_add_epi32(_sum00_11, _sum02_13);

                    _sum04_15 = _mm256_add_epi32(_sum04_15, _sum14_05);
                    _sum06_17 = _mm256_add_epi32(_sum06_17, _sum16_07);
                    _sum04_15 = _mm256_add_epi32(_sum04_15, _sum06_17);

                    __m256i _perm_mask = _mm256_set_epi32(6, 3, 4, 1, 7, 2, 5, 0);
                    _sum00_11 = _mm256_permutevar8x32_epi32(_sum00_11, _perm_mask);
                    _sum04_15 = _mm256_permutevar8x32_epi32(_sum04_15, _perm_mask);

                    int sum[16];
                    _mm256_storeu_si256((__m256i*)sum, _sum00_11);
                    _mm256_storeu_si256((__m256i*)(sum + 8), _sum04_15);

                    output0_tm[0] = sum[0];
                    output1_tm[0] = sum[1];
                    output2_tm[0] = sum[2];
                    output3_tm[0] = sum[3];
                    output0_tm[1] = sum[4];
                    output1_tm[1] = sum[5];
                    output2_tm[1] = sum[6];
                    output3_tm[1] = sum[7];
                    output0_tm[2] = sum[8];
                    output1_tm[2] = sum[9];
                    output2_tm[2] = sum[10];
                    output3_tm[2] = sum[11];
                    output0_tm[3] = sum[12];
                    output1_tm[3] = sum[13];
                    output2_tm[3] = sum[14];
                    output3_tm[3] = sum[15];
                    output0_tm += 4;
                    output1_tm += 4;
                    output2_tm += 4;
                    output3_tm += 4;
                }
#endif
                for (; i + 1 < tiles; i += 2)
                {
#if __AVX2__
                    const short* r0 = bb2.row<const short>(i / 4 + (i % 4) / 2);
#else
                    const short* r0 = bb2.row<const short>(i / 2);
#endif
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

#if __AVX2__
                    __m256i _sum00_11 = _mm256_setzero_si256();
                    __m256i _sum10_01 = _mm256_setzero_si256();
                    __m256i _sum02_13 = _mm256_setzero_si256();
                    __m256i _sum12_03 = _mm256_setzero_si256();
#else
                    __m128i _sum00 = _mm_setzero_si128();
                    __m128i _sum01 = _mm_setzero_si128();
                    __m128i _sum02 = _mm_setzero_si128();
                    __m128i _sum03 = _mm_setzero_si128();
                    __m128i _sum10 = _mm_setzero_si128();
                    __m128i _sum11 = _mm_setzero_si128();
                    __m128i _sum12 = _mm_setzero_si128();
                    __m128i _sum13 = _mm_setzero_si128();
#endif

                    for (int j = 0; j < nn; j++)
                    {
#if __AVX2__
                        // 0 1 2 3 4 5 6 7 8 9 a b c d e f
                        __m256i _val01 = _mm256_loadu_si256((const __m256i*)r0);

                        __m256i _w01 = _mm256_loadu_si256((const __m256i*)k0);
                        __m256i _w23 = _mm256_loadu_si256((const __m256i*)(k0 + 16));

                        __m256i _val10 = _mm256_permute4x64_epi64(_val01, 78);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum00_11 = _mm256_dpwssd_epi32(_sum00_11, _val01, _w01);
                        _sum10_01 = _mm256_dpwssd_epi32(_sum10_01, _val10, _w01);
                        _sum02_13 = _mm256_dpwssd_epi32(_sum02_13, _val01, _w23);
                        _sum12_03 = _mm256_dpwssd_epi32(_sum12_03, _val10, _w23);
#else
                        _sum00_11 = _mm256_add_epi32(_sum00_11, _mm256_madd_epi16(_val01, _w01));
                        _sum10_01 = _mm256_add_epi32(_sum10_01, _mm256_madd_epi16(_val10, _w01));
                        _sum02_13 = _mm256_add_epi32(_sum02_13, _mm256_madd_epi16(_val01, _w23));
                        _sum12_03 = _mm256_add_epi32(_sum12_03, _mm256_madd_epi16(_val10, _w23));
#endif
#else
                        // 0 1 2 3 4 5 6 7
                        __m128i _val0 = _mm_loadu_si128((const __m128i*)r0);
                        __m128i _val1 = _mm_loadu_si128((const __m128i*)(r0 + 8));

                        __m128i _w0 = _mm_loadu_si128((const __m128i*)k0);
                        __m128i _w1 = _mm_loadu_si128((const __m128i*)(k0 + 8));
                        __m128i _w2 = _mm_loadu_si128((const __m128i*)(k0 + 16));
                        __m128i _w3 = _mm_loadu_si128((const __m128i*)(k0 + 24));

#if __XOP__
                        _sum00 = _mm_maddd_epi16(_val0, _w0, _sum00);
                        _sum01 = _mm_maddd_epi16(_val0, _w1, _sum01);
                        _sum02 = _mm_maddd_epi16(_val0, _w2, _sum02);
                        _sum03 = _mm_maddd_epi16(_val0, _w3, _sum03);
                        _sum10 = _mm_maddd_epi16(_val1, _w0, _sum10);
                        _sum11 = _mm_maddd_epi16(_val1, _w1, _sum11);
                        _sum12 = _mm_maddd_epi16(_val1, _w2, _sum12);
                        _sum13 = _mm_maddd_epi16(_val1, _w3, _sum13);
#else
                        _sum00 = _mm_add_epi32(_mm_madd_epi16(_val0, _w0), _sum00);
                        _sum01 = _mm_add_epi32(_mm_madd_epi16(_val0, _w1), _sum01);
                        _sum02 = _mm_add_epi32(_mm_madd_epi16(_val0, _w2), _sum02);
                        _sum03 = _mm_add_epi32(_mm_madd_epi16(_val0, _w3), _sum03);
                        _sum10 = _mm_add_epi32(_mm_madd_epi16(_val1, _w0), _sum10);
                        _sum11 = _mm_add_epi32(_mm_madd_epi16(_val1, _w1), _sum11);
                        _sum12 = _mm_add_epi32(_mm_madd_epi16(_val1, _w2), _sum12);
                        _sum13 = _mm_add_epi32(_mm_madd_epi16(_val1, _w3), _sum13);
#endif
#endif

                        r0 += 16;
                        k0 += 32;
                    }

#if __AVX2__
                    // transpose 4x8
                    {
                        __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm256_unpacklo_epi32(_sum00_11, _sum10_01);
                        _tmp1 = _mm256_unpacklo_epi32(_sum02_13, _sum12_03);
                        _tmp2 = _mm256_unpackhi_epi32(_sum00_11, _sum10_01);
                        _tmp3 = _mm256_unpackhi_epi32(_sum02_13, _sum12_03);
                        _sum00_11 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                        _sum10_01 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                        _sum02_13 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                        _sum12_03 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                    }

                    _sum00_11 = _mm256_add_epi32(_sum00_11, _sum10_01);
                    _sum02_13 = _mm256_add_epi32(_sum02_13, _sum12_03);
                    _sum00_11 = _mm256_add_epi32(_sum00_11, _sum02_13);

                    __m256i _perm_mask = _mm256_set_epi32(6, 3, 4, 1, 7, 2, 5, 0);
                    _sum00_11 = _mm256_permutevar8x32_epi32(_sum00_11, _perm_mask);

                    int sum[8];
                    _mm256_storeu_si256((__m256i*)sum, _sum00_11);
#else
                    // transpose 4x4
                    {
                        __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm_unpacklo_epi32(_sum00, _sum01);
                        _tmp1 = _mm_unpacklo_epi32(_sum02, _sum03);
                        _tmp2 = _mm_unpackhi_epi32(_sum00, _sum01);
                        _tmp3 = _mm_unpackhi_epi32(_sum02, _sum03);
                        _sum00 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                        _sum01 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                        _sum02 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                        _sum03 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                    }
                    {
                        __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm_unpacklo_epi32(_sum10, _sum11);
                        _tmp1 = _mm_unpacklo_epi32(_sum12, _sum13);
                        _tmp2 = _mm_unpackhi_epi32(_sum10, _sum11);
                        _tmp3 = _mm_unpackhi_epi32(_sum12, _sum13);
                        _sum10 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                        _sum11 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                        _sum12 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                        _sum13 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                    }

                    _sum00 = _mm_add_epi32(_sum00, _sum01);
                    _sum02 = _mm_add_epi32(_sum02, _sum03);
                    _sum10 = _mm_add_epi32(_sum10, _sum11);
                    _sum12 = _mm_add_epi32(_sum12, _sum13);

                    _sum00 = _mm_add_epi32(_sum00, _sum02);
                    _sum10 = _mm_add_epi32(_sum10, _sum12);

                    int sum[8];
                    _mm_storeu_si128((__m128i*)sum, _sum00);
                    _mm_storeu_si128((__m128i*)(sum + 4), _sum10);
#endif

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
#if __AVX2__
                    const short* r0 = bb2.row<const short>(i / 4 + (i % 4) / 2 + i % 2);
#else
                    const short* r0 = bb2.row<const short>(i / 2 + i % 2);
#endif
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

#if __AVX2__
                    __m256i _sum0_1 = _mm256_setzero_si256();
                    __m256i _sum2_3 = _mm256_setzero_si256();
#else
                    __m128i _sum0 = _mm_setzero_si128();
                    __m128i _sum1 = _mm_setzero_si128();
                    __m128i _sum2 = _mm_setzero_si128();
                    __m128i _sum3 = _mm_setzero_si128();
#endif

                    for (int j = 0; j < nn; j++)
                    {
                        // 0 1 2 3 4 5 6 7
                        __m128i _val = _mm_loadu_si128((const __m128i*)r0);

#if __AVX2__
                        __m256i _w01 = _mm256_loadu_si256((const __m256i*)k0);
                        __m256i _w23 = _mm256_loadu_si256((const __m256i*)(k0 + 16));

                        __m256i _valval = _mm256_inserti128_si256(_mm256_castsi128_si256(_val), _val, 1);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0_1 = _mm256_dpwssd_epi32(_sum0_1, _valval, _w01);
                        _sum2_3 = _mm256_dpwssd_epi32(_sum2_3, _valval, _w23);
#else
                        _sum0_1 = _mm256_add_epi32(_sum0_1, _mm256_madd_epi16(_valval, _w01));
                        _sum2_3 = _mm256_add_epi32(_sum2_3, _mm256_madd_epi16(_valval, _w23));
#endif
#else
                        __m128i _w0 = _mm_loadu_si128((const __m128i*)k0);
                        __m128i _w1 = _mm_loadu_si128((const __m128i*)(k0 + 8));
                        __m128i _w2 = _mm_loadu_si128((const __m128i*)(k0 + 16));
                        __m128i _w3 = _mm_loadu_si128((const __m128i*)(k0 + 24));

#if __XOP__
                        _sum0 = _mm_maddd_epi16(_val, _w0, _sum0);
                        _sum1 = _mm_maddd_epi16(_val, _w1, _sum1);
                        _sum2 = _mm_maddd_epi16(_val, _w2, _sum2);
                        _sum3 = _mm_maddd_epi16(_val, _w3, _sum3);
#else
                        _sum0 = _mm_add_epi32(_mm_madd_epi16(_val, _w0), _sum0);
                        _sum1 = _mm_add_epi32(_mm_madd_epi16(_val, _w1), _sum1);
                        _sum2 = _mm_add_epi32(_mm_madd_epi16(_val, _w2), _sum2);
                        _sum3 = _mm_add_epi32(_mm_madd_epi16(_val, _w3), _sum3);
#endif
#endif

                        r0 += 8;
                        k0 += 32;
                    }

#if __AVX2__
                    __m128i _sum0 = _mm256_extracti128_si256(_sum0_1, 0);
                    __m128i _sum1 = _mm256_extracti128_si256(_sum0_1, 1);
                    __m128i _sum2 = _mm256_extracti128_si256(_sum2_3, 0);
                    __m128i _sum3 = _mm256_extracti128_si256(_sum2_3, 1);
#endif

                    // transpose 4x4
                    {
                        __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
                        _tmp1 = _mm_unpacklo_epi32(_sum2, _sum3);
                        _tmp2 = _mm_unpackhi_epi32(_sum0, _sum1);
                        _tmp3 = _mm_unpackhi_epi32(_sum2, _sum3);
                        _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                        _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                        _sum2 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                        _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                    }

                    _sum0 = _mm_add_epi32(_sum0, _sum1);
                    _sum2 = _mm_add_epi32(_sum2, _sum3);

                    _sum0 = _mm_add_epi32(_sum0, _sum2);

                    int sum[4];
                    _mm_storeu_si128((__m128i*)sum, _sum0);

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

        remain_outch_start += nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            int* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p / 4 + p % 4);

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __AVX2__
                for (; i + 3 < tiles; i += 4)
                {
                    const short* r0 = bb2.row<const short>(i / 4);
                    const short* k0 = kernel0_tm.row<const short>(r);

                    __m256i _sum01 = _mm256_setzero_si256();
                    __m256i _sum23 = _mm256_setzero_si256();

                    for (int q = 0; q < inch; q++)
                    {
                        __m256i _val01 = _mm256_loadu_si256((const __m256i*)r0);
                        __m256i _val23 = _mm256_loadu_si256((const __m256i*)(r0 + 16));

                        __m128i _w0 = _mm_loadu_si128((const __m128i*)k0);
                        __m256i _w01 = _mm256_inserti128_si256(_mm256_castsi128_si256(_w0), _w0, 1);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum01 = _mm256_dpwssd_epi32(_sum01, _val01, _w01);
                        _sum23 = _mm256_dpwssd_epi32(_sum23, _val23, _w01);
#else
                        _sum01 = _mm256_add_epi32(_sum01, _mm256_madd_epi16(_val01, _w01));
                        _sum23 = _mm256_add_epi32(_sum23, _mm256_madd_epi16(_val23, _w01));
#endif

                        k0 += 8;
                        r0 += 32;
                    }

                    __m128i _sum0 = _mm256_extracti128_si256(_sum01, 0);
                    __m128i _sum1 = _mm256_extracti128_si256(_sum01, 1);
                    __m128i _sum2 = _mm256_extracti128_si256(_sum23, 0);
                    __m128i _sum3 = _mm256_extracti128_si256(_sum23, 1);

                    output0_tm[0] = _mm_reduce_add_epi32(_sum0);
                    output0_tm[1] = _mm_reduce_add_epi32(_sum1);
                    output0_tm[2] = _mm_reduce_add_epi32(_sum2);
                    output0_tm[3] = _mm_reduce_add_epi32(_sum3);
                    output0_tm += 4;
                }
#endif
                for (; i + 1 < tiles; i += 2)
                {
#if __AVX2__
                    const short* r0 = bb2.row<const short>(i / 4 + (i % 4) / 2);
#else
                    const short* r0 = bb2.row<const short>(i / 2);
#endif
                    const short* k0 = kernel0_tm.row<const short>(r);

#if __AVX2__
                    __m256i _sum01 = _mm256_setzero_si256();
#else
                    __m128i _sum0 = _mm_setzero_si128();
                    __m128i _sum1 = _mm_setzero_si128();
#endif

                    for (int q = 0; q < inch; q++)
                    {
#if __AVX2__
                        __m256i _val01 = _mm256_loadu_si256((const __m256i*)r0);

                        __m128i _w0 = _mm_loadu_si128((const __m128i*)k0);
                        __m256i _w01 = _mm256_inserti128_si256(_mm256_castsi128_si256(_w0), _w0, 1);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum01 = _mm256_dpwssd_epi32(_sum01, _val01, _w01);
#else
                        _sum01 = _mm256_add_epi32(_sum01, _mm256_madd_epi16(_val01, _w01));
#endif
#else
                        __m128i _val0 = _mm_loadu_si128((const __m128i*)r0);
                        __m128i _val1 = _mm_loadu_si128((const __m128i*)(r0 + 8));

                        __m128i _w0 = _mm_loadu_si128((const __m128i*)k0);

#if __XOP__
                        _sum0 = _mm_maddd_epi16(_val0, _w0, _sum0);
                        _sum1 = _mm_maddd_epi16(_val1, _w0, _sum1);
#else
                        _sum0 = _mm_add_epi32(_mm_madd_epi16(_val0, _w0), _sum0);
                        _sum1 = _mm_add_epi32(_mm_madd_epi16(_val1, _w0), _sum1);
#endif
#endif

                        k0 += 8;
                        r0 += 16;
                    }

#if __AVX2__
                    __m128i _sum0 = _mm256_extracti128_si256(_sum01, 0);
                    __m128i _sum1 = _mm256_extracti128_si256(_sum01, 1);
#endif

                    output0_tm[0] = _mm_reduce_add_epi32(_sum0);
                    output0_tm[1] = _mm_reduce_add_epi32(_sum1);
                    output0_tm += 2;
                }
                for (; i < tiles; i++)
                {
#if __AVX2__
                    const short* r0 = bb2.row<const short>(i / 4 + (i % 4) / 2 + i % 2);
#else
                    const short* r0 = bb2.row<const short>(i / 2 + i % 2);
#endif
                    const short* k0 = kernel0_tm.row<const short>(r);

                    __m128i _sum0 = _mm_setzero_si128();

                    for (int q = 0; q < inch; q++)
                    {
                        __m128i _val0 = _mm_loadu_si128((const __m128i*)r0);

                        __m128i _w0 = _mm_loadu_si128((const __m128i*)k0);

#if __XOP__
                        _sum0 = _mm_maddd_epi16(_val0, _w0, _sum0);
#else
                        _sum0 = _mm_add_epi32(_mm_madd_epi16(_val0, _w0), _sum0);
#endif

                        k0 += 8;
                        r0 += 8;
                    }

                    output0_tm[0] = _mm_reduce_add_epi32(_sum0);
                    output0_tm++;
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 4u, 1, opt.workspace_allocator);
    }
    {
        // const float otm[4][6] = {
        //     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
        //     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
        // };

        // 0 = r00 + (r01 + r02) + (r03 + r04)
        // 1 =       (r01 - r02) + (r03 - r04) * 2
        // 2 =       (r01 + r02) + (r03 + r04) * 4
        // 3 = r05 + (r01 - r02) + (r03 - r04) * 8

        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;
        const int tiles = w_tm / 6 * h_tm / 6;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            int tmp[4][6];

            // tile
            for (int i = 0; i < outh / 4; i++)
            {
                for (int j = 0; j < outw / 4; j++)
                {
                    //                     top_blob_tm.create(tiles, 36, outch, 4u, 1, opt.workspace_allocator);

                    const int* output0_tm_0 = (const int*)out0_tm + (i * w_tm / 6 + j) * 1;
                    const int* output0_tm_1 = output0_tm_0 + tiles * 1;
                    const int* output0_tm_2 = output0_tm_0 + tiles * 2;
                    const int* output0_tm_3 = output0_tm_0 + tiles * 3;
                    const int* output0_tm_4 = output0_tm_0 + tiles * 4;
                    const int* output0_tm_5 = output0_tm_0 + tiles * 5;

                    int* output0 = out0.row<int>(i * 4) + j * 4;

                    // 0 = r00 + (r01 + r02) + (r03 + r04)
                    // 1 =       (r01 - r02) + (r03 - r04) * 2
                    // 2 =       (r01 + r02) + (r03 + r04) * 4
                    // 3 = r05 + (r01 - r02) + (r03 - r04) * 8

                    // TODO sse optimize
                    for (int m = 0; m < 5; m++)
                    {
                        int tmp02a = output0_tm_1[0] + output0_tm_2[0];
                        int tmp13a = output0_tm_1[0] - output0_tm_2[0];

                        int tmp02b = output0_tm_3[0] + output0_tm_4[0];
                        int tmp13b = output0_tm_3[0] - output0_tm_4[0];

                        tmp[0][m] = output0_tm_0[0] + tmp02a + tmp02b;
                        tmp[1][m] = tmp13a + tmp13b * 2;
                        tmp[2][m] = tmp02a + tmp02b * 4;
                        tmp[3][m] = output0_tm_5[0] * 4 + tmp13a + tmp13b * 8;

                        output0_tm_0 += tiles * 6;
                        output0_tm_1 += tiles * 6;
                        output0_tm_2 += tiles * 6;
                        output0_tm_3 += tiles * 6;
                        output0_tm_4 += tiles * 6;
                        output0_tm_5 += tiles * 6;
                    }
                    for (int m = 5; m < 6; m++)
                    {
                        int tmp02a = output0_tm_1[0] + output0_tm_2[0];
                        int tmp13a = output0_tm_1[0] - output0_tm_2[0];

                        int tmp02b = output0_tm_3[0] + output0_tm_4[0];
                        int tmp13b = output0_tm_3[0] - output0_tm_4[0];

                        tmp[0][m] = (output0_tm_0[0] + tmp02a + tmp02b) * 4;
                        tmp[1][m] = (tmp13a + tmp13b * 2) * 4;
                        tmp[2][m] = (tmp02a + tmp02b * 4) * 4;
                        tmp[3][m] = (output0_tm_5[0] * 4 + tmp13a + tmp13b * 8) * 4;

                        output0_tm_0 += tiles * 6;
                        output0_tm_1 += tiles * 6;
                        output0_tm_2 += tiles * 6;
                        output0_tm_3 += tiles * 6;
                        output0_tm_4 += tiles * 6;
                        output0_tm_5 += tiles * 6;
                    }

                    for (int m = 0; m < 4; m++)
                    {
                        const int* tmp0 = tmp[m];

                        int tmp02a = tmp0[1] + tmp0[2];
                        int tmp13a = tmp0[1] - tmp0[2];

                        int tmp02b = tmp0[3] + tmp0[4];
                        int tmp13b = tmp0[3] - tmp0[4];

                        output0[0] = (tmp0[0] + tmp02a + tmp02b) / 576;
                        output0[1] = (tmp13a + tmp13b * 2) / 576;
                        output0[2] = (tmp02a + tmp02b * 4) / 576;
                        output0[3] = (tmp0[5] + tmp13a + tmp13b * 8) / 576;

                        output0 += outw;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
