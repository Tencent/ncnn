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

static void im2col_sgemm_packnto1_fp16sa_rvv(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    // Mat bottom_im2col(size, maxk, inch, 2u * packn, packn, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const __fp16* bias = _bias;

    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 2u * packn, packn, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 2u * packn, packn, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 2u * packn, packn, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 2u * packn, packn, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        int nn_size = size >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            __fp16* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * packn;

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
                    vfloat16m1_t _val0 = vle16_v_f16m1(img0, vl);
                    vfloat16m1_t _val1 = vle16_v_f16m1(img0 + packn, vl);
                    vfloat16m1_t _val2 = vle16_v_f16m1(img0 + packn * 2, vl);
                    vfloat16m1_t _val3 = vle16_v_f16m1(img0 + packn * 3, vl);
                    vfloat16m1_t _val4 = vle16_v_f16m1(img0 + packn * 4, vl);
                    vfloat16m1_t _val5 = vle16_v_f16m1(img0 + packn * 5, vl);
                    vfloat16m1_t _val6 = vle16_v_f16m1(img0 + packn * 6, vl);
                    vfloat16m1_t _val7 = vle16_v_f16m1(img0 + packn * 7, vl);
                    vsseg8e16_v_f16m1(tmpptr, _val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7, vl);

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

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * packn;

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
                    vfloat16m1_t _val0 = vle16_v_f16m1(img0, vl);
                    vfloat16m1_t _val1 = vle16_v_f16m1(img0 + packn, vl);
                    vfloat16m1_t _val2 = vle16_v_f16m1(img0 + packn * 2, vl);
                    vfloat16m1_t _val3 = vle16_v_f16m1(img0 + packn * 3, vl);
                    vsseg4e16_v_f16m1(tmpptr, _val0, _val1, _val2, _val3, vl);

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

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * packn;

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
                    vfloat16m1_t _val0 = vle16_v_f16m1(img0, vl);
                    vfloat16m1_t _val1 = vle16_v_f16m1(img0 + packn, vl);
                    vsseg2e16_v_f16m1(tmpptr, _val0, _val1, vl);

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
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    vfloat16m1_t _val = vle16_v_f16m1(img0, vl);
                    vse16_v_f16m1(tmpptr, _val, vl);

                    img0 += size * packn;
                    tmpptr += packn;
                }
            }
        }
    }

    int nn_outch = outch / packn;
    int remain_outch_start = nn_outch * packn;

    // make clang happy with the following loop
#ifdef __clang__
    __fp16* _zero_tmp = new __fp16[packn]();
    for (int _zero_clean_idx = 0; _zero_clean_idx < packn; _zero_clean_idx++)
    {
        _zero_tmp[_zero_clean_idx] = 0.f;
    }
#endif // __clang__
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * packn;

        __fp16* outptr0 = top_blob.channel(p);

#ifdef __clang__
        const __fp16* zeros = _zero_tmp;
#else
        const __fp16 zeros[packn] = {0.f};
#endif // __clang__
        const __fp16* biasptr = bias ? bias + p : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum0 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum1 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum2 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum3 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum4 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum5 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum6 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum7 = vle16_v_f16m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                __fp16 val2 = *tmpptr++;
                __fp16 val3 = *tmpptr++;
                __fp16 val4 = *tmpptr++;
                __fp16 val5 = *tmpptr++;
                __fp16 val6 = *tmpptr++;
                __fp16 val7 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, val2, _w0, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, val3, _w0, vl);
                _sum4 = vfmacc_vf_f16m1(_sum4, val4, _w0, vl);
                _sum5 = vfmacc_vf_f16m1(_sum5, val5, _w0, vl);
                _sum6 = vfmacc_vf_f16m1(_sum6, val6, _w0, vl);
                _sum7 = vfmacc_vf_f16m1(_sum7, val7, _w0, vl);

                kptr0 += packn;
            }

#if C906
            vsse16_v_f16m1(outptr0, top_blob.cstep * sizeof(__fp16), _sum0, vl);
            vsse16_v_f16m1(outptr0 + 1, top_blob.cstep * sizeof(__fp16), _sum1, vl);
            vsse16_v_f16m1(outptr0 + 2, top_blob.cstep * sizeof(__fp16), _sum2, vl);
            vsse16_v_f16m1(outptr0 + 3, top_blob.cstep * sizeof(__fp16), _sum3, vl);
            vsse16_v_f16m1(outptr0 + 4, top_blob.cstep * sizeof(__fp16), _sum4, vl);
            vsse16_v_f16m1(outptr0 + 5, top_blob.cstep * sizeof(__fp16), _sum5, vl);
            vsse16_v_f16m1(outptr0 + 6, top_blob.cstep * sizeof(__fp16), _sum6, vl);
            vsse16_v_f16m1(outptr0 + 7, top_blob.cstep * sizeof(__fp16), _sum7, vl);
#else
            vssseg8e16_v_f16m1(outptr0, top_blob.cstep * sizeof(__fp16), _sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, vl);
#endif
            outptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum0 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum1 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum2 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum3 = vle16_v_f16m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                __fp16 val2 = *tmpptr++;
                __fp16 val3 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, val2, _w0, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, val3, _w0, vl);

                kptr0 += packn;
            }

#if C906
            vsse16_v_f16m1(outptr0, top_blob.cstep * sizeof(__fp16), _sum0, vl);
            vsse16_v_f16m1(outptr0 + 1, top_blob.cstep * sizeof(__fp16), _sum1, vl);
            vsse16_v_f16m1(outptr0 + 2, top_blob.cstep * sizeof(__fp16), _sum2, vl);
            vsse16_v_f16m1(outptr0 + 3, top_blob.cstep * sizeof(__fp16), _sum3, vl);
#else
            vssseg4e16_v_f16m1(outptr0, top_blob.cstep * sizeof(__fp16), _sum0, _sum1, _sum2, _sum3, vl);
#endif
            outptr0 += 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const __fp16* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum0 = vle16_v_f16m1(biasptr, vl);
            vfloat16m1_t _sum1 = vle16_v_f16m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);

                kptr0 += packn;
            }

#if C906
            vsse16_v_f16m1(outptr0, top_blob.cstep * sizeof(__fp16), _sum0, vl);
            vsse16_v_f16m1(outptr0 + 1, top_blob.cstep * sizeof(__fp16), _sum1, vl);
#else
            vssseg2e16_v_f16m1(outptr0, top_blob.cstep * sizeof(__fp16), _sum0, _sum1, vl);
#endif
            outptr0 += 2;
        }
        for (; i < size; i++)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const __fp16* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum = vle16_v_f16m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                __fp16 val = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum = vfmacc_vf_f16m1(_sum, val, _w0, vl);

                kptr0 += packn;
            }

            vsse16_v_f16m1(outptr0, top_blob.cstep * sizeof(__fp16), _sum, vl);

            outptr0 += 1;
        }
    }
#ifdef __clang__
    delete[] _zero_tmp;
#endif // __clang__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        __fp16* outptr0 = top_blob.channel(p);

        const __fp16 bias0 = bias ? bias[p] : 0.f;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            __fp16 sum0 = bias0;
            __fp16 sum1 = bias0;
            __fp16 sum2 = bias0;
            __fp16 sum3 = bias0;
            __fp16 sum4 = bias0;
            __fp16 sum5 = bias0;
            __fp16 sum6 = bias0;
            __fp16 sum7 = bias0;

            vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum2 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum3 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum4 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum5 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum6 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum7 = vfmv_v_f_f16m1(0.f, vl);

            for (int j = 0; j < nn; j++)
            {
                vfloat16m1_t _val0;
                vfloat16m1_t _val1;
                vfloat16m1_t _val2;
                vfloat16m1_t _val3;
                vfloat16m1_t _val4;
                vfloat16m1_t _val5;
                vfloat16m1_t _val6;
                vfloat16m1_t _val7;
                vlseg8e16_v_f16m1(&_val0, &_val1, &_val2, &_val3, &_val4, &_val5, &_val6, &_val7, tmpptr, vl);
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _val0, _w0, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _val1, _w0, vl);
                _sum2 = vfmacc_vv_f16m1(_sum2, _val2, _w0, vl);
                _sum3 = vfmacc_vv_f16m1(_sum3, _val3, _w0, vl);
                _sum4 = vfmacc_vv_f16m1(_sum4, _val4, _w0, vl);
                _sum5 = vfmacc_vv_f16m1(_sum5, _val5, _w0, vl);
                _sum6 = vfmacc_vv_f16m1(_sum6, _val6, _w0, vl);
                _sum7 = vfmacc_vv_f16m1(_sum7, _val7, _w0, vl);
                tmpptr += packn * 8;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<__fp16> ss0(packn);
            std::vector<__fp16> ss1(packn);
            std::vector<__fp16> ss2(packn);
            std::vector<__fp16> ss3(packn);
            std::vector<__fp16> ss4(packn);
            std::vector<__fp16> ss5(packn);
            std::vector<__fp16> ss6(packn);
            std::vector<__fp16> ss7(packn);
            vse16_v_f16m1((__fp16*)ss0.data(), _sum0, vl);
            vse16_v_f16m1((__fp16*)ss1.data(), _sum1, vl);
            vse16_v_f16m1((__fp16*)ss2.data(), _sum2, vl);
            vse16_v_f16m1((__fp16*)ss3.data(), _sum3, vl);
            vse16_v_f16m1((__fp16*)ss4.data(), _sum4, vl);
            vse16_v_f16m1((__fp16*)ss5.data(), _sum5, vl);
            vse16_v_f16m1((__fp16*)ss6.data(), _sum6, vl);
            vse16_v_f16m1((__fp16*)ss7.data(), _sum7, vl);
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
            sum0 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum0, vfmv_s_f_f16m1(vfloat16m1_t(), sum0, vl), vl));
            sum1 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum1, vfmv_s_f_f16m1(vfloat16m1_t(), sum1, vl), vl));
            sum2 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum2, vfmv_s_f_f16m1(vfloat16m1_t(), sum2, vl), vl));
            sum3 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum3, vfmv_s_f_f16m1(vfloat16m1_t(), sum3, vl), vl));
            sum4 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum4, vfmv_s_f_f16m1(vfloat16m1_t(), sum4, vl), vl));
            sum5 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum5, vfmv_s_f_f16m1(vfloat16m1_t(), sum5, vl), vl));
            sum6 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum6, vfmv_s_f_f16m1(vfloat16m1_t(), sum6, vl), vl));
            sum7 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum7, vfmv_s_f_f16m1(vfloat16m1_t(), sum7, vl), vl));
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
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            __fp16 sum0 = bias0;
            __fp16 sum1 = bias0;
            __fp16 sum2 = bias0;
            __fp16 sum3 = bias0;

            vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum2 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum3 = vfmv_v_f_f16m1(0.f, vl);

            for (int j = 0; j < nn; j++)
            {
                vfloat16m1_t _val0;
                vfloat16m1_t _val1;
                vfloat16m1_t _val2;
                vfloat16m1_t _val3;

                vlseg4e16_v_f16m1(&_val0, &_val1, &_val2, &_val3, tmpptr, vl);
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _val0, _w0, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _val1, _w0, vl);
                _sum2 = vfmacc_vv_f16m1(_sum2, _val2, _w0, vl);
                _sum3 = vfmacc_vv_f16m1(_sum3, _val3, _w0, vl);
                tmpptr += packn * 4;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<__fp16> ss0(packn);
            std::vector<__fp16> ss1(packn);
            std::vector<__fp16> ss2(packn);
            std::vector<__fp16> ss3(packn);
            vse16_v_f16m1((__fp16*)ss0.data(), _sum0, vl);
            vse16_v_f16m1((__fp16*)ss1.data(), _sum1, vl);
            vse16_v_f16m1((__fp16*)ss2.data(), _sum2, vl);
            vse16_v_f16m1((__fp16*)ss3.data(), _sum3, vl);
            for (int i = 0; i < packn; i++)
            {
                sum0 += ss0[i];
                sum1 += ss1[i];
                sum2 += ss2[i];
                sum3 += ss3[i];
            }
#else
            sum0 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum0, vfmv_s_f_f16m1(vfloat16m1_t(), sum0, vl), vl));
            sum1 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum1, vfmv_s_f_f16m1(vfloat16m1_t(), sum1, vl), vl));
            sum2 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum2, vfmv_s_f_f16m1(vfloat16m1_t(), sum2, vl), vl));
            sum3 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum3, vfmv_s_f_f16m1(vfloat16m1_t(), sum3, vl), vl));
#endif

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;

            outptr0 += 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const __fp16* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            __fp16 sum0 = bias0;
            __fp16 sum1 = bias0;

            vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);

            for (int j = 0; j < nn; j++)
            {
                vfloat16m1_t _val0;
                vfloat16m1_t _val1;
                vlseg2e16_v_f16m1(&_val0, &_val1, tmpptr, vl);
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _val0, _w0, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _val1, _w0, vl);
                tmpptr += packn * 2;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<__fp16> ss0(packn);
            std::vector<__fp16> ss1(packn);
            vse16_v_f16m1((__fp16*)ss0.data(), _sum0, vl);
            vse16_v_f16m1((__fp16*)ss1.data(), _sum1, vl);
            for (int i = 0; i < packn; i++)
            {
                sum0 += ss0[i];
                sum1 += ss1[i];
            }
#else
            sum0 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum0, vfmv_s_f_f16m1(vfloat16m1_t(), sum0, vl), vl));
            sum1 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum1, vfmv_s_f_f16m1(vfloat16m1_t(), sum1, vl), vl));
#endif

            outptr0[0] = sum0;
            outptr0[1] = sum1;

            outptr0 += 2;
        }
        for (; i < size; i++)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const __fp16* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            __fp16 sum0 = bias0;

            vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);

            for (int j = 0; j < nn; j++)
            {
                vfloat16m1_t _val0 = vle16_v_f16m1(tmpptr, vl);
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _val0, _w0, vl);
                tmpptr += packn;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<__fp16> ss0(packn);
            vse16_v_f16m1((__fp16*)ss0.data(), _sum0, vl);
            for (int i = 0; i < packn; i++)
            {
                sum0 += ss0[i];
            }
#else
            sum0 = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum0, vfmv_s_f_f16m1(vfloat16m1_t(), sum0, vl), vl));
#endif

            outptr0[0] = sum0;

            outptr0 += 1;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_packnto1_fp16sa_rvv(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int packn = csrr_vlenb() / 2;

    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(packn * packn * maxk, inch / packn, outch / packn + outch % packn, (size_t)2u);

    int q = 0;
    for (; q + (packn - 1) < outch; q += packn)
    {
        __fp16* g00 = kernel_tm.channel(q / packn);

        for (int p = 0; p + (packn - 1) < inch; p += packn)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < packn; i++)
                {
                    for (int j = 0; j < packn; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);

                        g00[0] = (__fp16)k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; q < outch; q++)
    {
        const Mat k0 = kernel.channel(q);

        __fp16* g00 = kernel_tm.channel(q / packn + q % packn);

        for (int p = 0; p + (packn - 1) < inch; p += packn)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < packn; j++)
                {
                    const float* k00 = k0.row(p + j);

                    g00[0] = (__fp16)k00[k];

                    g00++;
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_packnto1_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 2u * packn, packn, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * packn;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            __fp16* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v * packn;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            vfloat16m1_t _val = vle16_v_f16m1(sptr, vl);
                            vse16_v_f16m1(ptr, _val, vl);

                            sptr += stride_w * packn;
                            ptr += packn;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_packnto1_fp16sa_rvv(bottom_im2col, top_blob, kernel, _bias, opt);
}
