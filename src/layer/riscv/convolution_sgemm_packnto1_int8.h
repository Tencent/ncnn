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

static void im2col_sgemm_packnto1_int8_rvv(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    const int packn = csrr_vlenb();
    const size_t vl = vsetvl_e8m1(packn);

    // Mat bottom_im2col(size, maxk, inch, 1u * packn, packn, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 1u * packn, packn, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 1u * packn, packn, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 1u * packn, packn, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 1u * packn, packn, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        int nn_size = size >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            int8_t* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const int8_t* img0 = (const int8_t*)bottom_im2col.channel(q) + i * packn;

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
                    vint8m1_t _val0 = vle8_v_i8m1(img0, vl);
                    vint8m1_t _val1 = vle8_v_i8m1(img0 + packn, vl);
                    vint8m1_t _val2 = vle8_v_i8m1(img0 + packn * 2, vl);
                    vint8m1_t _val3 = vle8_v_i8m1(img0 + packn * 3, vl);
                    vint8m1_t _val4 = vle8_v_i8m1(img0 + packn * 4, vl);
                    vint8m1_t _val5 = vle8_v_i8m1(img0 + packn * 5, vl);
                    vint8m1_t _val6 = vle8_v_i8m1(img0 + packn * 6, vl);
                    vint8m1_t _val7 = vle8_v_i8m1(img0 + packn * 7, vl);
                    vsseg8e8_v_i8m1(tmpptr, _val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7, vl);

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

            int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const int8_t* img0 = (const int8_t*)bottom_im2col.channel(q) + i * packn;

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
                    vint8m1_t _val0 = vle8_v_i8m1(img0, vl);
                    vint8m1_t _val1 = vle8_v_i8m1(img0 + packn, vl);
                    vint8m1_t _val2 = vle8_v_i8m1(img0 + packn * 2, vl);
                    vint8m1_t _val3 = vle8_v_i8m1(img0 + packn * 3, vl);
                    vsseg4e8_v_i8m1(tmpptr, _val0, _val1, _val2, _val3, vl);

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

            int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                const int8_t* img0 = (const int8_t*)bottom_im2col.channel(q) + i * packn;

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
                    vint8m1_t _val0 = vle8_v_i8m1(img0, vl);
                    vint8m1_t _val1 = vle8_v_i8m1(img0 + packn, vl);
                    vsseg2e8_v_i8m1(tmpptr, _val0, _val1, vl);

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
            int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                const int8_t* img0 = (const int8_t*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    vint8m1_t _val = vle8_v_i8m1(img0, vl);
                    vse8_v_i8m1(tmpptr, _val, vl);

                    img0 += size * packn;
                    tmpptr += packn;
                }
            }
        }
    }

    // TODO
    int nn_outch = outch / packn;
    int remain_outch_start = nn_outch * packn;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * packn;

        int8_t* outptr0 = top_blob.channel(p);

#ifdef __clang__
        const int8_t* zeros = _zero_tmp;
#else
        const int8_t zeros[packn] = {0};
#endif // __clang__
        const int8_t* biasptr = bias ? bias + p : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const int8_t* tmpptr = tmp.channel(i / 8);
            const int8_t* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vint8m1_t _sum0 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum1 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum2 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum3 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum4 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum5 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum6 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum7 = vle8_v_i8m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                int8_t val0 = *tmpptr++;
                int8_t val1 = *tmpptr++;
                int8_t val2 = *tmpptr++;
                int8_t val3 = *tmpptr++;
                int8_t val4 = *tmpptr++;
                int8_t val5 = *tmpptr++;
                int8_t val6 = *tmpptr++;
                int8_t val7 = *tmpptr++;
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vmacc_vx_i8m1(_sum0, val0, _w0, vl);
                _sum1 = vmacc_vx_i8m1(_sum1, val1, _w0, vl);
                _sum2 = vmacc_vx_i8m1(_sum2, val2, _w0, vl);
                _sum3 = vmacc_vx_i8m1(_sum3, val3, _w0, vl);
                _sum4 = vmacc_vx_i8m1(_sum4, val4, _w0, vl);
                _sum5 = vmacc_vx_i8m1(_sum5, val5, _w0, vl);
                _sum6 = vmacc_vx_i8m1(_sum6, val6, _w0, vl);
                _sum7 = vmacc_vx_i8m1(_sum7, val7, _w0, vl);

                kptr0 += packn;
            }

#if C906
            vsse8_v_i8m1(outptr0, top_blob.cstep * sizeof(int8_t), _sum0, vl);
            vsse8_v_i8m1(outptr0 + 1, top_blob.cstep * sizeof(int8_t), _sum1, vl);
            vsse8_v_i8m1(outptr0 + 2, top_blob.cstep * sizeof(int8_t), _sum2, vl);
            vsse8_v_i8m1(outptr0 + 3, top_blob.cstep * sizeof(int8_t), _sum3, vl);
            vsse8_v_i8m1(outptr0 + 4, top_blob.cstep * sizeof(int8_t), _sum4, vl);
            vsse8_v_i8m1(outptr0 + 5, top_blob.cstep * sizeof(int8_t), _sum5, vl);
            vsse8_v_i8m1(outptr0 + 6, top_blob.cstep * sizeof(int8_t), _sum6, vl);
            vsse8_v_i8m1(outptr0 + 7, top_blob.cstep * sizeof(int8_t), _sum7, vl);
#else
            vssseg8e8_v_i8m1(outptr0, top_blob.cstep * sizeof(int8_t), _sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, vl);
#endif
            outptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const int8_t* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vint8m1_t _sum0 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum1 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum2 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum3 = vle8_v_i8m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                int8_t val0 = *tmpptr++;
                int8_t val1 = *tmpptr++;
                int8_t val2 = *tmpptr++;
                int8_t val3 = *tmpptr++;
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vmacc_vx_i8m1(_sum0, val0, _w0, vl);
                _sum1 = vmacc_vx_i8m1(_sum1, val1, _w0, vl);
                _sum2 = vmacc_vx_i8m1(_sum2, val2, _w0, vl);
                _sum3 = vmacc_vx_i8m1(_sum3, val3, _w0, vl);

                kptr0 += packn;
            }

#if C906
            vsse8_v_i8m1(outptr0, top_blob.cstep * sizeof(int8_t), _sum0, vl);
            vsse8_v_i8m1(outptr0 + 1, top_blob.cstep * sizeof(int8_t), _sum1, vl);
            vsse8_v_i8m1(outptr0 + 2, top_blob.cstep * sizeof(int8_t), _sum2, vl);
            vsse8_v_i8m1(outptr0 + 3, top_blob.cstep * sizeof(int8_t), _sum3, vl);
#else
            vssseg4e8_v_i8m1(outptr0, top_blob.cstep * sizeof(int8_t), _sum0, _sum1, _sum2, _sum3, vl);
#endif
            outptr0 += 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const int8_t* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vint8m1_t _sum0 = vle8_v_i8m1(biasptr, vl);
            vint8m1_t _sum1 = vle8_v_i8m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                int8_t val0 = *tmpptr++;
                int8_t val1 = *tmpptr++;
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vmacc_vx_i8m1(_sum0, val0, _w0, vl);
                _sum1 = vmacc_vx_i8m1(_sum1, val1, _w0, vl);

                kptr0 += packn;
            }

#if C906
            vsse8_v_i8m1(outptr0, top_blob.cstep * sizeof(int8_t), _sum0, vl);
            vsse8_v_i8m1(outptr0 + 1, top_blob.cstep * sizeof(int8_t), _sum1, vl);
#else
            vssseg2e8_v_i8m1(outptr0, top_blob.cstep * sizeof(int8_t), _sum0, _sum1, vl);
#endif
            outptr0 += 2;
        }
        for (; i < size; i++)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const int8_t* kptr0 = kernel.channel(p / packn);

            int nn = inch * maxk * packn; // inch always > 0

            vint8m1_t _sum = vle8_v_i8m1(biasptr, vl);

            for (int j = 0; j < nn; j++)
            {
                int8_t val = *tmpptr++;
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum = vmacc_vx_i8m1(_sum, val, _w0, vl);

                kptr0 += packn;
            }

            vsse8_v_i8m1(outptr0, top_blob.cstep * sizeof(int8_t), _sum, vl);

            outptr0 += 1;
        }
    }
#ifdef __clang__
    delete[] _zero_tmp;
#endif // __clang__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int8_t* outptr0 = top_blob.channel(p);

        const int8_t bias0 = bias ? bias[p] : 0;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const int8_t* tmpptr = tmp.channel(i / 8);
            const int8_t* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            int8_t sum0 = bias0;
            int8_t sum1 = bias0;
            int8_t sum2 = bias0;
            int8_t sum3 = bias0;
            int8_t sum4 = bias0;
            int8_t sum5 = bias0;
            int8_t sum6 = bias0;
            int8_t sum7 = bias0;

            vint8m1_t _sum0 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum1 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum2 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum3 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum4 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum5 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum6 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum7 = vmv_v_x_i8m1(0, vl);

            for (int j = 0; j < nn; j++)
            {
                vint8m1_t _val0;
                vint8m1_t _val1;
                vint8m1_t _val2;
                vint8m1_t _val3;
                vint8m1_t _val4;
                vint8m1_t _val5;
                vint8m1_t _val6;
                vint8m1_t _val7;
                vlseg8e8_v_i8m1(&_val0, &_val1, &_val2, &_val3, &_val4, &_val5, &_val6, &_val7, tmpptr, vl);
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vmacc_vv_i8m1(_sum0, _val0, _w0, vl);
                _sum1 = vmacc_vv_i8m1(_sum1, _val1, _w0, vl);
                _sum2 = vmacc_vv_i8m1(_sum2, _val2, _w0, vl);
                _sum3 = vmacc_vv_i8m1(_sum3, _val3, _w0, vl);
                _sum4 = vmacc_vv_i8m1(_sum4, _val4, _w0, vl);
                _sum5 = vmacc_vv_i8m1(_sum5, _val5, _w0, vl);
                _sum6 = vmacc_vv_i8m1(_sum6, _val6, _w0, vl);
                _sum7 = vmacc_vv_i8m1(_sum7, _val7, _w0, vl);
                tmpptr += packn * 8;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<int8_t> ss0(packn);
            std::vector<int8_t> ss1(packn);
            std::vector<int8_t> ss2(packn);
            std::vector<int8_t> ss3(packn);
            std::vector<int8_t> ss4(packn);
            std::vector<int8_t> ss5(packn);
            std::vector<int8_t> ss6(packn);
            std::vector<int8_t> ss7(packn);
            vse8_v_i8m1((int8_t*)ss0.data(), _sum0, vl);
            vse8_v_i8m1((int8_t*)ss1.data(), _sum1, vl);
            vse8_v_i8m1((int8_t*)ss2.data(), _sum2, vl);
            vse8_v_i8m1((int8_t*)ss3.data(), _sum3, vl);
            vse8_v_i8m1((int8_t*)ss4.data(), _sum4, vl);
            vse8_v_i8m1((int8_t*)ss5.data(), _sum5, vl);
            vse8_v_i8m1((int8_t*)ss6.data(), _sum6, vl);
            vse8_v_i8m1((int8_t*)ss7.data(), _sum7, vl);
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
            sum0 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum0, vfmv_s_f_f16m1(vint8m1_t(), sum0, vl), vl));
            sum1 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum1, vfmv_s_f_f16m1(vint8m1_t(), sum1, vl), vl));
            sum2 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum2, vfmv_s_f_f16m1(vint8m1_t(), sum2, vl), vl));
            sum3 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum3, vfmv_s_f_f16m1(vint8m1_t(), sum3, vl), vl));
            sum4 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum4, vfmv_s_f_f16m1(vint8m1_t(), sum4, vl), vl));
            sum5 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum5, vfmv_s_f_f16m1(vint8m1_t(), sum5, vl), vl));
            sum6 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum6, vfmv_s_f_f16m1(vint8m1_t(), sum6, vl), vl));
            sum7 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum7, vfmv_s_f_f16m1(vint8m1_t(), sum7, vl), vl));
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
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const int8_t* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            int8_t sum0 = bias0;
            int8_t sum1 = bias0;
            int8_t sum2 = bias0;
            int8_t sum3 = bias0;

            vint8m1_t _sum0 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum1 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum2 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum3 = vmv_v_x_i8m1(0, vl);

            for (int j = 0; j < nn; j++)
            {
                vint8m1_t _val0;
                vint8m1_t _val1;
                vint8m1_t _val2;
                vint8m1_t _val3;

                vlseg4e8_v_i8m1(&_val0, &_val1, &_val2, &_val3, tmpptr, vl);
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vmacc_vv_i8m1(_sum0, _val0, _w0, vl);
                _sum1 = vmacc_vv_i8m1(_sum1, _val1, _w0, vl);
                _sum2 = vmacc_vv_i8m1(_sum2, _val2, _w0, vl);
                _sum3 = vmacc_vv_i8m1(_sum3, _val3, _w0, vl);
                tmpptr += packn * 4;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<int8_t> ss0(packn);
            std::vector<int8_t> ss1(packn);
            std::vector<int8_t> ss2(packn);
            std::vector<int8_t> ss3(packn);
            vse8_v_i8m1((int8_t*)ss0.data(), _sum0, vl);
            vse8_v_i8m1((int8_t*)ss1.data(), _sum1, vl);
            vse8_v_i8m1((int8_t*)ss2.data(), _sum2, vl);
            vse8_v_i8m1((int8_t*)ss3.data(), _sum3, vl);
            for (int i = 0; i < packn; i++)
            {
                sum0 += ss0[i];
                sum1 += ss1[i];
                sum2 += ss2[i];
                sum3 += ss3[i];
            }
#else
            sum0 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum0, vfmv_s_f_f16m1(vint8m1_t(), sum0, vl), vl));
            sum1 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum1, vfmv_s_f_f16m1(vint8m1_t(), sum1, vl), vl));
            sum2 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum2, vfmv_s_f_f16m1(vint8m1_t(), sum2, vl), vl));
            sum3 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum3, vfmv_s_f_f16m1(vint8m1_t(), sum3, vl), vl));
#endif

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;

            outptr0 += 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const int8_t* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            int8_t sum0 = bias0;
            int8_t sum1 = bias0;

            vint8m1_t _sum0 = vmv_v_x_i8m1(0, vl);
            vint8m1_t _sum1 = vmv_v_x_i8m1(0, vl);

            for (int j = 0; j < nn; j++)
            {
                vint8m1_t _val0;
                vint8m1_t _val1;
                vlseg2e8_v_i8m1(&_val0, &_val1, tmpptr, vl);
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vmacc_vv_i8m1(_sum0, _val0, _w0, vl);
                _sum1 = vmacc_vv_i8m1(_sum1, _val1, _w0, vl);
                tmpptr += packn * 2;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<int8_t> ss0(packn);
            std::vector<int8_t> ss1(packn);
            vse8_v_i8m1((int8_t*)ss0.data(), _sum0, vl);
            vse8_v_i8m1((int8_t*)ss1.data(), _sum1, vl);
            for (int i = 0; i < packn; i++)
            {
                sum0 += ss0[i];
                sum1 += ss1[i];
            }
#else
            sum0 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum0, vfmv_s_f_f16m1(vint8m1_t(), sum0, vl), vl));
            sum1 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum1, vfmv_s_f_f16m1(vint8m1_t(), sum1, vl), vl));
#endif

            outptr0[0] = sum0;
            outptr0[1] = sum1;

            outptr0 += 2;
        }
        for (; i < size; i++)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const int8_t* kptr0 = kernel.channel(p / packn + p % packn);

            int nn = inch * maxk; // inch always > 0

            int8_t sum0 = bias0;

            vint8m1_t _sum0 = vmv_v_x_i8m1(0, vl);

            for (int j = 0; j < nn; j++)
            {
                vint8m1_t _val0 = vle8_v_i8m1(tmpptr, vl);
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vmacc_vv_i8m1(_sum0, _val0, _w0, vl);
                tmpptr += packn;
                kptr0 += packn;
            }

#if C906
            // TODO
            std::vector<int8_t> ss0(packn);
            vse8_v_i8m1((int8_t*)ss0.data(), _sum0, vl);
            for (int i = 0; i < packn; i++)
            {
                sum0 += ss0[i];
            }
#else
            sum0 = vmv_x_s_i8m1_i8(vfredusum_vs_f16m1_f16m1(vint8m1_t(), _sum0, vfmv_s_f_f16m1(vint8m1_t(), sum0, vl), vl));
#endif

            outptr0[0] = sum0;

            outptr0 += 1;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_packnto1_int8_rvv(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int packn = csrr_vlenb();

    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(packn * packn * maxk, inch / packn, outch / packn + outch % packn, (size_t)1u);

    int q = 0;
    for (; q + (packn - 1) < outch; q += packn)
    {
        int8_t* g00 = kernel_tm.channel(q / packn);

        for (int p = 0; p + (packn - 1) < inch; p += packn)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < packn; i++)
                {
                    for (int j = 0; j < packn; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);

                        g00[0] = (int8_t)k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; q < outch; q++)
    {
        const Mat k0 = kernel.channel(q);

        int8_t* g00 = kernel_tm.channel(q / packn + q % packn);

        for (int p = 0; p + (packn - 1) < inch; p += packn)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < packn; j++)
                {
                    const float* k00 = k0.row(p + j);

                    g00[0] = (int8_t)k00[k];

                    g00++;
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_packnto1_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int packn = csrr_vlenb();
    const size_t vl = vsetvl_e8m1(packn);

    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 1u * packn, packn, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * packn;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            int8_t* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const int8_t* sptr = img.row<const int8_t>(dilation_h * u) + dilation_w * v * packn;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            vint8m1_t _val = vle8_v_i8m1(sptr, vl);
                            vse8_v_i8m1(ptr, _val, vl);

                            sptr += stride_w * packn;
                            ptr += packn;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_packnto1_int8_rvv(bottom_im2col, top_blob, kernel, _bias, opt);
}
