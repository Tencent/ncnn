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

static void convolution_winograd_dot_packn_rvv(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);

    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 4u * packn, packn, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
    if (tiles >= 8)
        bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, batch, 4u * packn, packn, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, batch, 4u * packn, packn, opt.workspace_allocator);
    else if (tiles >= 2)
        bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, batch, 4u * packn, packn, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 4u * packn, packn, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
        for (; i + 7 < tiles; i += 8)
        {
            float* tmpptr = tm2.row<float>(i / 8);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * packn;

            for (int q = 0; q < inch; q++)
            {
#if C906
                for (int l = 0; l < packn; l++)
                {
                    tmpptr[0] = r0[l];
                    tmpptr[1] = r0[l + packn];
                    tmpptr[2] = r0[l + packn * 2];
                    tmpptr[3] = r0[l + packn * 3];
                    tmpptr[4] = r0[l + packn * 4];
                    tmpptr[5] = r0[l + packn * 5];
                    tmpptr[6] = r0[l + packn * 6];
                    tmpptr[7] = r0[l + packn * 7];
                    tmpptr += 8;
                }

                r0 += bottom_blob_tm.cstep * packn;
#else
                vfloat32m1_t _val0 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _val1 = vle32_v_f32m1(r0 + packn, vl);
                vfloat32m1_t _val2 = vle32_v_f32m1(r0 + packn * 2, vl);
                vfloat32m1_t _val3 = vle32_v_f32m1(r0 + packn * 3, vl);
                vfloat32m1_t _val4 = vle32_v_f32m1(r0 + packn * 4, vl);
                vfloat32m1_t _val5 = vle32_v_f32m1(r0 + packn * 5, vl);
                vfloat32m1_t _val6 = vle32_v_f32m1(r0 + packn * 6, vl);
                vfloat32m1_t _val7 = vle32_v_f32m1(r0 + packn * 7, vl);
                vsseg8e32_v_f32m1(tmpptr, _val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7, vl);

                r0 += bottom_blob_tm.cstep * packn;
                tmpptr += packn * 8;
#endif
            }
        }
        for (; i + 3 < tiles; i += 4)
        {
            float* tmpptr = tm2.row<float>(i / 8 + (i % 8) / 4);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * packn;

            for (int q = 0; q < inch; q++)
            {
#if C906
                for (int l = 0; l < packn; l++)
                {
                    tmpptr[0] = r0[l];
                    tmpptr[1] = r0[l + packn];
                    tmpptr[2] = r0[l + packn * 2];
                    tmpptr[3] = r0[l + packn * 3];
                    tmpptr += 4;
                }

                r0 += bottom_blob_tm.cstep * packn;
#else
                vfloat32m1_t _val0 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _val1 = vle32_v_f32m1(r0 + packn, vl);
                vfloat32m1_t _val2 = vle32_v_f32m1(r0 + packn * 2, vl);
                vfloat32m1_t _val3 = vle32_v_f32m1(r0 + packn * 3, vl);
                vsseg4e32_v_f32m1(tmpptr, _val0, _val1, _val2, _val3, vl);

                r0 += bottom_blob_tm.cstep * packn;
                tmpptr += packn * 4;
#endif
            }
        }
        for (; i + 1 < tiles; i += 2)
        {
            float* tmpptr = tm2.row<float>(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * packn;

            for (int q = 0; q < inch; q++)
            {
#if C906
                for (int l = 0; l < packn; l++)
                {
                    tmpptr[0] = r0[l];
                    tmpptr[1] = r0[l + packn];
                    tmpptr += 2;
                }

                r0 += bottom_blob_tm.cstep * packn;
#else
                vfloat32m1_t _val0 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _val1 = vle32_v_f32m1(r0 + packn, vl);
                vsseg2e32_v_f32m1(tmpptr, _val0, _val1, vl);

                r0 += bottom_blob_tm.cstep * packn;
                tmpptr += packn * 2;
#endif
            }
        }
        for (; i < tiles; i++)
        {
            float* tmpptr = tm2.row<float>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i) * packn;

            for (int q = 0; q < inch; q++)
            {
                vfloat32m1_t _val = vle32_v_f32m1(r0, vl);
                vse32_v_f32m1(tmpptr, _val, vl);

                r0 += bottom_blob_tm.cstep * packn;
                tmpptr += packn;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 4u * packn, packn, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* output0_tm = top_blob_tm.channel(p);

        const Mat kernel0_tm = kernel_tm.channel(p);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 7 < tiles; i += 8)
            {
                const float* r0 = bb2.row<const float>(i / 8);
                const float* k0 = kernel0_tm.row<const float>(r);

                int nn = inch * packn; // inch always > 0

                vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum4 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum5 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum6 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum7 = vfmv_v_f_f32m1(0.f, vl);

                for (int j = 0; j < nn; j++)
                {
                    float val0 = *r0++;
                    float val1 = *r0++;
                    float val2 = *r0++;
                    float val3 = *r0++;
                    float val4 = *r0++;
                    float val5 = *r0++;
                    float val6 = *r0++;
                    float val7 = *r0++;
                    vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                    _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                    _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                    _sum2 = vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                    _sum3 = vfmacc_vf_f32m1(_sum3, val3, _w0, vl);
                    _sum4 = vfmacc_vf_f32m1(_sum4, val4, _w0, vl);
                    _sum5 = vfmacc_vf_f32m1(_sum5, val5, _w0, vl);
                    _sum6 = vfmacc_vf_f32m1(_sum6, val6, _w0, vl);
                    _sum7 = vfmacc_vf_f32m1(_sum7, val7, _w0, vl);

                    k0 += packn;
                }

                vse32_v_f32m1(output0_tm, _sum0, vl);
                vse32_v_f32m1(output0_tm + packn, _sum1, vl);
                vse32_v_f32m1(output0_tm + packn * 2, _sum2, vl);
                vse32_v_f32m1(output0_tm + packn * 3, _sum3, vl);
                vse32_v_f32m1(output0_tm + packn * 4, _sum4, vl);
                vse32_v_f32m1(output0_tm + packn * 5, _sum5, vl);
                vse32_v_f32m1(output0_tm + packn * 6, _sum6, vl);
                vse32_v_f32m1(output0_tm + packn * 7, _sum7, vl);

                output0_tm += packn * 8;
            }
            for (; i + 3 < tiles; i += 4)
            {
                const float* r0 = bb2.row<const float>(i / 8 + (i % 8) / 4);
                const float* k0 = kernel0_tm.row<const float>(r);

                int nn = inch * packn; // inch always > 0

                vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);

                for (int j = 0; j < nn; j++)
                {
                    float val0 = *r0++;
                    float val1 = *r0++;
                    float val2 = *r0++;
                    float val3 = *r0++;
                    vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                    _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                    _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                    _sum2 = vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                    _sum3 = vfmacc_vf_f32m1(_sum3, val3, _w0, vl);

                    k0 += packn;
                }

                vse32_v_f32m1(output0_tm, _sum0, vl);
                vse32_v_f32m1(output0_tm + packn, _sum1, vl);
                vse32_v_f32m1(output0_tm + packn * 2, _sum2, vl);
                vse32_v_f32m1(output0_tm + packn * 3, _sum3, vl);

                output0_tm += packn * 4;
            }
            for (; i + 1 < tiles; i += 2)
            {
                const float* r0 = bb2.row<const float>(i / 8 + (i % 8) / 4 + (i % 4) / 2);
                const float* k0 = kernel0_tm.row<const float>(r);

                int nn = inch * packn; // inch always > 0

                vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);

                for (int j = 0; j < nn; j++)
                {
                    float val0 = *r0++;
                    float val1 = *r0++;
                    vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                    _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                    _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);

                    k0 += packn;
                }

                vse32_v_f32m1(output0_tm, _sum0, vl);
                vse32_v_f32m1(output0_tm + packn, _sum1, vl);

                output0_tm += packn * 2;
            }
            for (; i < tiles; i++)
            {
                const float* r0 = bb2.row<const float>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
                const float* k0 = kernel0_tm.row<const float>(r);

                int nn = inch * packn; // inch always > 0

                vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                for (int j = 0; j < nn; j++)
                {
                    float val = *r0++;
                    vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                    _sum = vfmacc_vf_f32m1(_sum, val, _w0, vl);

                    k0 += packn;
                }

                vse32_v_f32m1(output0_tm, _sum, vl);

                output0_tm += packn;
            }
        }
    }
}
