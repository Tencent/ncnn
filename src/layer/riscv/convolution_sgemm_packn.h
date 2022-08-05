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

static void im2col_sgemm_packn_rvv(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);

    // Mat bottom_im2col(size, maxk, inch, 4u * packn, packn, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 4u * packn, packn, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 4u * packn, packn, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 4u * packn, packn, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 4u * packn, packn, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        int nn_size = size >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            float* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

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
                    vfloat32m1_t _val0 = vle32_v_f32m1(img0, vl);
                    vfloat32m1_t _val1 = vle32_v_f32m1(img0 + packn, vl);
                    vfloat32m1_t _val2 = vle32_v_f32m1(img0 + packn * 2, vl);
                    vfloat32m1_t _val3 = vle32_v_f32m1(img0 + packn * 3, vl);
                    vfloat32m1_t _val4 = vle32_v_f32m1(img0 + packn * 4, vl);
                    vfloat32m1_t _val5 = vle32_v_f32m1(img0 + packn * 5, vl);
                    vfloat32m1_t _val6 = vle32_v_f32m1(img0 + packn * 6, vl);
                    vfloat32m1_t _val7 = vle32_v_f32m1(img0 + packn * 7, vl);
                    vsseg8e32_v_f32m1(tmpptr, _val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7, vl);

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

            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

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
                    vfloat32m1_t _val0 = vle32_v_f32m1(img0, vl);
                    vfloat32m1_t _val1 = vle32_v_f32m1(img0 + packn, vl);
                    vfloat32m1_t _val2 = vle32_v_f32m1(img0 + packn * 2, vl);
                    vfloat32m1_t _val3 = vle32_v_f32m1(img0 + packn * 3, vl);
                    vsseg4e32_v_f32m1(tmpptr, _val0, _val1, _val2, _val3, vl);

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

            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

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
                    vfloat32m1_t _val0 = vle32_v_f32m1(img0, vl);
                    vfloat32m1_t _val1 = vle32_v_f32m1(img0 + packn, vl);
                    vsseg2e32_v_f32m1(tmpptr, _val0, _val1, vl);

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
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    vfloat32m1_t _val = vle32_v_f32m1(img0, vl);
                    vse32_v_f32m1(tmpptr, _val, vl);

                    img0 += size * packn;
                    tmpptr += packn;
                }
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum4 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum5 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum6 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum7 = vfmv_v_f_f32m1(0.f, vl);

            if (bias)
            {
                _sum0 = vle32_v_f32m1(bias + p * packn, vl);
                _sum1 = vle32_v_f32m1(bias + p * packn, vl);
                _sum2 = vle32_v_f32m1(bias + p * packn, vl);
                _sum3 = vle32_v_f32m1(bias + p * packn, vl);
                _sum4 = vle32_v_f32m1(bias + p * packn, vl);
                _sum5 = vle32_v_f32m1(bias + p * packn, vl);
                _sum6 = vle32_v_f32m1(bias + p * packn, vl);
                _sum7 = vle32_v_f32m1(bias + p * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                float val0 = *tmpptr++;
                float val1 = *tmpptr++;
                float val2 = *tmpptr++;
                float val3 = *tmpptr++;
                float val4 = *tmpptr++;
                float val5 = *tmpptr++;
                float val6 = *tmpptr++;
                float val7 = *tmpptr++;
                vfloat32m1_t _w0 = vle32_v_f32m1(kptr0, vl);
                _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                _sum2 = vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                _sum3 = vfmacc_vf_f32m1(_sum3, val3, _w0, vl);
                _sum4 = vfmacc_vf_f32m1(_sum4, val4, _w0, vl);
                _sum5 = vfmacc_vf_f32m1(_sum5, val5, _w0, vl);
                _sum6 = vfmacc_vf_f32m1(_sum6, val6, _w0, vl);
                _sum7 = vfmacc_vf_f32m1(_sum7, val7, _w0, vl);

                kptr0 += packn;
            }

            vse32_v_f32m1(outptr0, _sum0, vl);
            vse32_v_f32m1(outptr0 + packn, _sum1, vl);
            vse32_v_f32m1(outptr0 + packn * 2, _sum2, vl);
            vse32_v_f32m1(outptr0 + packn * 3, _sum3, vl);
            vse32_v_f32m1(outptr0 + packn * 4, _sum4, vl);
            vse32_v_f32m1(outptr0 + packn * 5, _sum5, vl);
            vse32_v_f32m1(outptr0 + packn * 6, _sum6, vl);
            vse32_v_f32m1(outptr0 + packn * 7, _sum7, vl);

            outptr0 += packn * 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);

            if (bias)
            {
                _sum0 = vle32_v_f32m1(bias + p * packn, vl);
                _sum1 = vle32_v_f32m1(bias + p * packn, vl);
                _sum2 = vle32_v_f32m1(bias + p * packn, vl);
                _sum3 = vle32_v_f32m1(bias + p * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                float val0 = *tmpptr++;
                float val1 = *tmpptr++;
                float val2 = *tmpptr++;
                float val3 = *tmpptr++;
                vfloat32m1_t _w0 = vle32_v_f32m1(kptr0, vl);
                _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                _sum2 = vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                _sum3 = vfmacc_vf_f32m1(_sum3, val3, _w0, vl);

                kptr0 += packn;
            }

            vse32_v_f32m1(outptr0, _sum0, vl);
            vse32_v_f32m1(outptr0 + packn, _sum1, vl);
            vse32_v_f32m1(outptr0 + packn * 2, _sum2, vl);
            vse32_v_f32m1(outptr0 + packn * 3, _sum3, vl);

            outptr0 += packn * 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);

            if (bias)
            {
                _sum0 = vle32_v_f32m1(bias + p * packn, vl);
                _sum1 = vle32_v_f32m1(bias + p * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                float val0 = *tmpptr++;
                float val1 = *tmpptr++;
                vfloat32m1_t _w0 = vle32_v_f32m1(kptr0, vl);
                _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);

                kptr0 += packn;
            }

            vse32_v_f32m1(outptr0, _sum0, vl);
            vse32_v_f32m1(outptr0 + packn, _sum1, vl);

            outptr0 += packn * 2;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const float* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

            if (bias)
            {
                _sum = vle32_v_f32m1(bias + p * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                float val = *tmpptr++;
                vfloat32m1_t _w0 = vle32_v_f32m1(kptr0, vl);
                _sum = vfmacc_vf_f32m1(_sum, val, _w0, vl);

                kptr0 += packn;
            }

            vse32_v_f32m1(outptr0, _sum, vl);

            outptr0 += packn;
        }
    }
}

static void convolution_im2col_sgemm_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);

    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 4u * packn, packn, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * packn;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const float* sptr = img.row<const float>(dilation_h * u) + dilation_w * v * packn;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            vfloat32m1_t _val = vle32_v_f32m1(sptr, vl);
                            vse32_v_f32m1(ptr, _val, vl);

                            sptr += stride_w * packn;
                            ptr += packn;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_packn_rvv(bottom_im2col, top_blob, kernel, _bias, opt);
}
