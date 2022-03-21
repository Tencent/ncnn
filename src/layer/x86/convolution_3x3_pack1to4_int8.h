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

static void conv3x3s1_pack1to4_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = 9;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 1u, 1, opt.workspace_allocator);
    {
        const int gap = w - outw;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            signed char* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < 3; u++)
            {
                for (int v = 0; v < 3; v++)
                {
                    const signed char* sptr = img.row<const signed char>(u) + v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 3 < outw; j += 4)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[1];
                            ptr[2] = sptr[2];
                            ptr[3] = sptr[3];

                            sptr += 4;
                            ptr += 4;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[1];

                            sptr += 2;
                            ptr += 2;
                        }
                        for (; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += 1;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack1to4_int8_sse(bottom_im2col, top_blob, kernel, opt);
}

static void conv3x3s2_pack1to4_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = 9;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 1u, 1, opt.workspace_allocator);
    {
        const int gap = w * 2 - outw * 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            signed char* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < 3; u++)
            {
                for (int v = 0; v < 3; v++)
                {
                    const signed char* sptr = img.row<const signed char>(u) + v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 3 < outw; j += 4)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[2];
                            ptr[2] = sptr[4];
                            ptr[3] = sptr[6];

                            sptr += 8;
                            ptr += 4;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[2];

                            sptr += 4;
                            ptr += 2;
                        }
                        for (; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += 2;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack1to4_int8_sse(bottom_im2col, top_blob, kernel, opt);
}
