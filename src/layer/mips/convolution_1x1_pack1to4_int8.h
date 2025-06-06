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

static void conv1x1s1_sgemm_pack1to4_int8_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    const int size = w * h;

    Mat bottom_im2col = bottom_blob;
    bottom_im2col.w = size;
    bottom_im2col.h = 1;

    im2col_sgemm_pack1to4_int8_msa(bottom_im2col, top_blob, kernel, opt);
}

static void conv1x1s2_sgemm_pack1to4_int8_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = w - 2 * outw + w;

    Mat bottom_blob_shrinked;
    bottom_blob_shrinked.create(outw, outh, channels, elemsize, elempack, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const signed char* r0 = bottom_blob.channel(p);
        signed char* outptr = bottom_blob_shrinked.channel(p);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                outptr[0] = r0[0];
                outptr[1] = r0[2];
                outptr[2] = r0[4];
                outptr[3] = r0[6];

                r0 += 8;
                outptr += 4;
            }
            for (; j + 1 < outw; j += 2)
            {
                outptr[0] = r0[0];
                outptr[1] = r0[2];

                r0 += 4;
                outptr += 2;
            }
            for (; j < outw; j++)
            {
                outptr[0] = r0[0];

                r0 += 2;
                outptr += 1;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack1to4_int8_msa(bottom_blob_shrinked, top_blob, kernel, opt);
}
