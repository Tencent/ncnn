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

static void conv1x1s1_sgemm_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    const int size = w * h;

    Mat bottom_im2col = bottom_blob;
    bottom_im2col.w = size;
    bottom_im2col.h = 1;

    im2col_sgemm_packn_rvv(bottom_im2col, top_blob, kernel, _bias, opt);
}

static void conv1x1s2_sgemm_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);

    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * packn;

    Mat bottom_blob_shrinked;
    bottom_blob_shrinked.create(outw, outh, channels, elemsize, elempack, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const float* r0 = bottom_blob.channel(p);
        float* outptr = bottom_blob_shrinked.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                vfloat32m1_t _val = vle32_v_f32m1(r0, vl);
                vse32_v_f32m1(outptr, _val, vl);

                r0 += packn * 2;
                outptr += packn;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_packn_rvv(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}
