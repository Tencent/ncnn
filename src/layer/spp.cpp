// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "spp.h"
#include <math.h>
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(SPP)

SPP::SPP()
{
    one_blob_only = true;
    support_inplace = false;
}

int SPP::load_param(const ParamDict& pd)
{
    pooling_type = pd.get(0, 0);
    pyramid_height = pd.get(1, 1);

    return 0;
}

int SPP::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    size_t elemsize = bottom_blob.elemsize;

    // 1 + 4 + 16 + 64 + ... + (2*pyramid_height)^2
    int pyramid_num_bins = ((1 << (pyramid_height * 2)) - 1) / 3;
    top_blob.create(pyramid_num_bins, 1, 2, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    float* pyramid_ptr = top_blob;

    // all spatial pyramids
    for (int p = 0; p < pyramid_height; p++)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int num_bins = 1 << p;

        int kernel_h = ceil(h / (float)num_bins);
        int stride_h = kernel_h;
        int remainder_h = stride_h * num_bins - h;
        int pad_h = (remainder_h + 1) / 2;

        int kernel_w = ceil(w / (float)num_bins);
        int stride_w = kernel_w;
        int remainder_w = stride_w * num_bins - w;
        int pad_w = (remainder_w + 1) / 2;

        // max value in NxN window
        // avg value in NxN window

        int outw = num_bins;
        int outh = num_bins;

        Mat bottom_blob_bordered = bottom_blob;
        if (pad_h > 0 || pad_w > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
            if (bottom_blob_bordered.empty())
                return -100;

            w = bottom_blob_bordered.w;
            h = bottom_blob_bordered.h;
        }

        const int maxk = kernel_h * kernel_w;

        // kernel offsets
        std::vector<int> _space_ofs(maxk);
        int* space_ofs = &_space_ofs[0];
        {
            int p1 = 0;
            int p2 = 0;
            int gap = w - kernel_w;
            for (int i = 0; i < kernel_h; i++)
            {
                for (int j = 0; j < kernel_w; j++)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2++;
                }
                p2 += gap;
            }
        }

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const Mat m(w, h, bottom_blob_bordered.channel(q));
                float* outptr = pyramid_ptr + outh * outw * q;

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i*stride_h) + j*stride_w;

                        float max = sptr[0];

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = sptr[ space_ofs[k] ];
                            max = std::max(max, val);
                        }

                        outptr[j] = max;
                    }

                    outptr += outw;
                }
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const Mat m(w, h, bottom_blob_bordered.channel(q));
                float* outptr = pyramid_ptr + outh * outw * q;

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i*stride_h) + j*stride_w;

                        float sum = 0;

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = sptr[ space_ofs[k] ];
                            sum += val;
                        }

                        outptr[j] = sum / maxk;
                    }

                    outptr += outw;
                }
            }
        }

        pyramid_ptr += channels * outh * outw;
    }

    return 0;
}

} // namespace ncnn
