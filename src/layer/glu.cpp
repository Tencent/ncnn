// Copyright (c) 2022 Xiaomi Corp.        (author: Fangjun Kuang)
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "glu.h"

namespace ncnn {

GLU::GLU()
{
    one_blob_only = true;
    support_inplace = false;
}

int GLU::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

int GLU::forward(const Mat& bottom_blob, Mat& top_blob,
                 const Option& opt) const
{
    int dims = bottom_blob.dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1)
    {   // ignore axis
        int w = bottom_blob.w;
        int out_w = w / 2;
        top_blob.create(out_w, sizeof(float), opt.blob_allocator);

        const float* in_ptr = bottom_blob;
        float* out_ptr = top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int x = 0; x < out_w; ++x)
        {
            float sigmoid = 1.f / (1.f + expf(-in_ptr[x + out_w]));

            out_ptr[x] = in_ptr[x] * sigmoid;
        }

        return 0;
    } // if (dims == 1)

    if (dims == 2 && positive_axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int out_w = w;
        int out_h = h / 2;
        top_blob.create(out_w, out_h, sizeof(float), opt.blob_allocator);

        int offset = out_w * out_h;

#if 0
        // this one is equivalent to the else branch. It is more readable
        // but less efficient
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < out_h; ++y) {
            const float *in_ptr = bottom_blob.row(y);
            float *out_ptr = top_blob.row(y);

            for (int x = 0; x < w; ++x) {
                float sigmoid =
                    1.f / (1.f + expf(-in_ptr[x + offset]));

                out_ptr[x] = in_ptr[x] * sigmoid;
            }
        }
#else
        int size = offset;
        const float* in_ptr = bottom_blob;
        float* out_ptr = top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; ++i)
        {
            float sigmoid = 1.f / (1.f + expf(-in_ptr[i + offset]));
            out_ptr[i] = in_ptr[i] * sigmoid;
        }
#endif

        return 0;
    } // if (dims == 2 && positive_axis == 0)

    if (dims == 2 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int out_w = w / 2;
        int out_h = h;

        top_blob.create(out_w, out_h, sizeof(float), opt.blob_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; ++y)
        {
            const float* in_ptr = bottom_blob.row(y);
            float* out_ptr = top_blob.row(y);

            for (int x = 0; x < out_w; ++x)
            {
                float sigmoid = 1.f / (1.f + expf(-in_ptr[x + out_w]));
                out_ptr[x] = in_ptr[x] * sigmoid;
            }
        }

        return 0;
    } // if (dims == 2 && positive_axis == 1)

    if (dims == 3 && positive_axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int c = bottom_blob.c;

        int out_w = w;
        int out_h = h;
        int out_c = c / 2;

        top_blob.create(out_w, out_h, out_c, sizeof(float), opt.blob_allocator);

        int offset = out_c * bottom_blob.cstep;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < out_c; ++q)
        {
            const float* in_ptr = bottom_blob.channel(q);
            float* out_ptr = top_blob.channel(q);

            for (int i = 0; i < size; ++i)
            {
                float sigmoid = 1.f / (1.f + expf(-in_ptr[i + offset]));
                out_ptr[i] = in_ptr[i] * sigmoid;
            }
        }
        return 0;
    } //   if (dims == 3 && positive_axis == 0) {

    if (dims == 3 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int c = bottom_blob.c;

        int out_w = w;
        int out_h = h / 2;
        int out_c = c;

        top_blob.create(out_w, out_h, out_c, sizeof(float), opt.blob_allocator);

        int offset = out_h * out_w;
        int size = offset;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            const float* in_ptr = bottom_blob.channel(q);
            float* out_ptr = top_blob.channel(q);

            for (int i = 0; i < size; ++i)
            {
                float sigmoid = 1.f / (1.f + expf(-in_ptr[i + offset]));
                out_ptr[i] = in_ptr[i] * sigmoid;
            }
        }
        return 0;
    } // if (dims == 3 && positive_axis == 1)

    if (dims == 3 && positive_axis == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int c = bottom_blob.c;

        int out_w = w / 2;
        int out_h = h;
        int out_c = c;

        top_blob.create(out_w, out_h, out_c, sizeof(float), opt.blob_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            const float* in_ptr = bottom_blob.channel(q);
            float* out_ptr = top_blob.channel(q);
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < out_w; ++x)
                {
                    float sigmoid = 1.f / (1.f + expf(-in_ptr[x + out_w]));
                    out_ptr[x] = in_ptr[x] * sigmoid;
                }
                in_ptr += w;
                out_ptr += out_w;
            }
        }
        return 0;
    } // if (dims == 3 && positive_axis == 2)

    return -100;
}

} // namespace ncnn
