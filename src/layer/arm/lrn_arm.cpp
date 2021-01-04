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

#include "lrn_arm.h"

#include <math.h>

#if __ARM_NEON
#include "neon_mathfun.h"

#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

int LRN_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int size = w * h;

    // squared values with local_size padding
    Mat square_blob;
    square_blob.create(w, h, channels, elemsize, opt.workspace_allocator);
    if (square_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_top_blob.channel(q);
        float* outptr = square_blob.channel(q);

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _outp = vmulq_f32(_p, _p);
            vst1q_f32(outptr, _outp);

            ptr += 4;
            outptr += 4;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *outptr = *ptr * *ptr;

            ptr++;
            outptr++;
        }
    }

    if (region_type == NormRegion_ACROSS_CHANNELS)
    {
        Mat square_sum;
        square_sum.create(w, h, channels, elemsize, opt.workspace_allocator);
        if (square_sum.empty())
            return -100;
        square_sum.fill(0.f);

        const float alpha_div_size = alpha / local_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            // square sum
            for (int p = q - local_size / 2; p <= q + local_size / 2; p++)
            {
                if (p < 0 || p >= channels)
                    continue;

                const float* sptr = square_blob.channel(p);
                float* ssptr = square_sum.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                for (; nn > 0; nn--)
                {
                    float32x4_t _sp = vld1q_f32(sptr);
                    float32x4_t _ssp = vld1q_f32(ssptr);
                    _ssp = vaddq_f32(_ssp, _sp);
                    vst1q_f32(ssptr, _ssp);

                    sptr += 4;
                    ssptr += 4;
                }
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    *ssptr += *sptr;
                    sptr++;
                    ssptr++;
                }
            }

            float* ptr = bottom_top_blob.channel(q);
            float* ssptr = square_sum.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _bias = vdupq_n_f32(bias);
            float32x4_t _ads = vdupq_n_f32(alpha_div_size);
            float32x4_t _mb = vdupq_n_f32(-beta);
            for (; nn > 0; nn--)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _ssp = vld1q_f32(ssptr);
                _ssp = vmulq_f32(_ssp, _ads);
                _ssp = vaddq_f32(_ssp, _bias);
                _ssp = pow_ps(_ssp, _mb);
                _p = vmulq_f32(_p, _ssp);
                vst1q_f32(ptr, _p);

                ssptr += 4;
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                *ptr = *ptr * pow(bias + alpha_div_size * *ssptr, -beta);

                ssptr++;
                ptr++;
            }
        }
    }
    else if (region_type == NormRegion_WITHIN_CHANNEL)
    {
        int outw = w;
        int outh = h;

        Mat square_blob_bordered = square_blob;
        int pad = local_size / 2;
        if (pad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(square_blob, square_blob_bordered, pad, local_size - pad - 1, pad, local_size - pad - 1, BORDER_CONSTANT, 0.f, opt_b);
            if (square_blob_bordered.empty())
                return -100;

            w = square_blob_bordered.w;
            h = square_blob_bordered.h;
        }

        const int maxk = local_size * local_size;

        const float alpha_div_size = alpha / maxk;

        // norm window offsets
        std::vector<int> _space_ofs(maxk);
        int* space_ofs = &_space_ofs[0];
        {
            int p1 = 0;
            int p2 = 0;
            int gap = w - local_size;
            for (int i = 0; i < local_size; i++)
            {
                for (int j = 0; j < local_size; j++)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2++;
                }
                p2 += gap;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const Mat m = square_blob_bordered.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i) + j;

                    float ss = 0.f;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        ss += val;
                    }

                    ptr[j] = ptr[j] * pow(bias + alpha_div_size * ss, -beta);
                }

                ptr += outw;
            }
        }
    }

    return 0;
}

} // namespace ncnn
