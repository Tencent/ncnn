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

#include "pooling_mips.h"

#include <float.h>

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

Pooling_mips::Pooling_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
}

int Pooling_mips::create_pipeline(const Option& /*opt*/)
{
    if (adaptive_pooling)
    {
        support_packing = false;

        support_bf16_storage = false;
        support_fp16_storage = false;
        support_int8_storage = false;
        support_tensor_storage = false;
    }
    return 0;
}

int Pooling_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (adaptive_pooling)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __mips_msa
    //     NCNN_LOGE("Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);

    if (elempack == 4)
    {
        if (global_pooling)
        {
            top_blob.create(channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int size = w * h;

            if (pooling_type == PoolMethod_MAX)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);

                    v4f32 _max = (v4f32)__msa_ld_w(ptr, 0);
                    for (int i = 0; i < size; i++)
                    {
                        v4f32 _val = (v4f32)__msa_ld_w(ptr, 0);
                        _max = __msa_fmax_w(_max, _val);
                        ptr += 4;
                    }

                    float* outptr = top_blob;
                    __msa_st_w((v4i32)_max, outptr + q * 4, 0);
                }
            }
            else if (pooling_type == PoolMethod_AVE)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);

                    v4f32 _sum = (v4f32)__msa_fill_w(0);
                    for (int i = 0; i < size; i++)
                    {
                        v4f32 _val = (v4f32)__msa_ld_w(ptr, 0);
                        _sum = __msa_fadd_w(_sum, _val);
                        ptr += 4;
                    }

                    v4f32 _avg = __msa_fmul_w(_sum, __msa_fill_w_f32(1.f / size));

                    float* outptr = top_blob;
                    __msa_st_w((v4i32)_avg, outptr + q * 4, 0);
                }
            }

            return 0;
        }

        Mat bottom_blob_bordered;
        make_padding(bottom_blob, bottom_blob_bordered, opt);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;

        int outw = (w - kernel_w) / stride_w + 1;
        int outh = (h - kernel_h) / stride_h + 1;

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int maxk = kernel_w * kernel_h;

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
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                        v4f32 _max = (v4f32)__msa_ld_w(sptr, 0);

                        for (int k = 0; k < maxk; k++)
                        {
                            v4f32 _val = (v4f32)__msa_ld_w(sptr + space_ofs[k] * 4, 0);
                            _max = __msa_fmax_w(_max, _val);
                        }

                        __msa_st_w((v4i32)_max, outptr + j * 4, 0);
                    }

                    outptr += outw * 4;
                }
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            if (avgpool_count_include_pad == 0)
            {
                int wtailpad = 0;
                int htailpad = 0;

                if (pad_mode == 0) // full padding
                {
                    wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                    htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        int sy0 = i * stride_h;

                        for (int j = 0; j < outw; j++)
                        {
                            int sx0 = j * stride_w;

                            v4f32 _sum = (v4f32)__msa_fill_w(0);
                            int area = 0;

                            for (int ki = 0; ki < kernel_h; ki++)
                            {
                                int sy = sy0 + ki;

                                if (sy < pad_top)
                                    continue;

                                if (sy >= h - pad_bottom - htailpad)
                                    break;

                                for (int kj = 0; kj < kernel_w; kj++)
                                {
                                    int sx = sx0 + kj;

                                    if (sx < pad_left)
                                        continue;

                                    if (sx >= w - pad_right - wtailpad)
                                        break;

                                    v4f32 _val = (v4f32)__msa_ld_w(m.row(sy) + sx * 4, 0);
                                    _sum = __msa_fadd_w(_sum, _val);
                                    area += 1;
                                }
                            }

                            v4f32 _avg = __msa_fmul_w(_sum, __msa_fill_w_f32(1.f / area));
                            __msa_st_w((v4i32)_avg, outptr + j * 4, 0);
                        }

                        outptr += outw * 4;
                    }
                }
            }
            else // if (avgpool_count_include_pad == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    float* outptr = top_blob.channel(q);

                    const float inv_maxk = 1.f / maxk;

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                            v4f32 _sum = (v4f32)__msa_fill_w(0);

                            for (int k = 0; k < maxk; k++)
                            {
                                v4f32 _val = (v4f32)__msa_ld_w(sptr + space_ofs[k] * 4, 0);
                                _sum = __msa_fadd_w(_sum, _val);
                            }

                            v4f32 _avg = __msa_fmul_w(_sum, __msa_fill_w_f32(inv_maxk));
                            __msa_st_w((v4i32)_avg, outptr + j * 4, 0);
                        }

                        outptr += outw * 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __mips_msa

    return Pooling::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
