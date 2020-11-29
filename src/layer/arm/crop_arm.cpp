// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "crop_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

Crop_arm::Crop_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

#if __ARM_NEON
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void crop_pack8_neon(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const float* ptr = src.row(top) + left * 8;
    float* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            vst1q_f32(outptr, _p0);
            vst1q_f32(outptr + 4, _p1);
            ptr += 8;
            outptr += 8;
        }

        ptr += (left + right) * 8;
    }
}

static void crop_pack8_fp16_neon(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const __fp16* ptr = src.row<__fp16>(top) + left * 8;
    __fp16* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            vst1q_f16(outptr, _p);
            ptr += 8;
            outptr += 8;
        }

        ptr += (left + right) * 8;
    }
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

static void crop_pack4_neon(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const float* ptr = src.row(top) + left * 4;
    float* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            vst1q_f32(outptr, _p);
            ptr += 4;
            outptr += 4;
        }

        ptr += (left + right) * 4;
    }
}

static void crop_pack4_bf16_neon(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const unsigned short* ptr = src.row<unsigned short>(top) + left * 4;
    unsigned short* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            uint16x4_t _p = vld1_u16(ptr);
            vst1_u16(outptr, _p);
            ptr += 4;
            outptr += 4;
        }

        ptr += (left + right) * 4;
    }
}
#endif // __ARM_NEON

int Crop_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (elempack == 8)
    {
        int _woffset, _hoffset, _coffset;
        int _outw, _outh, _outc;
        resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);

        if (dims == 1)
        {
            int out_elempack = _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_woffset % 8 == 0 && out_elempack == 8)
            {
                if (elemsize == 16u)
                    crop_pack8_fp16_neon(bottom_blob, top_blob, 0, _woffset / elempack);
                else
                    crop_pack8_neon(bottom_blob, top_blob, 0, _woffset / elempack);

                return 0;
            }
        }

        if (dims == 2)
        {
            int out_elempack = _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_hoffset % 8 == 0 && out_elempack == 8)
            {
                if (elemsize == 16u)
                    crop_pack8_fp16_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);
                else
                    crop_pack8_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);

                return 0;
            }
        }

        if (dims == 3)
        {
            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_coffset % 8 == 0 && out_elempack == 8)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone();
                    if (top_blob.empty())
                        return -100;
                }

                if (_outw == w && _outh == h && _outc / out_elempack == channels)
                {
                    top_blob = bottom_blob;
                    return 0;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    if (elemsize == 16u)
                        crop_pack8_fp16_neon(m, borderm, _hoffset, _woffset);
                    else
                        crop_pack8_neon(m, borderm, _hoffset, _woffset);
                }

                return 0;
            }
        }
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

    if (elempack == 4)
    {
        int _woffset, _hoffset, _coffset;
        int _outw, _outh, _outc;
        resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);

        if (dims == 1)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            int out_elempack = opt.use_fp16_arithmetic && _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
#else
            int out_elempack = _outw % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_woffset % 4 == 0 && out_elempack == 4)
            {
                if (elemsize == 8u)
                    crop_pack4_bf16_neon(bottom_blob, top_blob, 0, _woffset / elempack);
                else
                    crop_pack4_neon(bottom_blob, top_blob, 0, _woffset / elempack);

                return 0;
            }
        }

        if (dims == 2)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            int out_elempack = opt.use_fp16_arithmetic && _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
#else
            int out_elempack = _outh % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_hoffset % 4 == 0 && out_elempack == 4)
            {
                if (elemsize == 8u)
                    crop_pack4_bf16_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);
                else
                    crop_pack4_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);

                return 0;
            }
        }

        if (dims == 3)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            int out_elempack = opt.use_fp16_arithmetic && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
#else
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_coffset % 4 == 0 && out_elempack == 4)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone();
                    if (top_blob.empty())
                        return -100;
                }

                if (_outw == w && _outh == h && _outc / out_elempack == channels)
                {
                    top_blob = bottom_blob;
                    return 0;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    if (elemsize == 8u)
                        crop_pack4_bf16_neon(m, borderm, _hoffset, _woffset);
                    else
                        crop_pack4_neon(m, borderm, _hoffset, _woffset);
                }

                return 0;
            }
        }
    }
#endif // __ARM_NEON

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    return Crop::forward(bottom_blob_unpacked, top_blob, opt);
}

int Crop_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int ref_elempack = reference_blob.elempack;

    Mat& top_blob = top_blobs[0];

#if __ARM_NEON
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (elempack == 8)
    {
        int _woffset, _hoffset, _coffset;
        int _outw, _outh, _outc;
        if (woffset == -233)
        {
            resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob, _woffset, _hoffset, _coffset, _outw, _outh, _outc);
        }
        else
        {
            resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);
        }

        if (dims == 1)
        {
            int out_elempack = _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_woffset % 8 == 0 && out_elempack == 8)
            {
                if (elemsize == 16u)
                    crop_pack8_fp16_neon(bottom_blob, top_blob, 0, _woffset / elempack);
                else
                    crop_pack8_neon(bottom_blob, top_blob, 0, _woffset / elempack);

                return 0;
            }
        }

        if (dims == 2)
        {
            int out_elempack = _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_hoffset % 8 == 0 && out_elempack == 8)
            {
                if (elemsize == 16u)
                    crop_pack8_fp16_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);
                else
                    crop_pack8_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);

                return 0;
            }
        }

        if (dims == 3)
        {
            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_coffset % 8 == 0 && out_elempack == 8)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone();
                    if (top_blob.empty())
                        return -100;
                }

                if (_outw == w && _outh == h && _outc / out_elempack == channels)
                {
                    top_blob = bottom_blob;
                    return 0;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    if (elemsize == 16u)
                        crop_pack8_fp16_neon(m, borderm, _hoffset, _woffset);
                    else
                        crop_pack8_neon(m, borderm, _hoffset, _woffset);
                }

                return 0;
            }
        }
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

    if (elempack == 4)
    {
        int _woffset, _hoffset, _coffset;
        int _outw, _outh, _outc;
        if (woffset == -233)
        {
            resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob, _woffset, _hoffset, _coffset, _outw, _outh, _outc);
        }
        else
        {
            resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);
        }

        if (dims == 1)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            int out_elempack = opt.use_fp16_arithmetic && _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
#else
            int out_elempack = _outw % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_woffset % 4 == 0 && out_elempack == 4)
            {
                if (elemsize == 8u)
                    crop_pack4_bf16_neon(bottom_blob, top_blob, 0, _woffset / elempack);
                else
                    crop_pack4_neon(bottom_blob, top_blob, 0, _woffset / elempack);

                return 0;
            }
        }

        if (dims == 2)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            int out_elempack = opt.use_fp16_arithmetic && _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
#else
            int out_elempack = _outh % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_hoffset % 4 == 0 && out_elempack == 4)
            {
                if (elemsize == 8u)
                    crop_pack4_bf16_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);
                else
                    crop_pack4_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);

                return 0;
            }
        }

        if (dims == 3)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            int out_elempack = opt.use_fp16_arithmetic && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
#else
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_coffset % 4 == 0 && out_elempack == 4)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone();
                    if (top_blob.empty())
                        return -100;
                }

                if (_outw == w && _outh == h && _outc / out_elempack == channels)
                {
                    top_blob = bottom_blob;
                    return 0;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    if (elemsize == 8u)
                        crop_pack4_bf16_neon(m, borderm, _hoffset, _woffset);
                    else
                        crop_pack4_neon(m, borderm, _hoffset, _woffset);
                }

                return 0;
            }
        }
    }
#endif // __ARM_NEON

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    Mat reference_blob_unpacked = reference_blob;
    if (ref_elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(reference_blob, reference_blob_unpacked, 1, opt_pack1);
    }

    std::vector<Mat> bottom_blobs_unpacked(2);
    bottom_blobs_unpacked[0] = bottom_blob_unpacked;
    bottom_blobs_unpacked[1] = reference_blob_unpacked;

    return Crop::forward(bottom_blobs_unpacked, top_blobs, opt);
}

} // namespace ncnn
