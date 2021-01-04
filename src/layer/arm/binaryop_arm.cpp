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

#include "binaryop_arm.h"

#include <math.h>

#if __ARM_NEON
#include "neon_mathfun.h"

#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

BinaryOp_arm::BinaryOp_arm()
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
// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

template<typename Op>
static int binary_op_pack4(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    size_t elemsize = a.elemsize;
    int elempack = a.elempack;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;
    int size1 = w1 * h1;
    size_t elemsize1 = b.elemsize;
    int elempack1 = b.elempack;

    if (a.dims == 3)
    {
        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* b0 = b.channel(q);
                    float* outptr = c.channel(q);
                    float32x4_t _b0 = vld1q_f32(b0);
                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _outp = op(_p, _b0);
                        vst1q_f32(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1 && elempack1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b;
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _p1 = vld1q_dup_f32(ptr1);
                        float32x4_t _outp = op(_p, _p1);
                        vst1q_f32(outptr, _outp);
                        ptr += 4;
                        ptr1 += 1;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* a0 = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);
                    float32x4_t _a0 = vld1q_f32(a0);
                    for (int i = 0; i < size1; i++)
                    {
                        float32x4_t _p1 = vld1q_f32(ptr1);
                        float32x4_t _outp = op(_a0, _p1);
                        vst1q_f32(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1 && elempack == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a;
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        float32x4_t _p = vld1q_dup_f32(ptr);
                        float32x4_t _p1 = vld1q_f32(ptr1);
                        float32x4_t _outp = op(_p, _p1);
                        vst1q_f32(outptr, _outp);
                        ptr += 1;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            // type 19
            c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _outp = op(_p, _p1);
                    vst1q_f32(outptr, _outp);
                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }

        c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    float32x4_t _b0 = vld1q_f32(ptr1);
                    for (int x = 0; x < w; x++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _outp = op(_p, _b0);
                        vst1q_f32(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }

                    ptr1 += 4;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 16
                float32x4_t _b0 = vdupq_n_f32(b[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _outp = op(_p, _b0);
                        vst1q_f32(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float32x4_t _b0 = vld1q_f32((const float*)b + q * 4);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _outp = op(_p, _b0);
                    vst1q_f32(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.row(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    float32x4_t _a0 = vld1q_f32(ptr);
                    for (int x = 0; x < w1; x++)
                    {
                        float32x4_t _p1 = vld1q_f32(ptr1);
                        float32x4_t _outp = op(_a0, _p1);
                        vst1q_f32(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }

                    ptr += 4;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;
            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr1);
                float32x4_t _outp = op(_p, _p1);
                vst1q_f32(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                float32x4_t _b0 = vdupq_n_f32(b[0]);
                const float* ptr = a;
                float* outptr = c;
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _outp = op(_p, _b0);
                    vst1q_f32(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }

            // type 12
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            for (int y = 0; y < h; y++)
            {
                float32x4_t _b0 = vld1q_f32(ptr1);
                for (int x = 0; x < w; x++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _outp = op(_p, _b0);
                    vst1q_f32(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                ptr1 += 4;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1 && elempack == 1)
        {
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float32x4_t _a0 = vdupq_n_f32(a[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        float32x4_t _p1 = vld1q_f32(ptr1);
                        float32x4_t _outp = op(_a0, _p1);
                        vst1q_f32(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                // type 3
                c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float32x4_t _a0 = vdupq_n_f32(a[0]);
                const float* ptr1 = b;
                float* outptr = c;
                for (int i = 0; i < size1; i++)
                {
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _outp = op(_a0, _p1);
                    vst1q_f32(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float32x4_t _a0 = vdupq_n_f32(a[0]);
                const float* ptr1 = b;
                float* outptr = c;
                for (int i = 0; i < w1; i++)
                {
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _outp = op(_a0, _p1);
                    vst1q_f32(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                float32x4_t _a0 = vld1q_f32((const float*)a + q * 4);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _outp = op(_a0, _p1);
                    vst1q_f32(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                float32x4_t _a0 = vld1q_f32(ptr);
                for (int x = 0; x < w1; x++)
                {
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _outp = op(_a0, _p1);
                    vst1q_f32(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                ptr += 4;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 6
                float32x4_t _b0 = vdupq_n_f32(b[0]);
                const float* ptr = a;
                float* outptr = c;
                for (int i = 0; i < w; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _outp = op(_p, _b0);
                    vst1q_f32(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }

            // type 7
            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;
            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr1);
                float32x4_t _outp = op(_p, _p1);
                vst1q_f32(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_pack4(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    float32x4_t _b = vdupq_n_f32(b);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = op(_p, _b);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
    }

    return 0;
}

struct binary_op_add_pack4
{
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
    {
        return vaddq_f32(x, y);
    }
};

struct binary_op_sub_pack4
{
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
    {
        return vsubq_f32(x, y);
    }
};

struct binary_op_mul_pack4
{
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
    {
        return vmulq_f32(x, y);
    }
};

struct binary_op_div_pack4
{
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
#if __aarch64__
    {
        return vdivq_f32(x, y);
    }
#else
    {
        return div_ps(x, y);
    }
#endif
};

struct binary_op_max_pack4
{
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
    {
        return vmaxq_f32(x, y);
    }
};

struct binary_op_min_pack4
{
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
    {
        return vminq_f32(x, y);
    }
};

struct binary_op_pow_pack4
{
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
    {
        return pow_ps(x, y);
    }
};

struct binary_op_rsub_pack4
{
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
    {
        return vsubq_f32(y, x);
    }
};

struct binary_op_rdiv_pack4
{
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
#if __aarch64__
    {
        return vdivq_f32(y, x);
    }
#else
    {
        return div_ps(y, x);
    }
#endif
};
#endif // __ARM_NEON

int BinaryOp_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = std::max(bottom_blobs[0].elembits(), bottom_blobs[1].elembits());

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_fp16s(bottom_blobs, top_blobs, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);

    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

#if __ARM_NEON
    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;

    if (elempack == 4 || elempack1 == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack4<binary_op_add_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack4<binary_op_sub_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack4<binary_op_mul_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack4<binary_op_div_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack4<binary_op_max_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack4<binary_op_min_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack4<binary_op_pow_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack4<binary_op_rsub_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack4<binary_op_rdiv_pack4>(bottom_blob, bottom_blob1, top_blob, opt);
    }
#endif // __ARM_NEON

    return BinaryOp::forward(bottom_blobs, top_blobs, opt);
}

int BinaryOp_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);

#if __ARM_NEON
    int elempack = bottom_top_blob.elempack;

    if (elempack == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_scalar_inplace_pack4<binary_op_add_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_SUB)
            return binary_op_scalar_inplace_pack4<binary_op_sub_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_MUL)
            return binary_op_scalar_inplace_pack4<binary_op_mul_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_DIV)
            return binary_op_scalar_inplace_pack4<binary_op_div_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_MAX)
            return binary_op_scalar_inplace_pack4<binary_op_max_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_MIN)
            return binary_op_scalar_inplace_pack4<binary_op_min_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_POW)
            return binary_op_scalar_inplace_pack4<binary_op_pow_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_RSUB)
            return binary_op_scalar_inplace_pack4<binary_op_rsub_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_RDIV)
            return binary_op_scalar_inplace_pack4<binary_op_rdiv_pack4>(bottom_top_blob, b, opt);
    }
#endif // __ARM_NEON

    return BinaryOp::forward_inplace(bottom_top_blob, opt);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template<typename Op>
static int binary_op_pack8_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    size_t elemsize = a.elemsize;
    int elempack = a.elempack;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;
    int size1 = w1 * h1;
    size_t elemsize1 = b.elemsize;
    int elempack1 = b.elempack;

    if (a.dims == 3)
    {
        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    __fp16* outptr = c.channel(q);
                    const __fp16* b0 = b.channel(q);
                    float16x8_t _b0 = vld1q_f16(b0);
                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _outp = op(_p, _b0);
                        vst1q_f16(outptr, _outp);
                        ptr += 8;
                        outptr += 8;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1 && elempack1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b;
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _p1 = vdupq_n_f16(*ptr1);
                        float16x8_t _outp = op(_p, _p1);
                        vst1q_f16(outptr, _outp);
                        ptr += 8;
                        ptr1 += 1;
                        outptr += 8;
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* a0 = a.channel(q);
                    __fp16* outptr = c.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    float16x8_t _a0 = vld1q_f16(a0);
                    for (int i = 0; i < size1; i++)
                    {
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        float16x8_t _outp = op(_a0, _p1);
                        vst1q_f16(outptr, _outp);
                        ptr1 += 8;
                        outptr += 8;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1 && elempack == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr = a;
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        float16x8_t _p = vdupq_n_f16(*ptr);
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        float16x8_t _outp = op(_p, _p1);
                        vst1q_f16(outptr, _outp);
                        ptr += 1;
                        ptr1 += 8;
                        outptr += 8;
                    }
                }

                return 0;
            }

            // type 19
            c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _outp = op(_p, _p1);
                    vst1q_f16(outptr, _outp);
                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
            }

            return 0;
        }

        c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.row<const __fp16>(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    float16x8_t _b0 = vld1q_f16(ptr1);
                    for (int x = 0; x < w; x++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _outp = op(_p, _b0);
                        vst1q_f16(outptr, _outp);
                        ptr += 8;
                        outptr += 8;
                    }

                    ptr1 += 8;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 16
                float16x8_t _b0 = vdupq_n_f16(((const __fp16*)b)[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _outp = op(_p, _b0);
                        vst1q_f16(outptr, _outp);
                        ptr += 8;
                        outptr += 8;
                    }
                }

                return 0;
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                float16x8_t _b0 = vld1q_f16((const __fp16*)b + q * 8);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _outp = op(_p, _b0);
                    vst1q_f16(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    float16x8_t _a0 = vld1q_f16(ptr);
                    for (int x = 0; x < w1; x++)
                    {
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        float16x8_t _outp = op(_a0, _p1);
                        vst1q_f16(outptr, _outp);
                        ptr1 += 8;
                        outptr += 8;
                    }

                    ptr += 8;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;
            for (int i = 0; i < size; i++)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float16x8_t _p1 = vld1q_f16(ptr1);
                float16x8_t _outp = op(_p, _p1);
                vst1q_f16(outptr, _outp);
                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                float16x8_t _b0 = vdupq_n_f16(((const __fp16*)b)[0]);
                const __fp16* ptr = a;
                __fp16* outptr = c;
                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _outp = op(_p, _b0);
                    vst1q_f16(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }

                return 0;
            }

            // type 12
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h; y++)
            {
                float16x8_t _b0 = vld1q_f16(ptr1);
                for (int x = 0; x < w; x++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _outp = op(_p, _b0);
                    vst1q_f16(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }

                ptr1 += 8;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1 && elempack == 1)
        {
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float16x8_t _a0 = vdupq_n_f16(((const __fp16*)a)[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        float16x8_t _outp = op(_a0, _p1);
                        vst1q_f16(outptr, _outp);
                        ptr1 += 8;
                        outptr += 8;
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                // type 3
                c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float16x8_t _a0 = vdupq_n_f16(((const __fp16*)a)[0]);
                const __fp16* ptr1 = b;
                __fp16* outptr = c;
                for (int i = 0; i < size1; i++)
                {
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _outp = op(_a0, _p1);
                    vst1q_f16(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float16x8_t _a0 = vdupq_n_f16(((const __fp16*)a)[0]);
                const __fp16* ptr1 = b;
                __fp16* outptr = c;
                for (int i = 0; i < w1; i++)
                {
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _outp = op(_a0, _p1);
                    vst1q_f16(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }

                return 0;
            }
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                float16x8_t _a0 = vld1q_f16((const __fp16*)a + q * 8);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _outp = op(_a0, _p1);
                    vst1q_f16(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                float16x8_t _a0 = vld1q_f16(ptr);
                for (int x = 0; x < w1; x++)
                {
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _outp = op(_a0, _p1);
                    vst1q_f16(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }

                ptr += 8;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 6
                float16x8_t _b0 = vdupq_n_f16(((const __fp16*)b)[0]);
                const __fp16* ptr = a;
                __fp16* outptr = c;
                for (int i = 0; i < w; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _outp = op(_p, _b0);
                    vst1q_f16(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }

                return 0;
            }

            // type 7
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;
            for (int i = 0; i < w; i++)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float16x8_t _p1 = vld1q_f16(ptr1);
                float16x8_t _outp = op(_p, _p1);
                vst1q_f16(outptr, _outp);
                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_pack8_fp16s(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    float16x8_t _b = vdupq_n_f16((__fp16)b);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = op(_p, _b);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
    }

    return 0;
}

struct binary_op_add_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const
    {
        return vaddq_f16(x, y);
    }
};

struct binary_op_sub_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const
    {
        return vsubq_f16(x, y);
    }
};

struct binary_op_mul_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const
    {
        return vmulq_f16(x, y);
    }
};

struct binary_op_div_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const
    {
        return vdivq_f16(x, y);
    }
};

struct binary_op_max_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const
    {
        return vmaxq_f16(x, y);
    }
};

struct binary_op_min_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const
    {
        return vminq_f16(x, y);
    }
};

struct binary_op_pow_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const
    {
        float16x4_t z_low = vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_low_f16(y))));
        float16x4_t z_high = vcvt_f16_f32(pow_ps(vcvt_f32_f16(vget_high_f16(x)), vcvt_f32_f16(vget_high_f16(y))));
        return vcombine_f16(z_low, z_high);
    }
};

struct binary_op_rsub_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const
    {
        return vsubq_f16(y, x);
    }
};

struct binary_op_rdiv_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x, const float16x8_t& y) const
    {
        return vdivq_f16(y, x);
    }
};

template<typename Op>
static int binary_op_pack4_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    size_t elemsize = a.elemsize;
    int elempack = a.elempack;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;
    int size1 = w1 * h1;
    size_t elemsize1 = b.elemsize;
    int elempack1 = b.elempack;

    if (a.dims == 3)
    {
        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    __fp16* outptr = c.channel(q);
                    const __fp16* b0 = b.channel(q);
                    float16x4_t _b0 = vld1_f16(b0);
                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _outp = op(_p, _b0);
                        vst1_f16(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1 && elempack1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b;
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _p1 = vdup_n_f16(*ptr1);
                        float16x4_t _outp = op(_p, _p1);
                        vst1_f16(outptr, _outp);
                        ptr += 4;
                        ptr1 += 1;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* a0 = a.channel(q);
                    __fp16* outptr = c.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    float16x4_t _a0 = vld1_f16(a0);
                    for (int i = 0; i < size1; i++)
                    {
                        float16x4_t _p1 = vld1_f16(ptr1);
                        float16x4_t _outp = op(_a0, _p1);
                        vst1_f16(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1 && elempack == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr = a;
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        float16x4_t _p = vdup_n_f16(*ptr);
                        float16x4_t _p1 = vld1_f16(ptr1);
                        float16x4_t _outp = op(_p, _p1);
                        vst1_f16(outptr, _outp);
                        ptr += 1;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            // type 19
            c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _p1 = vld1_f16(ptr1);
                    float16x4_t _outp = op(_p, _p1);
                    vst1_f16(outptr, _outp);
                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }

        c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.row<const __fp16>(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    float16x4_t _b0 = vld1_f16(ptr1);
                    for (int x = 0; x < w; x++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _outp = op(_p, _b0);
                        vst1_f16(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }

                    ptr1 += 4;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 16
                float16x4_t _b0 = vdup_n_f16(((const __fp16*)b)[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _outp = op(_p, _b0);
                        vst1_f16(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                float16x4_t _b0 = vld1_f16((const __fp16*)b + q * 4);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _outp = op(_p, _b0);
                    vst1_f16(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    float16x4_t _a0 = vld1_f16(ptr);
                    for (int x = 0; x < w1; x++)
                    {
                        float16x4_t _p1 = vld1_f16(ptr1);
                        float16x4_t _outp = op(_a0, _p1);
                        vst1_f16(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }

                    ptr += 4;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;
            for (int i = 0; i < size; i++)
            {
                float16x4_t _p = vld1_f16(ptr);
                float16x4_t _p1 = vld1_f16(ptr1);
                float16x4_t _outp = op(_p, _p1);
                vst1_f16(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                float16x4_t _b0 = vdup_n_f16(((const __fp16*)b)[0]);
                const __fp16* ptr = a;
                __fp16* outptr = c;
                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _outp = op(_p, _b0);
                    vst1_f16(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }

            // type 12
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h; y++)
            {
                float16x4_t _b0 = vld1_f16(ptr1);
                for (int x = 0; x < w; x++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _outp = op(_p, _b0);
                    vst1_f16(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                ptr1 += 4;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1 && elempack == 1)
        {
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float16x4_t _a0 = vdup_n_f16(((const __fp16*)a)[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        float16x4_t _p1 = vld1_f16(ptr1);
                        float16x4_t _outp = op(_a0, _p1);
                        vst1_f16(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                // type 3
                c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float16x4_t _a0 = vdup_n_f16(((const __fp16*)a)[0]);
                const __fp16* ptr1 = b;
                __fp16* outptr = c;
                for (int i = 0; i < size1; i++)
                {
                    float16x4_t _p1 = vld1_f16(ptr1);
                    float16x4_t _outp = op(_a0, _p1);
                    vst1_f16(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float16x4_t _a0 = vdup_n_f16(((const __fp16*)a)[0]);
                const __fp16* ptr1 = b;
                __fp16* outptr = c;
                for (int i = 0; i < w1; i++)
                {
                    float16x4_t _p1 = vld1_f16(ptr1);
                    float16x4_t _outp = op(_a0, _p1);
                    vst1_f16(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                float16x4_t _a0 = vld1_f16((const __fp16*)a + q * 4);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    float16x4_t _p1 = vld1_f16(ptr1);
                    float16x4_t _outp = op(_a0, _p1);
                    vst1_f16(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                float16x4_t _a0 = vld1_f16(ptr);
                for (int x = 0; x < w1; x++)
                {
                    float16x4_t _p1 = vld1_f16(ptr1);
                    float16x4_t _outp = op(_a0, _p1);
                    vst1_f16(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                ptr += 4;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 6
                float16x4_t _b0 = vdup_n_f16(((const __fp16*)b)[0]);
                const __fp16* ptr = a;
                __fp16* outptr = c;
                for (int i = 0; i < w; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _outp = op(_p, _b0);
                    vst1_f16(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }

            // type 7
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;
            for (int i = 0; i < w; i++)
            {
                float16x4_t _p = vld1_f16(ptr);
                float16x4_t _p1 = vld1_f16(ptr1);
                float16x4_t _outp = op(_p, _p1);
                vst1_f16(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_pack4_fp16s(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    float16x4_t _b = vdup_n_f16((__fp16)b);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = op(_p, _b);
            vst1_f16(ptr, _p);
            ptr += 4;
        }
    }

    return 0;
}

struct binary_op_add_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const
    {
        return vadd_f16(x, y);
    }
};

struct binary_op_sub_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const
    {
        return vsub_f16(x, y);
    }
};

struct binary_op_mul_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const
    {
        return vmul_f16(x, y);
    }
};

struct binary_op_div_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const
    {
        return vdiv_f16(x, y);
    }
};

struct binary_op_max_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const
    {
        return vmax_f16(x, y);
    }
};

struct binary_op_min_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const
    {
        return vmin_f16(x, y);
    }
};

struct binary_op_pow_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const
    {
        return vcvt_f16_f32(pow_ps(vcvt_f32_f16(x), vcvt_f32_f16(y)));
    }
};

struct binary_op_rsub_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const
    {
        return vsub_f16(y, x);
    }
};

struct binary_op_rdiv_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x, const float16x4_t& y) const
    {
        return vdiv_f16(y, x);
    }
};

template<typename Op>
static int binary_op_fp16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    size_t elemsize = a.elemsize;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;
    int size1 = w1 * h1;

    if (a.dims == 3)
    {
        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16* b0 = b.channel(q);
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], b0[0]);
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    const __fp16* ptr1 = b;
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], ptr1[i]);
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* a0 = a.channel(q);
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(a0[0], ptr1[i]);
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr = a;
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(ptr[i], ptr1[i]);
                    }
                }

                return 0;
            }

            // type 19
            c.create(w, h, channels, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], ptr1[i]);
                }
            }

            return 0;
        }

        c.create(w, h, channels, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16* ptr1 = b.row<const __fp16>(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    const __fp16 b0 = ptr1[y];
                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = op(ptr[x], b0);
                    }

                    ptr += w;
                    outptr += w;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1)
            {
                // type 16
                const __fp16 b0 = ((const __fp16*)b)[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = a.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], b0);
                    }
                }

                return 0;
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = a.channel(q);
                const __fp16 b0 = ((const __fp16*)b)[q];
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], b0);
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const __fp16* ptr = a.row<const __fp16>(q);
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    const __fp16 a0 = ptr[y];
                    for (int x = 0; x < w1; x++)
                    {
                        outptr[x] = op(a0, ptr1[x]);
                    }

                    ptr1 += w1;
                    outptr += w1;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;
            for (int i = 0; i < size; i++)
            {
                outptr[i] = op(ptr[i], ptr1[i]);
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                // type 11
                const __fp16 b0 = ((const __fp16*)b)[0];
                const __fp16* ptr = a;
                __fp16* outptr = c;
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], b0);
                }

                return 0;
            }

            // type 12
            const __fp16* ptr = a;
            __fp16* outptr = c;

            for (int y = 0; y < h; y++)
            {
                const __fp16 b0 = ((const __fp16*)b)[y];
                for (int x = 0; x < w; x++)
                {
                    outptr[x] = op(ptr[x], b0);
                }

                ptr += w;
                outptr += w;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1)
        {
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const __fp16 a0 = ((const __fp16*)a)[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const __fp16* ptr1 = b.channel(q);
                    __fp16* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(a0, ptr1[i]);
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                // type 3
                c.create(w1, h1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const __fp16 a0 = ((const __fp16*)a)[0];
                const __fp16* ptr1 = b;
                __fp16* outptr = c;
                for (int i = 0; i < size1; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const __fp16 a0 = ((const __fp16*)a)[0];
                const __fp16* ptr1 = b;
                __fp16* outptr = c;
                for (int i = 0; i < w1; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }

                return 0;
            }
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const __fp16 a0 = ((const __fp16*)a)[q];
                const __fp16* ptr1 = b.channel(q);
                __fp16* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            const __fp16* ptr1 = b;
            __fp16* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                const __fp16 a0 = ((const __fp16*)a)[y];
                for (int x = 0; x < w1; x++)
                {
                    outptr[x] = op(a0, ptr1[x]);
                }

                ptr1 += w1;
                outptr += w1;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                // type 6
                const __fp16 b0 = ((const __fp16*)b)[0];
                const __fp16* ptr = a;
                __fp16* outptr = c;
                for (int i = 0; i < w; i++)
                {
                    outptr[i] = op(ptr[i], b0);
                }

                return 0;
            }

            // type 7
            const __fp16* ptr = a;
            const __fp16* ptr1 = b;
            __fp16* outptr = c;
            for (int i = 0; i < w; i++)
            {
                outptr[i] = op(ptr[i], ptr1[i]);
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_fp16s(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    __fp16 b16 = (__fp16)b;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            ptr[i] = op(ptr[i], b16);
        }
    }

    return 0;
}

struct binary_op_add_fp16s
{
    __fp16 operator()(const __fp16& x, const __fp16& y) const
    {
        return x + y;
    }
};

struct binary_op_sub_fp16s
{
    __fp16 operator()(const __fp16& x, const __fp16& y) const
    {
        return x - y;
    }
};

struct binary_op_mul_fp16s
{
    __fp16 operator()(const __fp16& x, const __fp16& y) const
    {
        return x * y;
    }
};

struct binary_op_div_fp16s
{
    __fp16 operator()(const __fp16& x, const __fp16& y) const
    {
        return x / y;
    }
};

struct binary_op_max_fp16s
{
    __fp16 operator()(const __fp16& x, const __fp16& y) const
    {
        return std::max(x, y);
    }
};

struct binary_op_min_fp16s
{
    __fp16 operator()(const __fp16& x, const __fp16& y) const
    {
        return std::min(x, y);
    }
};

struct binary_op_pow_fp16s
{
    __fp16 operator()(const __fp16& x, const __fp16& y) const
    {
        return (__fp16)pow(x, y);
    }
};

struct binary_op_rsub_fp16s
{
    __fp16 operator()(const __fp16& x, const __fp16& y) const
    {
        return y - x;
    }
};

struct binary_op_rdiv_fp16s
{
    __fp16 operator()(const __fp16& x, const __fp16& y) const
    {
        return y / x;
    }
};

int BinaryOp_arm::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];

    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;

    if (elempack == 8 || elempack1 == 8)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack8_fp16s<binary_op_add_pack8_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack8_fp16s<binary_op_sub_pack8_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack8_fp16s<binary_op_mul_pack8_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack8_fp16s<binary_op_div_pack8_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack8_fp16s<binary_op_max_pack8_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack8_fp16s<binary_op_min_pack8_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack8_fp16s<binary_op_pow_pack8_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack8_fp16s<binary_op_rsub_pack8_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack8_fp16s<binary_op_rdiv_pack8_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);
    }

    if (elempack == 4 || elempack1 == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack4_fp16s<binary_op_add_pack4_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack4_fp16s<binary_op_sub_pack4_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack4_fp16s<binary_op_mul_pack4_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack4_fp16s<binary_op_div_pack4_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack4_fp16s<binary_op_max_pack4_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack4_fp16s<binary_op_min_pack4_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack4_fp16s<binary_op_pow_pack4_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack4_fp16s<binary_op_rsub_pack4_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack4_fp16s<binary_op_rdiv_pack4_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);
    }

    if (elempack == 1 && elempack1 == 1)
    {
        if (op_type == Operation_ADD)
            return binary_op_fp16s<binary_op_add_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_fp16s<binary_op_sub_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_fp16s<binary_op_mul_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_fp16s<binary_op_div_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_fp16s<binary_op_max_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_fp16s<binary_op_min_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_fp16s<binary_op_pow_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_fp16s<binary_op_rsub_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_fp16s<binary_op_rdiv_fp16s>(bottom_blob, bottom_blob1, top_blob, opt);
    }

    return 0;
}

int BinaryOp_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;

    if (elempack == 8)
    {
        if (op_type == Operation_ADD)
            return binary_op_scalar_inplace_pack8_fp16s<binary_op_add_pack8_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_SUB)
            return binary_op_scalar_inplace_pack8_fp16s<binary_op_sub_pack8_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_MUL)
            return binary_op_scalar_inplace_pack8_fp16s<binary_op_mul_pack8_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_DIV)
            return binary_op_scalar_inplace_pack8_fp16s<binary_op_div_pack8_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_MAX)
            return binary_op_scalar_inplace_pack8_fp16s<binary_op_max_pack8_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_MIN)
            return binary_op_scalar_inplace_pack8_fp16s<binary_op_min_pack8_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_POW)
            return binary_op_scalar_inplace_pack8_fp16s<binary_op_pow_pack8_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_RSUB)
            return binary_op_scalar_inplace_pack8_fp16s<binary_op_rsub_pack8_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_RDIV)
            return binary_op_scalar_inplace_pack8_fp16s<binary_op_rdiv_pack8_fp16s>(bottom_top_blob, b, opt);
    }

    if (elempack == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_scalar_inplace_pack4_fp16s<binary_op_add_pack4_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_SUB)
            return binary_op_scalar_inplace_pack4_fp16s<binary_op_sub_pack4_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_MUL)
            return binary_op_scalar_inplace_pack4_fp16s<binary_op_mul_pack4_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_DIV)
            return binary_op_scalar_inplace_pack4_fp16s<binary_op_div_pack4_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_MAX)
            return binary_op_scalar_inplace_pack4_fp16s<binary_op_max_pack4_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_MIN)
            return binary_op_scalar_inplace_pack4_fp16s<binary_op_min_pack4_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_POW)
            return binary_op_scalar_inplace_pack4_fp16s<binary_op_pow_pack4_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_RSUB)
            return binary_op_scalar_inplace_pack4_fp16s<binary_op_rsub_pack4_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_RDIV)
            return binary_op_scalar_inplace_pack4_fp16s<binary_op_rdiv_pack4_fp16s>(bottom_top_blob, b, opt);
    }

    if (elempack == 1)
    {
        if (op_type == Operation_ADD)
            return binary_op_scalar_inplace_fp16s<binary_op_add_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_SUB)
            return binary_op_scalar_inplace_fp16s<binary_op_sub_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_MUL)
            return binary_op_scalar_inplace_fp16s<binary_op_mul_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_DIV)
            return binary_op_scalar_inplace_fp16s<binary_op_div_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_MAX)
            return binary_op_scalar_inplace_fp16s<binary_op_max_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_MIN)
            return binary_op_scalar_inplace_fp16s<binary_op_min_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_POW)
            return binary_op_scalar_inplace_fp16s<binary_op_pow_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_RSUB)
            return binary_op_scalar_inplace_fp16s<binary_op_rsub_fp16s>(bottom_top_blob, b, opt);

        if (op_type == Operation_RDIV)
            return binary_op_scalar_inplace_fp16s<binary_op_rdiv_fp16s>(bottom_top_blob, b, opt);
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if __ARM_NEON
template<typename Op>
static int binary_op_pack4_bf16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    size_t elemsize = a.elemsize;
    int elempack = a.elempack;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;
    int size1 = w1 * h1;
    size_t elemsize1 = b.elemsize;
    int elempack1 = b.elempack;

    if (a.dims == 3)
    {
        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = a.channel(q);
                    unsigned short* outptr = c.channel(q);
                    const unsigned short* b0 = b.channel(q);
                    float32x4_t _b0 = vcvt_f32_bf16(vld1_u16(b0));
                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                        float32x4_t _outp = op(_p, _b0);
                        vst1_u16(outptr, vcvt_bf16_f32(_outp));
                        ptr += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1 && elempack1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = a.channel(q);
                    const unsigned short* ptr1 = b;
                    unsigned short* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                        float32x4_t _p1 = vdupq_n_f32(bfloat16_to_float32(*ptr1));
                        float32x4_t _outp = op(_p, _p1);
                        vst1_u16(outptr, vcvt_bf16_f32(_outp));
                        ptr += 4;
                        ptr1 += 1;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const unsigned short* a0 = a.channel(q);
                    unsigned short* outptr = c.channel(q);
                    const unsigned short* ptr1 = b.channel(q);
                    float32x4_t _a0 = vcvt_f32_bf16(vld1_u16(a0));
                    for (int i = 0; i < size1; i++)
                    {
                        float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                        float32x4_t _outp = op(_a0, _p1);
                        vst1_u16(outptr, vcvt_bf16_f32(_outp));
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1 && elempack == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const unsigned short* ptr = a;
                    const unsigned short* ptr1 = b.channel(q);
                    unsigned short* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        float32x4_t _p = vdupq_n_f32(bfloat16_to_float32(*ptr));
                        float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                        float32x4_t _outp = op(_p, _p1);
                        vst1_u16(outptr, vcvt_bf16_f32(_outp));
                        ptr += 1;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            // type 19
            c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = a.channel(q);
                const unsigned short* ptr1 = b.channel(q);
                unsigned short* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                    float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                    float32x4_t _outp = op(_p, _p1);
                    vst1_u16(outptr, vcvt_bf16_f32(_outp));
                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }

        c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = a.channel(q);
                const unsigned short* ptr1 = b.row<const unsigned short>(q);
                unsigned short* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    float32x4_t _b0 = vcvt_f32_bf16(vld1_u16(ptr1));
                    for (int x = 0; x < w; x++)
                    {
                        float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                        float32x4_t _outp = op(_p, _b0);
                        vst1_u16(outptr, vcvt_bf16_f32(_outp));
                        ptr += 4;
                        outptr += 4;
                    }

                    ptr1 += 4;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1 && elempack1 == 1)
            {
                // type 16
                float32x4_t _b0 = vdupq_n_f32(bfloat16_to_float32(((const unsigned short*)b)[0]));
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = a.channel(q);
                    unsigned short* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                        float32x4_t _outp = op(_p, _b0);
                        vst1_u16(outptr, vcvt_bf16_f32(_outp));
                        ptr += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = a.channel(q);
                float32x4_t _b0 = vcvt_f32_bf16(vld1_u16((const unsigned short*)b + q * 4));
                unsigned short* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                    float32x4_t _outp = op(_p, _b0);
                    vst1_u16(outptr, vcvt_bf16_f32(_outp));
                    ptr += 4;
                    outptr += 4;
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const unsigned short* ptr = a.row<const unsigned short>(q);
                const unsigned short* ptr1 = b.channel(q);
                unsigned short* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    float32x4_t _a0 = vcvt_f32_bf16(vld1_u16(ptr));
                    for (int x = 0; x < w1; x++)
                    {
                        float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                        float32x4_t _outp = op(_a0, _p1);
                        vst1_u16(outptr, vcvt_bf16_f32(_outp));
                        ptr1 += 4;
                        outptr += 4;
                    }

                    ptr += 4;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            const unsigned short* ptr = a;
            const unsigned short* ptr1 = b;
            unsigned short* outptr = c;
            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                float32x4_t _outp = op(_p, _p1);
                vst1_u16(outptr, vcvt_bf16_f32(_outp));
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 11
                float32x4_t _b0 = vdupq_n_f32(bfloat16_to_float32(((const unsigned short*)b)[0]));
                const unsigned short* ptr = a;
                unsigned short* outptr = c;
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                    float32x4_t _outp = op(_p, _b0);
                    vst1_u16(outptr, vcvt_bf16_f32(_outp));
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }

            // type 12
            const unsigned short* ptr = a;
            const unsigned short* ptr1 = b;
            unsigned short* outptr = c;

            for (int y = 0; y < h; y++)
            {
                float32x4_t _b0 = vcvt_f32_bf16(vld1_u16(ptr1));
                for (int x = 0; x < w; x++)
                {
                    float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                    float32x4_t _outp = op(_p, _b0);
                    vst1_u16(outptr, vcvt_bf16_f32(_outp));
                    ptr += 4;
                    outptr += 4;
                }

                ptr1 += 4;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1 && elempack == 1)
        {
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float32x4_t _a0 = vdupq_n_f32(bfloat16_to_float32(((const unsigned short*)a)[0]));
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const unsigned short* ptr1 = b.channel(q);
                    unsigned short* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                        float32x4_t _outp = op(_a0, _p1);
                        vst1_u16(outptr, vcvt_bf16_f32(_outp));
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                // type 3
                c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float32x4_t _a0 = vdupq_n_f32(bfloat16_to_float32(((const unsigned short*)a)[0]));
                const unsigned short* ptr1 = b;
                unsigned short* outptr = c;
                for (int i = 0; i < size1; i++)
                {
                    float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                    float32x4_t _outp = op(_a0, _p1);
                    vst1_u16(outptr, vcvt_bf16_f32(_outp));
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize1, elempack1, opt.blob_allocator);
                if (c.empty())
                    return -100;

                float32x4_t _a0 = vdupq_n_f32(bfloat16_to_float32(((const unsigned short*)a)[0]));
                const unsigned short* ptr1 = b;
                unsigned short* outptr = c;
                for (int i = 0; i < w1; i++)
                {
                    float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                    float32x4_t _outp = op(_a0, _p1);
                    vst1_u16(outptr, vcvt_bf16_f32(_outp));
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                float32x4_t _a0 = vcvt_f32_bf16(vld1_u16((const unsigned short*)a + q * 4));
                const unsigned short* ptr1 = b.channel(q);
                unsigned short* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                    float32x4_t _outp = op(_a0, _p1);
                    vst1_u16(outptr, vcvt_bf16_f32(_outp));
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize1, elempack1, opt.blob_allocator);
            if (c.empty())
                return -100;

            const unsigned short* ptr = a;
            const unsigned short* ptr1 = b;
            unsigned short* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                float32x4_t _a0 = vcvt_f32_bf16(vld1_u16(ptr));
                for (int x = 0; x < w1; x++)
                {
                    float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                    float32x4_t _outp = op(_a0, _p1);
                    vst1_u16(outptr, vcvt_bf16_f32(_outp));
                    ptr1 += 4;
                    outptr += 4;
                }

                ptr += 4;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1 && elempack1 == 1)
            {
                // type 6
                float32x4_t _b0 = vdupq_n_f32(bfloat16_to_float32(((const unsigned short*)b)[0]));
                const unsigned short* ptr = a;
                unsigned short* outptr = c;
                for (int i = 0; i < w; i++)
                {
                    float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                    float32x4_t _outp = op(_p, _b0);
                    vst1_u16(outptr, vcvt_bf16_f32(_outp));
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }

            // type 7
            const unsigned short* ptr = a;
            const unsigned short* ptr1 = b;
            unsigned short* outptr = c;
            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                float32x4_t _p1 = vcvt_f32_bf16(vld1_u16(ptr1));
                float32x4_t _outp = op(_p, _p1);
                vst1_u16(outptr, vcvt_bf16_f32(_outp));
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_pack4_bf16s(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    float32x4_t _b = vdupq_n_f32(b);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
            _p = op(_p, _b);
            vst1_u16(ptr, vcvt_bf16_f32(_p));
            ptr += 4;
        }
    }

    return 0;
}
#endif // __ARM_NEON

template<typename Op>
static int binary_op_bf16s(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    size_t elemsize = a.elemsize;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;
    int size1 = w1 * h1;

    if (a.dims == 3)
    {
        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = a.channel(q);
                    const unsigned short* b0 = b.channel(q);
                    unsigned short* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), bfloat16_to_float32(b0[0])));
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = a.channel(q);
                    const unsigned short* ptr1 = b;
                    unsigned short* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), bfloat16_to_float32(ptr1[i])));
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const unsigned short* a0 = a.channel(q);
                    const unsigned short* ptr1 = b.channel(q);
                    unsigned short* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(a0[0]), bfloat16_to_float32(ptr1[i])));
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const unsigned short* ptr = a;
                    const unsigned short* ptr1 = b.channel(q);
                    unsigned short* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), bfloat16_to_float32(ptr1[i])));
                    }
                }

                return 0;
            }

            // type 19
            c.create(w, h, channels, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = a.channel(q);
                const unsigned short* ptr1 = b.channel(q);
                unsigned short* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), bfloat16_to_float32(ptr1[i])));
                }
            }

            return 0;
        }

        c.create(w, h, channels, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = a.channel(q);
                const unsigned short* ptr1 = b.row<const unsigned short>(q);
                unsigned short* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    const float b0 = bfloat16_to_float32(ptr1[y]);
                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[x]), b0));
                    }

                    ptr += w;
                    outptr += w;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1)
            {
                // type 16
                const float b0 = bfloat16_to_float32(((const unsigned short*)b)[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = a.channel(q);
                    unsigned short* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), b0));
                    }
                }

                return 0;
            }

            // type 17
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = a.channel(q);
                const float b0 = bfloat16_to_float32(((const unsigned short*)b)[q]);
                unsigned short* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), b0));
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const unsigned short* ptr = a.row<const unsigned short>(q);
                const unsigned short* ptr1 = b.channel(q);
                unsigned short* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    const float a0 = bfloat16_to_float32(ptr[y]);
                    for (int x = 0; x < w1; x++)
                    {
                        outptr[x] = float32_to_bfloat16(op(a0, bfloat16_to_float32(ptr1[x])));
                    }

                    ptr1 += w1;
                    outptr += w1;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            const unsigned short* ptr = a;
            const unsigned short* ptr1 = b;
            unsigned short* outptr = c;
            for (int i = 0; i < size; i++)
            {
                outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), bfloat16_to_float32(ptr1[i])));
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                // type 11
                const float b0 = bfloat16_to_float32(((const unsigned short*)b)[0]);
                const unsigned short* ptr = a;
                unsigned short* outptr = c;
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), b0));
                }

                return 0;
            }

            // type 12
            const unsigned short* ptr = a;
            unsigned short* outptr = c;

            for (int y = 0; y < h; y++)
            {
                const float b0 = bfloat16_to_float32(((const unsigned short*)b)[y]);
                for (int x = 0; x < w; x++)
                {
                    outptr[x] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[x]), b0));
                }

                ptr += w;
                outptr += w;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1)
        {
            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = bfloat16_to_float32(((const unsigned short*)a)[0]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels1; q++)
                {
                    const unsigned short* ptr1 = b.channel(q);
                    unsigned short* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = float32_to_bfloat16(op(a0, bfloat16_to_float32(ptr1[i])));
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                // type 3
                c.create(w1, h1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = bfloat16_to_float32(((const unsigned short*)a)[0]);
                const unsigned short* ptr1 = b;
                unsigned short* outptr = c;
                for (int i = 0; i < size1; i++)
                {
                    outptr[i] = float32_to_bfloat16(op(a0, bfloat16_to_float32(ptr1[i])));
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = bfloat16_to_float32(((const unsigned short*)a)[0]);
                const unsigned short* ptr1 = b;
                unsigned short* outptr = c;
                for (int i = 0; i < w1; i++)
                {
                    outptr[i] = float32_to_bfloat16(op(a0, bfloat16_to_float32(ptr1[i])));
                }

                return 0;
            }
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels1; q++)
            {
                const float a0 = bfloat16_to_float32(((const unsigned short*)a)[q]);
                const unsigned short* ptr1 = b.channel(q);
                unsigned short* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    outptr[i] = float32_to_bfloat16(op(a0, bfloat16_to_float32(ptr1[i])));
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            const unsigned short* ptr1 = b;
            unsigned short* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                const float a0 = bfloat16_to_float32(((const unsigned short*)a)[y]);
                for (int x = 0; x < w1; x++)
                {
                    outptr[x] = float32_to_bfloat16(op(a0, bfloat16_to_float32(ptr1[x])));
                }

                ptr1 += w1;
                outptr += w1;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                // type 6
                const float b0 = bfloat16_to_float32(((const unsigned short*)b)[0]);
                const unsigned short* ptr = a;
                unsigned short* outptr = c;
                for (int i = 0; i < w; i++)
                {
                    outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), b0));
                }

                return 0;
            }

            // type 7
            const unsigned short* ptr = a;
            const unsigned short* ptr1 = b;
            unsigned short* outptr = c;
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), bfloat16_to_float32(ptr1[i])));
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace_bf16s(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            ptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i]), b));
        }
    }

    return 0;
}

struct binary_op_add
{
    float operator()(const float& x, const float& y) const
    {
        return x + y;
    }
};

struct binary_op_sub
{
    float operator()(const float& x, const float& y) const
    {
        return x - y;
    }
};

struct binary_op_mul
{
    float operator()(const float& x, const float& y) const
    {
        return x * y;
    }
};

struct binary_op_div
{
    float operator()(const float& x, const float& y) const
    {
        return x / y;
    }
};

struct binary_op_max
{
    float operator()(const float& x, const float& y) const
    {
        return std::max(x, y);
    }
};

struct binary_op_min
{
    float operator()(const float& x, const float& y) const
    {
        return std::min(x, y);
    }
};

struct binary_op_pow
{
    float operator()(const float& x, const float& y) const
    {
        return (float)pow(x, y);
    }
};

struct binary_op_rsub
{
    float operator()(const float& x, const float& y) const
    {
        return y - x;
    }
};

struct binary_op_rdiv
{
    float operator()(const float& x, const float& y) const
    {
        return y / x;
    }
};

int BinaryOp_arm::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];

    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;

#if __ARM_NEON
    if (elempack == 4 || elempack1 == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_pack4_bf16s<binary_op_add_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_pack4_bf16s<binary_op_sub_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_pack4_bf16s<binary_op_mul_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_pack4_bf16s<binary_op_div_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_pack4_bf16s<binary_op_max_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_pack4_bf16s<binary_op_min_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_pack4_bf16s<binary_op_pow_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_pack4_bf16s<binary_op_rsub_pack4>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_pack4_bf16s<binary_op_rdiv_pack4>(bottom_blob, bottom_blob1, top_blob, opt);
    }
#endif // __ARM_NEON

    if (elempack == 1 && elempack1 == 1)
    {
        if (op_type == Operation_ADD)
            return binary_op_bf16s<binary_op_add>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op_bf16s<binary_op_sub>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op_bf16s<binary_op_mul>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op_bf16s<binary_op_div>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op_bf16s<binary_op_max>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op_bf16s<binary_op_min>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op_bf16s<binary_op_pow>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op_bf16s<binary_op_rsub>(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op_bf16s<binary_op_rdiv>(bottom_blob, bottom_blob1, top_blob, opt);
    }

    return 0;
}

int BinaryOp_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_scalar_inplace_pack4_bf16s<binary_op_add_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_SUB)
            return binary_op_scalar_inplace_pack4_bf16s<binary_op_sub_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_MUL)
            return binary_op_scalar_inplace_pack4_bf16s<binary_op_mul_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_DIV)
            return binary_op_scalar_inplace_pack4_bf16s<binary_op_div_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_MAX)
            return binary_op_scalar_inplace_pack4_bf16s<binary_op_max_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_MIN)
            return binary_op_scalar_inplace_pack4_bf16s<binary_op_min_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_POW)
            return binary_op_scalar_inplace_pack4_bf16s<binary_op_pow_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_RSUB)
            return binary_op_scalar_inplace_pack4_bf16s<binary_op_rsub_pack4>(bottom_top_blob, b, opt);

        if (op_type == Operation_RDIV)
            return binary_op_scalar_inplace_pack4_bf16s<binary_op_rdiv_pack4>(bottom_top_blob, b, opt);
    }
#endif // __ARM_NEON

    if (elempack == 1)
    {
        if (op_type == Operation_ADD)
            return binary_op_scalar_inplace_bf16s<binary_op_add>(bottom_top_blob, b, opt);

        if (op_type == Operation_SUB)
            return binary_op_scalar_inplace_bf16s<binary_op_sub>(bottom_top_blob, b, opt);

        if (op_type == Operation_MUL)
            return binary_op_scalar_inplace_bf16s<binary_op_mul>(bottom_top_blob, b, opt);

        if (op_type == Operation_DIV)
            return binary_op_scalar_inplace_bf16s<binary_op_div>(bottom_top_blob, b, opt);

        if (op_type == Operation_MAX)
            return binary_op_scalar_inplace_bf16s<binary_op_max>(bottom_top_blob, b, opt);

        if (op_type == Operation_MIN)
            return binary_op_scalar_inplace_bf16s<binary_op_min>(bottom_top_blob, b, opt);

        if (op_type == Operation_POW)
            return binary_op_scalar_inplace_bf16s<binary_op_pow>(bottom_top_blob, b, opt);

        if (op_type == Operation_RSUB)
            return binary_op_scalar_inplace_bf16s<binary_op_rsub>(bottom_top_blob, b, opt);

        if (op_type == Operation_RDIV)
            return binary_op_scalar_inplace_bf16s<binary_op_rdiv>(bottom_top_blob, b, opt);
    }

    return 0;
}

} // namespace ncnn
