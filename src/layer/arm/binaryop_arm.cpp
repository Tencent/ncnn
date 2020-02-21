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
#include <algorithm>
#include <functional>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(BinaryOp_arm)

BinaryOp_arm::BinaryOp_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

#if __ARM_NEON
// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

template<typename Op>
static int binary_op(const Mat& a, const Mat& b, Mat& c, const Option& opt)
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
        c.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = c.channel(q);
                    const float* b0 = b.channel(q);
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

            // type 19
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i=0; i<size; i++)
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

        if (b.dims == 2)
        {
            // type 18
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int y=0; y<h; y++)
                {
                    float32x4_t _b0 = vld1q_f32(ptr1);
                    for (int x=0; x<w; x++)
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
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = c.channel(q);

                    for (int i=0; i<size; i++)
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
            for (int q=0; q<channels; q++)
            {
                const float* ptr = a.channel(q);
                float32x4_t _b0 = vld1q_f32((const float*)b + q * 4);
                float* outptr = c.channel(q);

                for (int i=0; i<size; i++)
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
            c.create(w1, h1, channels1, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels1; q++)
            {
                const float* ptr = a.row(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int y=0; y<h1; y++)
                {
                    float32x4_t _a0 = vld1q_f32(ptr);
                    for (int x=0; x<w1; x++)
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
            for (int i=0; i<size; i++)
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
                for (int i=0; i<size; i++)
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

            for (int y=0; y<h; y++)
            {
                float32x4_t _b0 = vld1q_f32(ptr1);
                for (int x=0; x<w; x++)
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
                for (int q=0; q<channels1; q++)
                {
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int i=0; i<size1; i++)
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
                for (int i=0; i<size1; i++)
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
                for (int i=0; i<w1; i++)
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
            c.create(w1, h1, channels1, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels1; q++)
            {
                float32x4_t _a0 = vld1q_f32((const float*)a + q * 4);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i=0; i<size1; i++)
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
            c.create(w1, h1, elemsize, elempack, opt.blob_allocator);
            if (c.empty())
                return -100;

            const float* ptr = a;
            const float* ptr1 = b;
            float* outptr = c;

            for (int y=0; y<h1; y++)
            {
                float32x4_t _a0 = vld1q_f32(ptr);
                for (int x=0; x<w1; x++)
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
                for (int i=0; i<w; i++)
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
            for (int i=0; i<w; i++)
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
static int binary_op_scalar_inplace(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    float32x4_t _b = vdupq_n_f32(b);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i=0; i<size; i++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = op(_p, _b);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
    }

    return 0;
}

template<typename T>
struct binary_op_add {
    T operator() (const T& x, const T& y) const { return vaddq_f32(x, y); }
};

template<typename T>
struct binary_op_sub {
    T operator() (const T& x, const T& y) const { return vsubq_f32(x, y); }
};

template<typename T>
struct binary_op_mul {
    T operator() (const T& x, const T& y) const { return vmulq_f32(x, y); }
};

template<typename T>
struct binary_op_div {
    T operator() (const T& x, const T& y) const
#if __aarch64__
    { return vdivq_f32(x, y); }
#else
    { return div_ps(x, y); }
#endif
};

template<typename T>
struct binary_op_max {
    T operator() (const T& x, const T& y) const { return vmaxq_f32(x, y); }
};

template<typename T>
struct binary_op_min {
    T operator() (const T& x, const T& y) const { return vminq_f32(x, y); }
};

template<typename T>
struct binary_op_pow {
    T operator() (const T& x, const T& y) const { return pow_ps(x, y); }
};

template<typename T>
struct binary_op_rsub {
    T operator() (const T& x, const T& y) const { return vsubq_f32(y, x); }
};

template<typename T>
struct binary_op_rdiv {
    T operator() (const T& x, const T& y) const
#if __aarch64__
    { return vdivq_f32(y, x); }
#else
    { return div_ps(y, x); }
#endif
};
#endif // __ARM_NEON

int BinaryOp_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    int elempack = bottom_blob.elempack;
    int elempack1 = bottom_blob1.elempack;

    if (elempack == 4 || elempack1 == 4)
    {

        if (op_type == Operation_ADD)
            return binary_op< binary_op_add<float32x4_t> >(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_SUB)
            return binary_op< binary_op_sub<float32x4_t> >(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MUL)
            return binary_op< binary_op_mul<float32x4_t> >(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_DIV)
            return binary_op< binary_op_div<float32x4_t> >(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MAX)
            return binary_op< binary_op_max<float32x4_t> >(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_MIN)
            return binary_op< binary_op_min<float32x4_t> >(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_POW)
            return binary_op< binary_op_pow<float32x4_t> >(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RSUB)
            return binary_op< binary_op_rsub<float32x4_t> >(bottom_blob, bottom_blob1, top_blob, opt);

        if (op_type == Operation_RDIV)
            return binary_op< binary_op_rdiv<float32x4_t> >(bottom_blob, bottom_blob1, top_blob, opt);

    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return BinaryOp::forward(bottom_blobs, top_blobs, opt);
}

int BinaryOp_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    int elempack = bottom_top_blob.elempack;

    if (elempack == 4)
    {
        if (op_type == Operation_ADD)
            return binary_op_scalar_inplace< binary_op_add<float32x4_t> >(bottom_top_blob, b, opt);

        if (op_type == Operation_SUB)
            return binary_op_scalar_inplace< binary_op_sub<float32x4_t> >(bottom_top_blob, b, opt);

        if (op_type == Operation_MUL)
            return binary_op_scalar_inplace< binary_op_mul<float32x4_t> >(bottom_top_blob, b, opt);

        if (op_type == Operation_DIV)
            return binary_op_scalar_inplace< binary_op_div<float32x4_t> >(bottom_top_blob, b, opt);

        if (op_type == Operation_MAX)
            return binary_op_scalar_inplace< binary_op_max<float32x4_t> >(bottom_top_blob, b, opt);

        if (op_type == Operation_MIN)
            return binary_op_scalar_inplace< binary_op_min<float32x4_t> >(bottom_top_blob, b, opt);

        if (op_type == Operation_POW)
            return binary_op_scalar_inplace< binary_op_pow<float32x4_t> >(bottom_top_blob, b, opt);

        if (op_type == Operation_RSUB)
            return binary_op_scalar_inplace< binary_op_rsub<float32x4_t> >(bottom_top_blob, b, opt);

        if (op_type == Operation_RDIV)
            return binary_op_scalar_inplace< binary_op_rdiv<float32x4_t> >(bottom_top_blob, b, opt);

    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return BinaryOp::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
