// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "glu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static inline float32x4_t load_fp16s(const __fp16* ptr)
{
    return vcvt_f32_f16(vld1_f16(ptr));
}

static inline void store_fp16s(__fp16* ptr, float32x4_t v)
{
    vst1_f16(ptr, vcvt_f16_f32(v));
}

static inline float load_fp16s1(const __fp16* ptr)
{
    return (float)*ptr;
}

static inline void store_fp16s1(__fp16* ptr, float v)
{
    *ptr = (__fp16)v;
}

static void glu_vector_fp16s(const __fp16* ptr, const __fp16* gateptr, __fp16* outptr, int size)
{
    int i = 0;
#if __aarch64__
    for (; i + 15 < size; i += 16)
    {
        float16x8_t _p01 = vld1q_f16(ptr);
        float16x8_t _p23 = vld1q_f16(ptr + 8);
        float16x8_t _g01 = vld1q_f16(gateptr);
        float16x8_t _g23 = vld1q_f16(gateptr + 8);
        float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p01));
        float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p01));
        float32x4_t _p2 = vcvt_f32_f16(vget_low_f16(_p23));
        float32x4_t _p3 = vcvt_f32_f16(vget_high_f16(_p23));
        float32x4_t _g0 = vcvt_f32_f16(vget_low_f16(_g01));
        float32x4_t _g1 = vcvt_f32_f16(vget_high_f16(_g01));
        float32x4_t _g2 = vcvt_f32_f16(vget_low_f16(_g23));
        float32x4_t _g3 = vcvt_f32_f16(vget_high_f16(_g23));
        vst1q_f16(outptr, vcombine_f16(vcvt_f16_f32(vmulq_f32(_p0, sigmoid_ps(_g0))), vcvt_f16_f32(vmulq_f32(_p1, sigmoid_ps(_g1)))));
        vst1q_f16(outptr + 8, vcombine_f16(vcvt_f16_f32(vmulq_f32(_p2, sigmoid_ps(_g2))), vcvt_f16_f32(vmulq_f32(_p3, sigmoid_ps(_g3)))));
        ptr += 16;
        gateptr += 16;
        outptr += 16;
    }
#endif // __aarch64__
    for (; i + 7 < size; i += 8)
    {
        float16x8_t _p = vld1q_f16(ptr);
        float16x8_t _g = vld1q_f16(gateptr);
        float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
        float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
        float32x4_t _g0 = vcvt_f32_f16(vget_low_f16(_g));
        float32x4_t _g1 = vcvt_f32_f16(vget_high_f16(_g));
        vst1q_f16(outptr, vcombine_f16(vcvt_f16_f32(vmulq_f32(_p0, sigmoid_ps(_g0))), vcvt_f16_f32(vmulq_f32(_p1, sigmoid_ps(_g1)))));
        ptr += 8;
        gateptr += 8;
        outptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = load_fp16s(ptr);
        float32x4_t _g = load_fp16s(gateptr);
        store_fp16s(outptr, vmulq_f32(_p, sigmoid_ps(_g)));
        ptr += 4;
        gateptr += 4;
        outptr += 4;
    }
    for (; i < size; i++)
    {
        store_fp16s1(outptr, load_fp16s1(ptr) / (1.f + expf(-load_fp16s1(gateptr))));
        ptr++;
        gateptr++;
        outptr++;
    }
}

static int glu_forward_fp16s_pack1(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt)
{
    const int dims = bottom_blob.dims;
    const int positive_axis = dims == 1 ? 0 : axis < 0 ? dims + axis : axis;
    if (positive_axis < 0 || positive_axis >= dims)
        return -100;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int c = bottom_blob.c;
    const size_t elemsize = 2u;

    const int block_size = 1024;

    if (dims == 1)
    {
        const int outw = w / 2;
        top_blob.create(outw, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int nn_block = (outw + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int bi = 0; bi < nn_block; bi++)
        {
            const int offset = bi * block_size;
            const int size = std::min(block_size, outw - offset);

            const __fp16* ptr = (const __fp16*)bottom_blob + offset;
            const __fp16* gateptr = (const __fp16*)bottom_blob + outw + offset;
            __fp16* outptr = (__fp16*)top_blob + offset;

            glu_vector_fp16s(ptr, gateptr, outptr, size);
        }
        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int outh = h / 2;
        top_blob.create(w, outh, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * outh;
        const int nn_block = (size + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int bi = 0; bi < nn_block; bi++)
        {
            const int offset = bi * block_size;
            const int size1 = std::min(block_size, size - offset);

            const __fp16* ptr = (const __fp16*)bottom_blob + offset;
            const __fp16* gateptr = (const __fp16*)bottom_blob + size + offset;
            __fp16* outptr = (__fp16*)top_blob + offset;

            glu_vector_fp16s(ptr, gateptr, outptr, size1);
        }
        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        const int outw = w / 2;
        top_blob.create(outw, h, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int nn_block = (outw + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < h * nn_block; ii++)
        {
            const int i = ii / nn_block;
            const int bi = ii - i * nn_block;
            const int offset = bi * block_size;
            const int size = std::min(block_size, outw - offset);

            const __fp16* ptr = bottom_blob.row<__fp16>(i) + offset;
            const __fp16* gateptr = bottom_blob.row<__fp16>(i) + outw + offset;
            __fp16* outptr = top_blob.row<__fp16>(i) + offset;

            glu_vector_fp16s(ptr, gateptr, outptr, size);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 0)
    {
        const int outc = c / 2;
        top_blob.create(w, h, outc, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * h;
        const size_t gate_offset = (size_t)outc * bottom_blob.cstep;
        const int nn_block = (size + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < outc * nn_block; ii++)
        {
            const int q = ii / nn_block;
            const int bi = ii - q * nn_block;
            const int offset = bi * block_size;
            const int size1 = std::min(block_size, size - offset);

            const __fp16* ptr = (const __fp16*)bottom_blob.channel(q) + offset;
            const __fp16* gateptr = (const __fp16*)bottom_blob.channel(q) + gate_offset + offset;
            __fp16* outptr = (__fp16*)top_blob.channel(q) + offset;

            glu_vector_fp16s(ptr, gateptr, outptr, size1);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 1)
    {
        const int outh = h / 2;
        top_blob.create(w, outh, c, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * outh;
        const int nn_block = (size + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < c * nn_block; ii++)
        {
            const int q = ii / nn_block;
            const int bi = ii - q * nn_block;
            const int offset = bi * block_size;
            const int size1 = std::min(block_size, size - offset);

            const __fp16* ptr = (const __fp16*)bottom_blob.channel(q) + offset;
            const __fp16* gateptr = (const __fp16*)bottom_blob.channel(q) + size + offset;
            __fp16* outptr = (__fp16*)top_blob.channel(q) + offset;

            glu_vector_fp16s(ptr, gateptr, outptr, size1);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 2)
    {
        const int outw = w / 2;
        top_blob.create(outw, h, c, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int nn_block = (outw + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qi = 0; qi < c * h * nn_block; qi++)
        {
            const int q = qi / (h * nn_block);
            const int ri = qi - q * (h * nn_block);
            const int i = ri / nn_block;
            const int bi = ri - i * nn_block;
            const int offset = bi * block_size;
            const int size = std::min(block_size, outw - offset);

            const __fp16* ptr = bottom_blob.channel(q).row<__fp16>(i) + offset;
            const __fp16* gateptr = bottom_blob.channel(q).row<__fp16>(i) + outw + offset;
            __fp16* outptr = top_blob.channel(q).row<__fp16>(i) + offset;

            glu_vector_fp16s(ptr, gateptr, outptr, size);
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 0)
    {
        const int outc = c / 2;
        top_blob.create(w, h, d, outc, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * h * d;
        const size_t gate_offset = (size_t)outc * bottom_blob.cstep;
        const int nn_block = (size + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < outc * nn_block; ii++)
        {
            const int q = ii / nn_block;
            const int bi = ii - q * nn_block;
            const int offset = bi * block_size;
            const int size1 = std::min(block_size, size - offset);

            const __fp16* ptr = (const __fp16*)bottom_blob.channel(q) + offset;
            const __fp16* gateptr = (const __fp16*)bottom_blob.channel(q) + gate_offset + offset;
            __fp16* outptr = (__fp16*)top_blob.channel(q) + offset;

            glu_vector_fp16s(ptr, gateptr, outptr, size1);
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int outd = d / 2;
        top_blob.create(w, h, outd, c, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * h * outd;
        const int nn_block = (size + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < c * nn_block; ii++)
        {
            const int q = ii / nn_block;
            const int bi = ii - q * nn_block;
            const int offset = bi * block_size;
            const int size1 = std::min(block_size, size - offset);

            const __fp16* ptr = (const __fp16*)bottom_blob.channel(q) + offset;
            const __fp16* gateptr = (const __fp16*)bottom_blob.channel(q) + size + offset;
            __fp16* outptr = (__fp16*)top_blob.channel(q) + offset;

            glu_vector_fp16s(ptr, gateptr, outptr, size1);
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 2)
    {
        const int outh = h / 2;
        top_blob.create(w, outh, d, c, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * outh;
        const int nn_block = (size + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qz = 0; qz < c * d * nn_block; qz++)
        {
            const int q = qz / (d * nn_block);
            const int rz = qz - q * (d * nn_block);
            const int z = rz / nn_block;
            const int bi = rz - z * nn_block;
            const int offset = bi * block_size;
            const int size1 = std::min(block_size, size - offset);

            const __fp16* ptr = (const __fp16*)bottom_blob.channel(q).depth(z) + offset;
            const __fp16* gateptr = (const __fp16*)bottom_blob.channel(q).depth(z) + size + offset;
            __fp16* outptr = (__fp16*)top_blob.channel(q).depth(z) + offset;

            glu_vector_fp16s(ptr, gateptr, outptr, size1);
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 3)
    {
        const int outw = w / 2;
        top_blob.create(outw, h, d, c, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int nn_block = (outw + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qzi = 0; qzi < c * d * h * nn_block; qzi++)
        {
            const int q = qzi / (d * h * nn_block);
            const int rzi = qzi - q * (d * h * nn_block);
            const int z = rzi / (h * nn_block);
            const int ri = rzi - z * (h * nn_block);
            const int i = ri / nn_block;
            const int bi = ri - i * nn_block;
            const int offset = bi * block_size;
            const int size = std::min(block_size, outw - offset);

            const __fp16* ptr = bottom_blob.channel(q).depth(z).row<__fp16>(i) + offset;
            const __fp16* gateptr = bottom_blob.channel(q).depth(z).row<__fp16>(i) + outw + offset;
            __fp16* outptr = top_blob.channel(q).depth(z).row<__fp16>(i) + offset;

            glu_vector_fp16s(ptr, gateptr, outptr, size);
        }
        return 0;
    }

    return -100;
}

static int glu_forward_fp16s(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt)
{
    if (bottom_blob.elempack == 1)
        return glu_forward_fp16s_pack1(bottom_blob, top_blob, axis, opt);

    Option opt_pack1 = opt;
    opt_pack1.blob_allocator = opt.workspace_allocator;

    Mat bottom_blob_pack1;
    convert_packing(bottom_blob, bottom_blob_pack1, 1, opt_pack1);
    if (bottom_blob_pack1.empty())
        return -100;

    return glu_forward_fp16s_pack1(bottom_blob_pack1, top_blob, axis, opt);
}

int GLU_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    return glu_forward_fp16s(bottom_blob, top_blob, axis, opt);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
