// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "glu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

GLU_arm::GLU_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

static void glu_vector(const float* ptr, const float* gateptr, float* outptr, int size)
{
    int i = 0;
#if __ARM_NEON
#if __aarch64__
    for (; i + 15 < size; i += 16)
    {
        float32x4_t _p0 = vld1q_f32(ptr);
        float32x4_t _p1 = vld1q_f32(ptr + 4);
        float32x4_t _p2 = vld1q_f32(ptr + 8);
        float32x4_t _p3 = vld1q_f32(ptr + 12);
        float32x4_t _g0 = vld1q_f32(gateptr);
        float32x4_t _g1 = vld1q_f32(gateptr + 4);
        float32x4_t _g2 = vld1q_f32(gateptr + 8);
        float32x4_t _g3 = vld1q_f32(gateptr + 12);
        vst1q_f32(outptr, vmulq_f32(_p0, sigmoid_ps(_g0)));
        vst1q_f32(outptr + 4, vmulq_f32(_p1, sigmoid_ps(_g1)));
        vst1q_f32(outptr + 8, vmulq_f32(_p2, sigmoid_ps(_g2)));
        vst1q_f32(outptr + 12, vmulq_f32(_p3, sigmoid_ps(_g3)));
        ptr += 16;
        gateptr += 16;
        outptr += 16;
    }
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _p0 = vld1q_f32(ptr);
        float32x4_t _p1 = vld1q_f32(ptr + 4);
        float32x4_t _g0 = vld1q_f32(gateptr);
        float32x4_t _g1 = vld1q_f32(gateptr + 4);
        vst1q_f32(outptr, vmulq_f32(_p0, sigmoid_ps(_g0)));
        vst1q_f32(outptr + 4, vmulq_f32(_p1, sigmoid_ps(_g1)));
        ptr += 8;
        gateptr += 8;
        outptr += 8;
    }
#endif // __aarch64__
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _g = vld1q_f32(gateptr);
        vst1q_f32(outptr, vmulq_f32(_p, sigmoid_ps(_g)));
        ptr += 4;
        gateptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *outptr = *ptr * (1.f / (1.f + expf(-*gateptr)));
        ptr++;
        gateptr++;
        outptr++;
    }
}

// the four lanes of ptr are the values of four pack1 output channels
// their gates are lanes 2 3 of gateptr0 and lanes 0 1 of gateptr1
// only the first two output channels are stored when outch == 2
static void glu_pack4to1(const float* ptr, const float* gateptr0, const float* gateptr1, float* outptr, size_t out_cstep, int outch, int size)
{
    float* outptr0 = outptr;
    float* outptr1 = outptr + out_cstep;
    float* outptr2 = outptr + out_cstep * 2;
    float* outptr3 = outptr + out_cstep * 3;

    int i = 0;
#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4x4_t _p = vld4q_f32(ptr);
        float32x4x4_t _g0 = vld4q_f32(gateptr0);
        vst1q_f32(outptr0, vmulq_f32(_p.val[0], sigmoid_ps(_g0.val[2])));
        vst1q_f32(outptr1, vmulq_f32(_p.val[1], sigmoid_ps(_g0.val[3])));
        if (outch == 4)
        {
            float32x4x4_t _g1 = vld4q_f32(gateptr1);
            vst1q_f32(outptr2, vmulq_f32(_p.val[2], sigmoid_ps(_g1.val[0])));
            vst1q_f32(outptr3, vmulq_f32(_p.val[3], sigmoid_ps(_g1.val[1])));
        }

        ptr += 16;
        gateptr0 += 16;
        gateptr1 += 16;
        outptr0 += 4;
        outptr1 += 4;
        outptr2 += 4;
        outptr3 += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        outptr0[0] = ptr[0] * (1.f / (1.f + expf(-gateptr0[2])));
        outptr1[0] = ptr[1] * (1.f / (1.f + expf(-gateptr0[3])));
        if (outch == 4)
        {
            outptr2[0] = ptr[2] * (1.f / (1.f + expf(-gateptr1[0])));
            outptr3[0] = ptr[3] * (1.f / (1.f + expf(-gateptr1[1])));
        }

        ptr += 4;
        gateptr0 += 4;
        gateptr1 += 4;
        outptr0++;
        outptr1++;
        outptr2++;
        outptr3++;
    }
}

int GLU_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int positive_axis = dims == 1 ? 0 : axis < 0 ? dims + axis : axis;
    if (positive_axis < 0 || positive_axis >= dims)
        return -100;

    const int elempack = bottom_blob.elempack;

    int outw = bottom_blob.w;
    int outh = bottom_blob.h;
    int outd = bottom_blob.d;
    int outc = bottom_blob.c;
    int out_elempack = elempack;

    if (positive_axis == 0)
    {
        const int out_size = (dims == 1 ? outw : dims == 2 ? outh : outc) * elempack / 2;
        if (out_size % out_elempack != 0)
            out_elempack = 1;

        if (dims == 1)
            outw = out_size / out_elempack;
        else if (dims == 2)
            outh = out_size / out_elempack;
        else
            outc = out_size / out_elempack;
    }
    else if (positive_axis == dims - 1)
    {
        outw /= 2;
    }
    else if (positive_axis == dims - 2)
    {
        outh /= 2;
    }
    else
    {
        // dims == 4 && positive_axis == 1
        outd /= 2;
    }

    const size_t out_elemsize = bottom_blob.elemsize / elempack * out_elempack;

    if (dims == 1)
        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
    else if (dims == 2)
        top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    else if (dims == 3)
        top_blob.create(outw, outh, outc, out_elemsize, out_elempack, opt.blob_allocator);
    else
        top_blob.create(outw, outh, outd, outc, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims >= 2 && positive_axis == 0 && out_elempack != elempack)
    {
        // elempack == 4 and out_size % 4 == 2, the gates of four output channels start at lane 2 of an input pack
        const int size = dims == 2 ? outw : outw * outh * outd;
        const int out_outer = dims == 2 ? outh : outc;
        const int out_size = out_outer;
        const size_t cstep = (dims == 2 ? (size_t)bottom_blob.w : bottom_blob.cstep) * elempack;
        const size_t out_cstep = dims == 2 ? (size_t)outw : top_blob.cstep;

        const int block_size = 256;
        const int nn_outer = (out_outer + 3) / 4;
        const int nn_block = (size + block_size - 1) / block_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_outer * nn_block; ii++)
        {
            const int qq = ii / nn_block;
            const int offset = ii % nn_block * block_size;
            const int q = qq * 4;
            const int outch = std::min(out_outer - q, 4);

            const float* ptr = (const float*)bottom_blob + qq * cstep + offset * 4;
            const float* gateptr0 = (const float*)bottom_blob + (q + out_size) / 4 * cstep + offset * 4;
            const float* gateptr1 = outch == 4 ? gateptr0 + cstep : gateptr0;
            float* outptr = (float*)top_blob + q * out_cstep + offset;

            glu_pack4to1(ptr, gateptr0, gateptr1, outptr, out_cstep, outch, std::min(size - offset, block_size));
        }

        return 0;
    }

    // every output channel is a series of contiguous spans
    // dims == 1 is contiguous whatever the packing is
    const int inner_axis = dims - 1 - positive_axis;
    int span_size = outw * out_elempack;
    if (inner_axis >= 1)
        span_size *= outh;
    if (inner_axis >= 2)
        span_size *= outd;

    int input_stride = bottom_blob.w * elempack;
    if (inner_axis >= 1)
        input_stride *= bottom_blob.h;
    if (inner_axis >= 2)
        input_stride *= bottom_blob.d;

    const int spans = outw * outh * outd * out_elempack / span_size;
    const size_t gate_offset = dims >= 3 && positive_axis == 0 ? outc * bottom_blob.cstep * elempack : (size_t)span_size;

    // gather short spans and split long spans, so that each task handles about block_size elements
    const int block_size = 1024;
    const int span_count = span_size < block_size ? block_size / span_size : 1;
    const int nn_span = (spans + span_count - 1) / span_count;
    const int nn_block = (span_size + block_size - 1) / block_size;

    if (span_size < 4)
    {
        // spans shorter than one vector, plain scalar code avoids entering the simd kernel for every span
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < outc * nn_span; ii++)
        {
            const int q = ii / nn_span;
            const int span_start = ii % nn_span * span_count;
            const int span_end = std::min(span_start + span_count, spans);

            const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep * elempack + span_start * input_stride;
            float* outptr = (float*)top_blob + q * top_blob.cstep * out_elempack + span_start * span_size;

            for (int j = span_start; j < span_end; j++)
            {
                for (int k = 0; k < span_size; k++)
                {
                    outptr[k] = ptr[k] * (1.f / (1.f + expf(-ptr[k + gate_offset])));
                }
                ptr += input_stride;
                outptr += span_size;
            }
        }

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < outc * nn_span * nn_block; ii++)
    {
        const int q = ii / (nn_span * nn_block);
        const int span_start = ii % (nn_span * nn_block) / nn_block * span_count;
        const int span_end = std::min(span_start + span_count, spans);
        const int offset = ii % nn_block * block_size;
        const int size = std::min(span_size - offset, block_size);

        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep * elempack + span_start * input_stride + offset;
        float* outptr = (float*)top_blob + q * top_blob.cstep * out_elempack + span_start * span_size + offset;

        for (int j = span_start; j < span_end; j++)
        {
            glu_vector(ptr, ptr + gate_offset, outptr, size);
            ptr += input_stride;
            outptr += span_size;
        }
    }

    return 0;
}

} // namespace ncnn
