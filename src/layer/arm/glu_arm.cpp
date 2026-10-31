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

static void glu(const float* input, const float* gate, float* output, int size)
{
    int i = 0;
#if __ARM_NEON
#if __aarch64__
    for (; i + 15 < size; i += 16)
    {
        float32x4_t _x0 = vld1q_f32(input);
        float32x4_t _x1 = vld1q_f32(input + 4);
        float32x4_t _x2 = vld1q_f32(input + 8);
        float32x4_t _x3 = vld1q_f32(input + 12);
        float32x4_t _g0 = vld1q_f32(gate);
        float32x4_t _g1 = vld1q_f32(gate + 4);
        float32x4_t _g2 = vld1q_f32(gate + 8);
        float32x4_t _g3 = vld1q_f32(gate + 12);
        _g0 = sigmoid_ps(_g0);
        _g1 = sigmoid_ps(_g1);
        _g2 = sigmoid_ps(_g2);
        _g3 = sigmoid_ps(_g3);
        vst1q_f32(output, vmulq_f32(_x0, _g0));
        vst1q_f32(output + 4, vmulq_f32(_x1, _g1));
        vst1q_f32(output + 8, vmulq_f32(_x2, _g2));
        vst1q_f32(output + 12, vmulq_f32(_x3, _g3));
        input += 16;
        gate += 16;
        output += 16;
    }
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _x0 = vld1q_f32(input);
        float32x4_t _x1 = vld1q_f32(input + 4);
        float32x4_t _g0 = vld1q_f32(gate);
        float32x4_t _g1 = vld1q_f32(gate + 4);
        _g0 = sigmoid_ps(_g0);
        _g1 = sigmoid_ps(_g1);
        vst1q_f32(output, vmulq_f32(_x0, _g0));
        vst1q_f32(output + 4, vmulq_f32(_x1, _g1));
        input += 8;
        gate += 8;
        output += 8;
    }
#endif // __aarch64__
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _x = vld1q_f32(input);
        float32x4_t _g = vld1q_f32(gate);
        _g = sigmoid_ps(_g);
        vst1q_f32(output, vmulq_f32(_x, _g));
        input += 4;
        gate += 4;
        output += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *output = *input * (1.f / (1.f + expf(-*gate)));
        input++;
        gate++;
        output++;
    }
}

// Deinterleave four spatial positions, then write the selected logical channel
// directly to its final pack1 output. Input/gate may select different lanes.
static void glu_pack4to1(const float* input, const float* gate, float* output, int size, int input_lane, int gate_lane)
{
    int i = 0;
#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        const float32x4x4_t _input = vld4q_f32(input);
        const float32x4x4_t _gate = vld4q_f32(gate);
        const float32x4_t _x = input_lane == 0 ? _input.val[0] : input_lane == 1 ? _input.val[1]
                                                             : input_lane == 2   ? _input.val[2]
                                                                                 : _input.val[3];
        const float32x4_t _g = gate_lane == 0 ? _gate.val[0] : gate_lane == 1 ? _gate.val[1]
                                                           : gate_lane == 2   ? _gate.val[2]
                                                                              : _gate.val[3];
        vst1q_f32(output, vmulq_f32(_x, sigmoid_ps(_g)));
        input += 16;
        gate += 16;
        output += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        *output = input[input_lane] * (1.f / (1.f + expf(-gate[gate_lane])));
        input += 4;
        gate += 4;
        output++;
    }
}

int GLU_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const int positive_axis = dims == 1 ? 0 : axis < 0 ? dims + axis
                                                       : axis;
    if (dims < 1 || dims > 4 || positive_axis < 0 || positive_axis >= dims
        || (elempack != 1 && elempack != 4) || bottom_blob.elembits() != 32)
        return -100;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int out_elempack = elempack;

    if (positive_axis == 0)
    {
        const int outer = dims == 1 ? w : dims == 2 ? h
                                                    : channels;
        const int out_outer = outer * elempack / 2;
        if (out_outer % out_elempack != 0)
            out_elempack = 1;

        if (dims == 1)
            w = out_outer / out_elempack;
        else if (dims == 2)
            h = out_outer / out_elempack;
        else
            channels = out_outer / out_elempack;
    }
    else if (positive_axis == dims - 1)
        w /= 2;
    else if (positive_axis == dims - 2)
        h /= 2;
    else if (dims == 4 && positive_axis == 1)
        d /= 2;

    const size_t out_elemsize = bottom_blob.elemsize / elempack * out_elempack;
    if (dims == 1)
        top_blob.create(w, out_elemsize, out_elempack, opt.blob_allocator);
    else if (dims == 2)
        top_blob.create(w, h, out_elemsize, out_elempack, opt.blob_allocator);
    else if (dims == 3)
        top_blob.create(w, h, channels, out_elemsize, out_elempack, opt.blob_allocator);
    else
        top_blob.create(w, h, d, channels, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims >= 2 && elempack == 4 && out_elempack == 1)
    {
        const int size = dims == 2 ? w : w * h * d;
        const int out_outer = dims == 2 ? h : channels;
        const size_t input_outer_stride = dims == 2 ? (size_t)bottom_blob.w * elempack : bottom_blob.cstep * elempack;
        const size_t output_outer_stride = dims == 2 ? (size_t)w : top_blob.cstep;

        if (opt.num_threads == 1)
        {
            for (int q = 0; q < out_outer; q++)
            {
                const int gate_q = q + out_outer;
                const float* input = (const float*)bottom_blob + (q / 4) * input_outer_stride;
                const float* gate = (const float*)bottom_blob + (gate_q / 4) * input_outer_stride;
                float* output = (float*)top_blob + q * output_outer_stride;
                glu_pack4to1(input, gate, output, size, q % 4, gate_q % 4);
            }
            return 0;
        }

        const int block_size = 1024;
        const int blocks_per_outer = (size + block_size - 1) / block_size;
        const int block_count = out_outer * blocks_per_outer;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < block_count; i++)
        {
            const int q = i / blocks_per_outer;
            const int gate_q = q + out_outer;
            const int offset = i % blocks_per_outer * block_size;
            const int block_length = size - offset < block_size ? size - offset : block_size;
            const float* input = (const float*)bottom_blob + (q / 4) * input_outer_stride + offset * elempack;
            const float* gate = (const float*)bottom_blob + (gate_q / 4) * input_outer_stride + offset * elempack;
            float* output = (float*)top_blob + q * output_outer_stride + offset;
            glu_pack4to1(input, gate, output, block_length, q % 4, gate_q % 4);
        }
        return 0;
    }

    const int inner_axis = dims - 1 - positive_axis;
    int span_size = w * out_elempack;
    if (inner_axis >= 1)
        span_size *= h;
    if (inner_axis >= 2)
        span_size *= d;

    const bool split_channels = dims >= 3 && positive_axis == 0;
    const int spans_per_channel = w * h * d * out_elempack / span_size;
    const size_t gate_offset = split_channels ? channels * bottom_blob.cstep * elempack : (size_t)span_size;
    int input_stride = bottom_blob.w * elempack;
    if (inner_axis >= 1)
        input_stride *= bottom_blob.h;
    if (inner_axis >= 2)
        input_stride *= bottom_blob.d;

    // A single worker can consume each complete span without an OpenMP region
    // or repeatedly splitting the same input/gate pair into small blocks.
    if (opt.num_threads == 1)
    {
        for (int q = 0; q < channels; q++)
        {
            const float* input = (const float*)bottom_blob + q * bottom_blob.cstep * elempack;
            float* output = (float*)top_blob + q * top_blob.cstep * out_elempack;
            for (int span = 0; span < spans_per_channel; span++)
            {
                glu(input, input + gate_offset, output, span_size);
                input += input_stride;
                output += span_size;
            }
        }
        return 0;
    }

    // Split long spans too, so vectors and single-row decode use all threads.
    const int block_size = 1024;
    const int blocks_per_span = (span_size + block_size - 1) / block_size;
    const int blocks_per_channel = spans_per_channel * blocks_per_span;
    const int block_count = channels * blocks_per_channel;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < block_count; i++)
    {
        const int q = i / blocks_per_channel;
        const int span = i % blocks_per_channel / blocks_per_span;
        const int offset = i % blocks_per_span * block_size;
        const int size = span_size - offset < block_size ? span_size - offset : block_size;
        const float* input = (const float*)bottom_blob + q * bottom_blob.cstep * elempack + span * input_stride + offset;
        const float* gate = input + gate_offset;
        float* output = (float*)top_blob + q * top_blob.cstep * out_elempack + span * span_size + offset;
        glu(input, gate, output, size);
    }

    return 0;
}

} // namespace ncnn
