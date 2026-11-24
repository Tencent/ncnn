// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

// full register tiles have no size or packing branches

// tile names use input columns x rows
// contiguous tiles take only pointers; stride variants use scalar-element strides

static NCNN_FORCEINLINE void permute_transpose2x4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    int _p0;
    memcpy(&_p0, ptr, 4);
    uint16x8_t _r0 = permute_set_u32(_p0);
    int _p1;
    memcpy(&_p1, ptr + stride, 4);
    uint16x8_t _r1 = permute_set_u32(_p1);
    int _p2;
    memcpy(&_p2, ptr + 2 * stride, 4);
    uint16x8_t _r2 = permute_set_u32(_p2);
    int _p3;
    memcpy(&_p3, ptr + 3 * stride, 4);
    uint16x8_t _r3 = permute_set_u32(_p3);
    uint16x8_t _t0 = permute_zip1_u32(_r0, _r1);
    uint16x8_t _t1 = permute_zip1_u32(_r2, _r3);
    uint16x8_t _v0 = permute_zip1_u64(_t0, _t1);
    _v0 = permute_deinterleave_u16(_v0);
    permute_storel_u16x8((unsigned short*)outptr, _v0);
    vst1_u16(outptr + outstride, vget_high_u16(_v0));
}

static NCNN_FORCEINLINE void permute_transpose4x4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr)
{
    uint16x4x4_t _v = vld4_u16(ptr);
    vst1_u16(outptr, _v.val[0]);
    vst1_u16(outptr + 4, _v.val[1]);
    vst1_u16(outptr + 8, _v.val[2]);
    vst1_u16(outptr + 12, _v.val[3]);
}

static NCNN_FORCEINLINE void permute_transpose4x4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    uint16x4_t _r0 = vld1_u16(ptr);
    uint16x4_t _r1 = vld1_u16(ptr + stride);
    uint16x4_t _r2 = vld1_u16(ptr + 2 * stride);
    uint16x4_t _r3 = vld1_u16(ptr + 3 * stride);
    transpose4x4_u16(_r0, _r1, _r2, _r3);
    vst1_u16(outptr, _r0);
    vst1_u16(outptr + outstride, _r1);
    vst1_u16(outptr + 2 * outstride, _r2);
    vst1_u16(outptr + 3 * outstride, _r3);
}

static NCNN_FORCEINLINE void permute_transpose8x4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr)
{
    uint16x8_t _r0 = vld1q_u16((const unsigned short*)(ptr));
    uint16x8_t _r1 = vld1q_u16((const unsigned short*)(ptr + 8));
    uint16x8_t _r2 = vld1q_u16((const unsigned short*)(ptr + 16));
    uint16x8_t _r3 = vld1q_u16((const unsigned short*)(ptr + 24));
    transpose8x4_u16(_r0, _r1, _r2, _r3);
    vst1q_u16((unsigned short*)(outptr), _r0);
    vst1q_u16((unsigned short*)(outptr + 8), _r1);
    vst1q_u16((unsigned short*)(outptr + 16), _r2);
    vst1q_u16((unsigned short*)(outptr + 24), _r3);
}

static NCNN_FORCEINLINE void permute_transpose8x4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    uint16x8_t _r0 = vld1q_u16((const unsigned short*)(ptr));
    uint16x8_t _r1 = vld1q_u16((const unsigned short*)(ptr + stride));
    uint16x8_t _r2 = vld1q_u16((const unsigned short*)(ptr + 2 * stride));
    uint16x8_t _r3 = vld1q_u16((const unsigned short*)(ptr + 3 * stride));
    transpose8x4_u16(_r0, _r1, _r2, _r3);
    permute_storel_u16x8((unsigned short*)(outptr), _r0);
    vst1_u16(outptr + outstride, vget_high_u16(_r0));
    permute_storel_u16x8((unsigned short*)(outptr + 2 * outstride), _r1);
    vst1_u16(outptr + 3 * outstride, vget_high_u16(_r1));
    permute_storel_u16x8((unsigned short*)(outptr + 4 * outstride), _r2);
    vst1_u16(outptr + 5 * outstride, vget_high_u16(_r2));
    permute_storel_u16x8((unsigned short*)(outptr + 6 * outstride), _r3);
    vst1_u16(outptr + 7 * outstride, vget_high_u16(_r3));
}

static NCNN_FORCEINLINE void permute_transpose2x8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    int _p0;
    memcpy(&_p0, ptr, 4);
    uint16x8_t _r0 = permute_set_u32(_p0);
    int _p1;
    memcpy(&_p1, ptr + stride, 4);
    uint16x8_t _r1 = permute_set_u32(_p1);
    int _p2;
    memcpy(&_p2, ptr + 2 * stride, 4);
    uint16x8_t _r2 = permute_set_u32(_p2);
    int _p3;
    memcpy(&_p3, ptr + 3 * stride, 4);
    uint16x8_t _r3 = permute_set_u32(_p3);
    uint16x8_t _t0 = permute_zip1_u32(_r0, _r1);
    uint16x8_t _t1 = permute_zip1_u32(_r2, _r3);
    uint16x8_t _v0 = permute_zip1_u64(_t0, _t1);
    _v0 = permute_deinterleave_u16(_v0);
    int _p4;
    memcpy(&_p4, ptr + 4 * stride, 4);
    uint16x8_t _r4 = permute_set_u32(_p4);
    int _p5;
    memcpy(&_p5, ptr + 5 * stride, 4);
    uint16x8_t _r5 = permute_set_u32(_p5);
    int _p6;
    memcpy(&_p6, ptr + 6 * stride, 4);
    uint16x8_t _r6 = permute_set_u32(_p6);
    int _p7;
    memcpy(&_p7, ptr + 7 * stride, 4);
    uint16x8_t _r7 = permute_set_u32(_p7);
    uint16x8_t _t2 = permute_zip1_u32(_r4, _r5);
    uint16x8_t _t3 = permute_zip1_u32(_r6, _r7);
    uint16x8_t _v1 = permute_zip1_u64(_t2, _t3);
    _v1 = permute_deinterleave_u16(_v1);
    vst1q_u16((unsigned short*)outptr, permute_zip1_u64(_v0, _v1));
    vst1q_u16((unsigned short*)(outptr + outstride), permute_zip2_u64(_v0, _v1));
}

static NCNN_FORCEINLINE void permute_transpose4x8_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr)
{
    uint16x8_t _r0 = vld1q_u16((const unsigned short*)(ptr));
    uint16x8_t _r1 = vld1q_u16((const unsigned short*)(ptr + 8));
    uint16x8_t _r2 = vld1q_u16((const unsigned short*)(ptr + 16));
    uint16x8_t _r3 = vld1q_u16((const unsigned short*)(ptr + 24));
    uint16x8_t _t0 = vzip1q_u16(_r0, _r1);
    uint16x8_t _t1 = vzip2q_u16(_r0, _r1);
    uint16x8_t _t2 = vzip1q_u16(_r2, _r3);
    uint16x8_t _t3 = vzip2q_u16(_r2, _r3);
    uint16x8_t _a = vzip1q_u16(_t0, _t1);
    uint16x8_t _b = vzip2q_u16(_t0, _t1);
    uint16x8_t _c = vzip1q_u16(_t2, _t3);
    uint16x8_t _d = vzip2q_u16(_t2, _t3);
    _r0 = permute_zip1_u64(_a, _c);
    _r1 = permute_zip2_u64(_a, _c);
    _r2 = permute_zip1_u64(_b, _d);
    _r3 = permute_zip2_u64(_b, _d);
    vst1q_u16((unsigned short*)(outptr), _r0);
    vst1q_u16((unsigned short*)(outptr + 8), _r1);
    vst1q_u16((unsigned short*)(outptr + 16), _r2);
    vst1q_u16((unsigned short*)(outptr + 24), _r3);
}

static NCNN_FORCEINLINE void permute_transpose4x8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    uint16x4_t _r0 = vld1_u16(ptr);
    uint16x4_t _r1 = vld1_u16(ptr + stride);
    uint16x4_t _r2 = vld1_u16(ptr + 2 * stride);
    uint16x4_t _r3 = vld1_u16(ptr + 3 * stride);
    uint16x4_t _r4 = vld1_u16(ptr + 4 * stride);
    uint16x4_t _r5 = vld1_u16(ptr + 5 * stride);
    uint16x4_t _r6 = vld1_u16(ptr + 6 * stride);
    uint16x4_t _r7 = vld1_u16(ptr + 7 * stride);
    transpose4x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
    vst1q_u16(outptr, vcombine_u16(_r0, _r1));
    vst1q_u16(outptr + outstride, vcombine_u16(_r2, _r3));
    vst1q_u16(outptr + 2 * outstride, vcombine_u16(_r4, _r5));
    vst1q_u16(outptr + 3 * outstride, vcombine_u16(_r6, _r7));
}

static NCNN_FORCEINLINE void permute_transpose8x8_u16(uint16x8_t& _r0, uint16x8_t& _r1, uint16x8_t& _r2, uint16x8_t& _r3, uint16x8_t& _r4, uint16x8_t& _r5, uint16x8_t& _r6, uint16x8_t& _r7)
{
    uint16x8x2_t _t0 = vtrnq_u16(_r0, _r1);
    uint16x8x2_t _t1 = vtrnq_u16(_r2, _r3);
    uint16x8x2_t _t2 = vtrnq_u16(_r4, _r5);
    uint16x8x2_t _t3 = vtrnq_u16(_r6, _r7);
    uint32x4x2_t _u0 = vtrnq_u32(vreinterpretq_u32_u16(_t0.val[0]), vreinterpretq_u32_u16(_t1.val[0]));
    uint32x4x2_t _u1 = vtrnq_u32(vreinterpretq_u32_u16(_t0.val[1]), vreinterpretq_u32_u16(_t1.val[1]));
    uint32x4x2_t _u2 = vtrnq_u32(vreinterpretq_u32_u16(_t2.val[0]), vreinterpretq_u32_u16(_t3.val[0]));
    uint32x4x2_t _u3 = vtrnq_u32(vreinterpretq_u32_u16(_t2.val[1]), vreinterpretq_u32_u16(_t3.val[1]));
    uint64x2_t _v0 = vreinterpretq_u64_u32(_u0.val[0]);
    uint64x2_t _v1 = vreinterpretq_u64_u32(_u2.val[0]);
    uint64x2_t _v2 = vreinterpretq_u64_u32(_u0.val[1]);
    uint64x2_t _v3 = vreinterpretq_u64_u32(_u2.val[1]);
    uint64x2_t _v4 = vreinterpretq_u64_u32(_u1.val[0]);
    uint64x2_t _v5 = vreinterpretq_u64_u32(_u3.val[0]);
    uint64x2_t _v6 = vreinterpretq_u64_u32(_u1.val[1]);
    uint64x2_t _v7 = vreinterpretq_u64_u32(_u3.val[1]);
    _r0 = vreinterpretq_u16_u64(vzip1q_u64(_v0, _v1));
    _r4 = vreinterpretq_u16_u64(vzip2q_u64(_v0, _v1));
    _r2 = vreinterpretq_u16_u64(vzip1q_u64(_v2, _v3));
    _r6 = vreinterpretq_u16_u64(vzip2q_u64(_v2, _v3));
    _r1 = vreinterpretq_u16_u64(vzip1q_u64(_v4, _v5));
    _r5 = vreinterpretq_u16_u64(vzip2q_u64(_v4, _v5));
    _r3 = vreinterpretq_u16_u64(vzip1q_u64(_v6, _v7));
    _r7 = vreinterpretq_u16_u64(vzip2q_u64(_v6, _v7));
}

static NCNN_FORCEINLINE void permute_transpose8x8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    uint16x8_t _r0 = vld1q_u16((const unsigned short*)(ptr));
    uint16x8_t _r1 = vld1q_u16((const unsigned short*)(ptr + stride));
    uint16x8_t _r2 = vld1q_u16((const unsigned short*)(ptr + 2 * stride));
    uint16x8_t _r3 = vld1q_u16((const unsigned short*)(ptr + 3 * stride));
    uint16x8_t _r4 = vld1q_u16((const unsigned short*)(ptr + 4 * stride));
    uint16x8_t _r5 = vld1q_u16((const unsigned short*)(ptr + 5 * stride));
    uint16x8_t _r6 = vld1q_u16((const unsigned short*)(ptr + 6 * stride));
    uint16x8_t _r7 = vld1q_u16((const unsigned short*)(ptr + 7 * stride));
    permute_transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
    vst1q_u16((unsigned short*)(outptr), _r0);
    vst1q_u16((unsigned short*)(outptr + outstride), _r1);
    vst1q_u16((unsigned short*)(outptr + 2 * outstride), _r2);
    vst1q_u16((unsigned short*)(outptr + 3 * outstride), _r3);
    vst1q_u16((unsigned short*)(outptr + 4 * outstride), _r4);
    vst1q_u16((unsigned short*)(outptr + 5 * outstride), _r5);
    vst1q_u16((unsigned short*)(outptr + 6 * outstride), _r6);
    vst1q_u16((unsigned short*)(outptr + 7 * outstride), _r7);
}

// final two or one rows: load complete vectors and write only the valid output lanes

static NCNN_FORCEINLINE void permute_transpose8x2_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    uint16x8_t _r0 = vld1q_u16((const unsigned short*)ptr);
    uint16x8_t _r1 = vld1q_u16((const unsigned short*)(ptr + stride));
    uint16x8_t _t0 = vzip1q_u16(_r0, _r1);
    uint16x8_t _t1 = vzip2q_u16(_r0, _r1);
    uint16x8_t _v0 = _t0;
    int _p0 = permute_extract_u32(_v0);
    memcpy(outptr, &_p0, 4);
    int _p1 = permute_extract_u32(permute_shift_right_u16x8(_v0, 4));
    memcpy(outptr + outstride, &_p1, 4);
    int _p2 = permute_extract_u32(permute_shift_right_u16x8(_v0, 8));
    memcpy(outptr + 2 * outstride, &_p2, 4);
    int _p3 = permute_extract_u32(permute_shift_right_u16x8(_v0, 12));
    memcpy(outptr + 3 * outstride, &_p3, 4);
    uint16x8_t _v1 = _t1;
    int _p4 = permute_extract_u32(_v1);
    memcpy(outptr + 4 * outstride, &_p4, 4);
    int _p5 = permute_extract_u32(permute_shift_right_u16x8(_v1, 4));
    memcpy(outptr + 5 * outstride, &_p5, 4);
    int _p6 = permute_extract_u32(permute_shift_right_u16x8(_v1, 8));
    memcpy(outptr + 6 * outstride, &_p6, 4);
    int _p7 = permute_extract_u32(permute_shift_right_u16x8(_v1, 12));
    memcpy(outptr + 7 * outstride, &_p7, 4);
}

static NCNN_FORCEINLINE void permute_transpose4x2_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    uint16x8_t _r0 = permute_loadl_u16x8((const unsigned short*)ptr);
    uint16x8_t _r1 = permute_loadl_u16x8((const unsigned short*)(ptr + stride));
    uint16x8_t _t0 = vzip1q_u16(_r0, _r1);
    uint16x8_t _v0 = _t0;
    int _p0 = permute_extract_u32(_v0);
    memcpy(outptr, &_p0, 4);
    int _p1 = permute_extract_u32(permute_shift_right_u16x8(_v0, 4));
    memcpy(outptr + outstride, &_p1, 4);
    int _p2 = permute_extract_u32(permute_shift_right_u16x8(_v0, 8));
    memcpy(outptr + 2 * outstride, &_p2, 4);
    int _p3 = permute_extract_u32(permute_shift_right_u16x8(_v0, 12));
    memcpy(outptr + 3 * outstride, &_p3, 4);
}

static NCNN_FORCEINLINE void permute_transpose8x1_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, size_t outstride)
{
    uint16x8_t _v = vld1q_u16(ptr);
    outptr[0] = vgetq_lane_u16(_v, 0);
    outptr[1 * outstride] = vgetq_lane_u16(_v, 1);
    outptr[2 * outstride] = vgetq_lane_u16(_v, 2);
    outptr[3 * outstride] = vgetq_lane_u16(_v, 3);
    outptr[4 * outstride] = vgetq_lane_u16(_v, 4);
    outptr[5 * outstride] = vgetq_lane_u16(_v, 5);
    outptr[6 * outstride] = vgetq_lane_u16(_v, 6);
    outptr[7 * outstride] = vgetq_lane_u16(_v, 7);
}

static NCNN_FORCEINLINE void permute_transpose4x1_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, size_t outstride)
{
    uint16x4_t _v = vld1_u16(ptr);
    outptr[0] = vget_lane_u16(_v, 0);
    outptr[1 * outstride] = vget_lane_u16(_v, 1);
    outptr[2 * outstride] = vget_lane_u16(_v, 2);
    outptr[3 * outstride] = vget_lane_u16(_v, 3);
}

static void permute_pack4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int cols)
{
    int j = 0;
    for (; j + 7 < cols; j += 8)
    {
        permute_transpose8x4_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 3 < cols; j += 4)
    {
        permute_transpose4x4_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 1 < cols; j += 2)
    {
        permute_transpose2x4_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j < cols; j++)
    {
        for (int k = 0; k < 4; k++)
            outptr[j * outstride + k] = ptr[k * stride + j];
    }
}

static void permute_pack8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int cols)
{
    int j = 0;
    for (; j + 7 < cols; j += 8)
    {
        permute_transpose8x8_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 3 < cols; j += 4)
    {
        permute_transpose4x8_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 1 < cols; j += 2)
    {
        permute_transpose2x8_stride_bf16s_fp16s(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j < cols; j++)
    {
        for (int k = 0; k < 8; k++)
            outptr[j * outstride + k] = ptr[k * stride + j];
    }
}

// unpacked matrix transpose, shared by 2d and channel/spatial permutations
// fixed input width; callers select the packing before traversing the rows
static void permute_unpack4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows)
{
    int i = 0;
    for (; i + 7 < rows; i += 8)
    {
        permute_transpose4x8_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose4x4_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose4x2_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose4x1_stride_bf16s_fp16s(ptr + i * stride, outptr + i, outstride);
    }
}

// fixed input width; callers select the packing before traversing the rows
static void permute_unpack8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows)
{
    int i = 0;
    for (; i + 7 < rows; i += 8)
    {
        permute_transpose8x8_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose8x4_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose8x2_stride_bf16s_fp16s(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose8x1_stride_bf16s_fp16s(ptr + i * stride, outptr + i, outstride);
    }
}

static void permute_transpose_pack1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    if (cols == 1)
    {
        if (stride == 1)
            memcpy(outptr, ptr, (size_t)rows * sizeof(unsigned short));
        else
            for (int i = 0; i < rows; i++)
                outptr[i] = ptr[i * stride];
        return;
    }
    if (rows == 1 && outstride == 1)
    {
        memcpy(outptr, ptr, (size_t)cols * sizeof(unsigned short));
        return;
    }

    int i = 0;
    for (; i + 7 < rows; i += 8)
    {
        int j = 0;
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose8x8_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x8_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose2x8_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j < cols; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                outptr[j * outstride + i + k] = ptr[(i + k) * stride + j];
            }
        }
    }
    for (; i + 3 < rows; i += 4)
    {
        int j = 0;
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose8x4_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x4_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose2x4_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j < cols; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                outptr[j * outstride + i + k] = ptr[(i + k) * stride + j];
            }
        }
    }
    for (; i + 1 < rows; i += 2)
    {
        int j = 0;
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose8x2_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x2_stride_bf16s_fp16s(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j < cols; j++)
        {
            outptr[j * outstride + i] = ptr[i * stride + j];
            outptr[j * outstride + i + 1] = ptr[(i + 1) * stride + j];
        }
    }
    for (; i < rows; i++)
    {
        int j = 0;
        for (; j + 7 < cols; j += 8)
        {
            permute_transpose8x1_stride_bf16s_fp16s(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x1_stride_bf16s_fp16s(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
        for (; j < cols; j++)
        {
            outptr[j * outstride + i] = ptr[i * stride + j];
        }
    }
}

// 2d: packed rows become packed output rows after transposing w and h
static void permute_transpose_pack1to4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 4)
    {
        for (int y = 0; y < rows; y++)
        {
            const unsigned short* p = ptr + y * stride + x;
            unsigned short* out = outptr + (x / 4) * outstride + y * 4;
            vst1_u16(out, vld1_u16(p));
        }
    }
}

static void permute_transpose_pack1to8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 8)
    {
        for (int y = 0; y < rows; y++)
        {
            const unsigned short* p = ptr + y * stride + x;
            unsigned short* out = outptr + (x / 8) * outstride + y * 8;
            vst1q_u16(out, vld1q_u16(p));
        }
    }
}

static void permute_transpose_pack4to1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x++)
        {
            const unsigned short* p = ptr + (y / 4) * stride + x * 4;
            unsigned short* out = outptr + x * outstride + y;
            vst1_u16(out, vld1_u16(p));
        }
    }
}

static void permute_transpose_pack4to4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const unsigned short* p = ptr + (y / 4) * stride + x * 4;
            unsigned short* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose4x4_bf16s_fp16s(p, out);
        }
    }
}

static void permute_transpose_pack4to4_tiled_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int nT)
{
    const int tile_size = 32;
    #pragma omp parallel for num_threads(nT)
    for (int x = 0; x < cols; x += 4)
    {
        const unsigned short* p = ptr + (size_t)x * 4;
        unsigned short* out = outptr + (size_t)(x / 4) * outstride;
        for (int y = 0; y < rows; y += tile_size * 4)
        {
            const int tile_rows = std::min(tile_size * 4, rows - y);
            permute_transpose_pack4to4_bf16s_fp16s(p + (size_t)(y / 4) * stride, stride, out + (size_t)y * 4, outstride, tile_rows, 4);
        }
    }
}

static void permute_transpose_pack4to8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 8)
        {
            const unsigned short* p = ptr + (y / 4) * stride + x * 4;
            unsigned short* out = outptr + (x / 8) * outstride + y * 8;
            permute_transpose4x8_bf16s_fp16s(p, out);
        }
    }
}

static void permute_transpose_pack8to1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x++)
        {
            const unsigned short* p = ptr + (y / 8) * stride + x * 8;
            unsigned short* out = outptr + x * outstride + y;
            uint16x8_t _v = vld1q_u16((const unsigned short*)p);
            vst1q_u16((unsigned short*)out, _v);
        }
    }
}

static void permute_transpose_pack8to4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const unsigned short* p = ptr + (y / 8) * stride + x * 8;
            unsigned short* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose8x4_bf16s_fp16s(p, out);
        }
    }
}

static void permute_transpose_pack8to4_tiled_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int nT)
{
    const int tile_size = 16;
    #pragma omp parallel for num_threads(nT)
    for (int x = 0; x < cols; x += 4)
    {
        const unsigned short* p = ptr + (size_t)x * 8;
        unsigned short* out = outptr + (size_t)(x / 4) * outstride;
        for (int y = 0; y < rows; y += tile_size * 8)
        {
            const int tile_rows = std::min(tile_size * 8, rows - y);
            permute_transpose_pack8to4_bf16s_fp16s(p + (size_t)(y / 8) * stride, stride, out + (size_t)y * 4, outstride, tile_rows, 4);
        }
    }
}

static void permute_transpose_pack8to8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 8)
    {
        for (int x = 0; x < cols; x += 8)
        {
            const unsigned short* p = ptr + (y / 8) * stride + x * 8;
            unsigned short* out = outptr + (x / 8) * outstride + y * 8;
            permute_transpose8x8_stride_bf16s_fp16s(p, 8, out, 8);
        }
    }
}

static void permute_transpose2d_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int elempack, int out_elempack)
{
    if (elempack == 1 && out_elempack == 1)
    {
        permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 1 && out_elempack == 4)
    {
        permute_transpose_pack1to4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 1 && out_elempack == 8)
    {
        permute_transpose_pack1to8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 4 && out_elempack == 1)
    {
        permute_transpose_pack4to1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 4 && out_elempack == 4)
    {
        permute_transpose_pack4to4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 4 && out_elempack == 8)
    {
        permute_transpose_pack4to8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 8 && out_elempack == 1)
    {
        permute_transpose_pack8to1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 8 && out_elempack == 4)
    {
        permute_transpose_pack8to4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 8 && out_elempack == 8)
    {
        permute_transpose_pack8to8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
}

// spatial transpose within one input channel group
// outcstep is used when unpacking
static NCNN_FORCEINLINE void permute_spatial2x2_pack4_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    uint16x8_t _a = vld1q_u16((const unsigned short*)ptr);
    uint16x8_t _b = vld1q_u16((const unsigned short*)(ptr + stride));
    vst1q_u16((unsigned short*)outptr, permute_zip1_u64(_a, _b));
    vst1q_u16((unsigned short*)(outptr + outstride), permute_zip2_u64(_a, _b));
}

static void permute_spatial_pack4_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    int i = 0;
    for (; i + 3 < rows; i += 4)
    {
        int j = 0;
        for (; j + 3 < cols; j += 4)
        {
            const unsigned short* p = ptr + i * stride + j * 4;
            unsigned short* out = outptr + j * outstride + i * 4;
            uint16x4_t _r00 = vld1_u16(p + 0 * stride + 0);
            uint16x4_t _r01 = vld1_u16(p + 0 * stride + 4);
            uint16x4_t _r02 = vld1_u16(p + 0 * stride + 8);
            uint16x4_t _r03 = vld1_u16(p + 0 * stride + 12);
            uint16x4_t _r10 = vld1_u16(p + 1 * stride + 0);
            uint16x4_t _r11 = vld1_u16(p + 1 * stride + 4);
            uint16x4_t _r12 = vld1_u16(p + 1 * stride + 8);
            uint16x4_t _r13 = vld1_u16(p + 1 * stride + 12);
            uint16x4_t _r20 = vld1_u16(p + 2 * stride + 0);
            uint16x4_t _r21 = vld1_u16(p + 2 * stride + 4);
            uint16x4_t _r22 = vld1_u16(p + 2 * stride + 8);
            uint16x4_t _r23 = vld1_u16(p + 2 * stride + 12);
            uint16x4_t _r30 = vld1_u16(p + 3 * stride + 0);
            uint16x4_t _r31 = vld1_u16(p + 3 * stride + 4);
            uint16x4_t _r32 = vld1_u16(p + 3 * stride + 8);
            uint16x4_t _r33 = vld1_u16(p + 3 * stride + 12);

            vst1_u16(out + 0 * outstride + 0, _r00);
            vst1_u16(out + 0 * outstride + 4, _r10);
            vst1_u16(out + 0 * outstride + 8, _r20);
            vst1_u16(out + 0 * outstride + 12, _r30);
            vst1_u16(out + 1 * outstride + 0, _r01);
            vst1_u16(out + 1 * outstride + 4, _r11);
            vst1_u16(out + 1 * outstride + 8, _r21);
            vst1_u16(out + 1 * outstride + 12, _r31);
            vst1_u16(out + 2 * outstride + 0, _r02);
            vst1_u16(out + 2 * outstride + 4, _r12);
            vst1_u16(out + 2 * outstride + 8, _r22);
            vst1_u16(out + 2 * outstride + 12, _r32);
            vst1_u16(out + 3 * outstride + 0, _r03);
            vst1_u16(out + 3 * outstride + 4, _r13);
            vst1_u16(out + 3 * outstride + 8, _r23);
            vst1_u16(out + 3 * outstride + 12, _r33);
        }
        for (; j < cols; j++)
        {
            const unsigned short* p = ptr + i * stride + j * 4;
            unsigned short* out = outptr + j * outstride + i * 4;
            vst1_u16(out + 0, vld1_u16(p + 0 * stride));
            vst1_u16(out + 4, vld1_u16(p + 1 * stride));
            vst1_u16(out + 8, vld1_u16(p + 2 * stride));
            vst1_u16(out + 12, vld1_u16(p + 3 * stride));
        }
    }
    for (; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            const unsigned short* p = ptr + i * stride + j * 4;
            unsigned short* out = outptr + j * outstride + i * 4;
            vst1_u16(out, vld1_u16(p));
        }
    }
}

static void permute_spatial_pack4to1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, size_t outcstep, int rows, int cols)
{
    for (int x = 0; x < cols; x++)
        permute_unpack4_stride_bf16s_fp16s(ptr + x * 4, stride, outptr + x * outstride, outcstep, rows);
}

static NCNN_FORCEINLINE void permute_spatial2x2_pack8_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride)
{
    permute_f32x8 _a = permute_load_f32x8((const float*)ptr);
    permute_f32x8 _b = permute_load_f32x8((const float*)(ptr + stride));
    permute_store_f32x8((float*)(outptr), permute_shuffle_f32x8_halves(_a, _b, 0x20));
    permute_store_f32x8((float*)(outptr + outstride), permute_shuffle_f32x8_halves(_a, _b, 0x31));
}

static void permute_spatial_pack8_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    int i = 0;
    for (; i + 3 < rows; i += 4)
    {
        int j = 0;
        for (; j + 3 < cols; j += 4)
        {
            const unsigned short* p = ptr + i * stride + j * 8;
            unsigned short* out = outptr + j * outstride + i * 8;
            uint16x8_t _r00 = vld1q_u16(p + 0 * stride + 0);
            uint16x8_t _r01 = vld1q_u16(p + 0 * stride + 8);
            uint16x8_t _r02 = vld1q_u16(p + 0 * stride + 16);
            uint16x8_t _r03 = vld1q_u16(p + 0 * stride + 24);
            uint16x8_t _r10 = vld1q_u16(p + 1 * stride + 0);
            uint16x8_t _r11 = vld1q_u16(p + 1 * stride + 8);
            uint16x8_t _r12 = vld1q_u16(p + 1 * stride + 16);
            uint16x8_t _r13 = vld1q_u16(p + 1 * stride + 24);
            uint16x8_t _r20 = vld1q_u16(p + 2 * stride + 0);
            uint16x8_t _r21 = vld1q_u16(p + 2 * stride + 8);
            uint16x8_t _r22 = vld1q_u16(p + 2 * stride + 16);
            uint16x8_t _r23 = vld1q_u16(p + 2 * stride + 24);
            uint16x8_t _r30 = vld1q_u16(p + 3 * stride + 0);
            uint16x8_t _r31 = vld1q_u16(p + 3 * stride + 8);
            uint16x8_t _r32 = vld1q_u16(p + 3 * stride + 16);
            uint16x8_t _r33 = vld1q_u16(p + 3 * stride + 24);

            vst1q_u16(out + 0 * outstride + 0, _r00);
            vst1q_u16(out + 0 * outstride + 8, _r10);
            vst1q_u16(out + 0 * outstride + 16, _r20);
            vst1q_u16(out + 0 * outstride + 24, _r30);
            vst1q_u16(out + 1 * outstride + 0, _r01);
            vst1q_u16(out + 1 * outstride + 8, _r11);
            vst1q_u16(out + 1 * outstride + 16, _r21);
            vst1q_u16(out + 1 * outstride + 24, _r31);
            vst1q_u16(out + 2 * outstride + 0, _r02);
            vst1q_u16(out + 2 * outstride + 8, _r12);
            vst1q_u16(out + 2 * outstride + 16, _r22);
            vst1q_u16(out + 2 * outstride + 24, _r32);
            vst1q_u16(out + 3 * outstride + 0, _r03);
            vst1q_u16(out + 3 * outstride + 8, _r13);
            vst1q_u16(out + 3 * outstride + 16, _r23);
            vst1q_u16(out + 3 * outstride + 24, _r33);
        }
        for (; j < cols; j++)
        {
            const unsigned short* p = ptr + i * stride + j * 8;
            unsigned short* out = outptr + j * outstride + i * 8;
            vst1q_u16(out + 0, vld1q_u16(p + 0 * stride));
            vst1q_u16(out + 8, vld1q_u16(p + 1 * stride));
            vst1q_u16(out + 16, vld1q_u16(p + 2 * stride));
            vst1q_u16(out + 24, vld1q_u16(p + 3 * stride));
        }
    }
    for (; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            const unsigned short* p = ptr + i * stride + j * 8;
            unsigned short* out = outptr + j * outstride + i * 8;
            vst1q_u16(out, vld1q_u16(p));
        }
    }
}

static void permute_spatial_pack8to1_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, size_t outcstep, int rows, int cols)
{
    for (int x = 0; x < cols; x++)
        permute_unpack8_stride_bf16s_fp16s(ptr + x * 8, stride, outptr + x * outstride, outcstep, rows);
}

static void permute_transpose_blocks_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int size);

static void permute_transpose_blocks_tiled_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int size, int tile_size, int nT)
{
    if (rows >= 256 && cols >= 256 && rows % 32 == 0 && cols % 32 == 0)
    {
        #pragma omp parallel for num_threads(nT)
        for (int i = 0; i < rows; i += tile_size)
        {
            const int tile_rows = std::min(tile_size, rows - i);
            for (int j = 0; j < cols; j += tile_size)
            {
                const int tile_cols = std::min(tile_size, cols - j);
                permute_transpose_blocks_bf16s_fp16s(ptr + (size_t)i * stride + (size_t)j * size, stride, outptr + (size_t)j * outstride + (size_t)i * size, outstride, tile_rows, tile_cols, size);
            }
        }
        return;
    }

    permute_transpose_blocks_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols, size);
}

static void permute_transpose_spatial_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, size_t outcstep, int rows, int cols, int elempack,
                                                  int out_elempack)
{
    if (elempack == out_elempack)
    {
        if (rows == 1 && outstride == (size_t)elempack)
        {
            memcpy(outptr, ptr, (size_t)cols * elempack * sizeof(unsigned short));
            return;
        }
        if (cols == 1 && stride == (size_t)elempack)
        {
            memcpy(outptr, ptr, (size_t)rows * elempack * sizeof(unsigned short));
            return;
        }
    }

    if (elempack == 1)
    {
        permute_transpose_blocks_tiled_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols, 1, 32, 1);
        return;
    }
    if (elempack == 4 && out_elempack == 4)
    {
        permute_transpose_blocks_tiled_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols, 4, 32, 1);
        return;
    }
    if (elempack == 4 && out_elempack == 1)
    {
        permute_spatial_pack4to1_bf16s_fp16s(ptr, stride, outptr, outstride, outcstep, rows, cols);
        return;
    }

    if (elempack == 8 && out_elempack == 8)
    {
        permute_transpose_blocks_tiled_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols, 8, 32, 1);
        return;
    }
    if (elempack == 8 && out_elempack == 1)
    {
        permute_spatial_pack8to1_bf16s_fp16s(ptr, stride, outptr, outstride, outcstep, rows, cols);
        return;
    }
}

static void permute_transpose_blocks2_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols)
{
    int j = 0;
    for (; j + 1 < cols; j += 2)
    {
        const unsigned short* p = ptr + (size_t)j * 2;
        unsigned short* out0 = outptr + j * outstride;
        unsigned short* out1 = out0 + outstride;
        int i = 0;
        for (; i + 1 < rows; i += 2)
        {
            uint32x2_t _a = vreinterpret_u32_u16(vld1_u16(p));
            uint32x2_t _b = vreinterpret_u32_u16(vld1_u16(p + stride));
            uint32x2x2_t _ab = vzip_u32(_a, _b);
            vst1_u16(out0, vreinterpret_u16_u32(_ab.val[0]));
            vst1_u16(out1, vreinterpret_u16_u32(_ab.val[1]));
            p += stride * 2;
            out0 += 4;
            out1 += 4;
        }
        for (; i < rows; i++)
        {
            memcpy(out0, p, 2 * sizeof(unsigned short));
            memcpy(out1, p + 2, 2 * sizeof(unsigned short));
            p += stride;
            out0 += 2;
            out1 += 2;
        }
    }
    for (; j < cols; j++)
    {
        const unsigned short* p = ptr + (size_t)j * 2;
        unsigned short* out = outptr + j * outstride;
        for (int i = 0; i < rows; i++)
        {
            memcpy(out, p, 2 * sizeof(unsigned short));
            p += stride;
            out += 2;
        }
    }
}

// transpose rows of contiguous blocks
// size is independent of elempack
// strides include padding; the block contents keep their original order
static void permute_transpose_blocks_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int size)
{
    if (rows == 1 && outstride == (size_t)size)
    {
        memcpy(outptr, ptr, (size_t)cols * size * sizeof(unsigned short));
        return;
    }
    if (cols == 1 && stride == (size_t)size)
    {
        memcpy(outptr, ptr, (size_t)rows * size * sizeof(unsigned short));
        return;
    }

    if (size == 1)
    {
        permute_transpose_pack1_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (size == 2)
    {
        permute_transpose_blocks2_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (size == 4)
    {
        permute_spatial_pack4_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (size == 8)
    {
        permute_spatial_pack8_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (size >= 2 && size < 4)
    {
        for (int j = 0; j < cols; j++)
        {
            const unsigned short* p = ptr + (size_t)j * size;
            unsigned short* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                int a;
                int b;
                memcpy(&a, p, 4);
                memcpy(&b, p + size - 2, 4);
                memcpy(out, &a, 4);
                memcpy(out + size - 2, &b, 4);
                p += stride;
                out += size;
            }
        }
        return;
    }
    if (size >= 4 && size < 8)
    {
        for (int j = 0; j < cols; j++)
        {
            const unsigned short* p = ptr + (size_t)j * size;
            unsigned short* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                uint16x8_t _a = permute_loadl_u16x8((const unsigned short*)p);
                uint16x8_t _b = permute_loadl_u16x8((const unsigned short*)(p + size - 4));
                permute_storel_u16x8((unsigned short*)out, _a);
                permute_storel_u16x8((unsigned short*)(out + size - 4), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
    if (size >= 8 && size < 16)
    {
        for (int j = 0; j < cols; j++)
        {
            const unsigned short* p = ptr + (size_t)j * size;
            unsigned short* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                uint16x8_t _a = vld1q_u16((const unsigned short*)p);
                uint16x8_t _b = vld1q_u16((const unsigned short*)(p + size - 8));
                vst1q_u16((unsigned short*)out, _a);
                vst1q_u16((unsigned short*)(out + size - 8), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
    if (size >= 16 && size < 32)
    {
        for (int j = 0; j < cols; j++)
        {
            const unsigned short* p = ptr + (size_t)j * size;
            unsigned short* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                permute_u16x16 _a = permute_load_u16x16((const unsigned short*)p);
                permute_u16x16 _b = permute_load_u16x16((const unsigned short*)(p + size - 16));
                permute_store_u16x16((unsigned short*)out, _a);
                permute_store_u16x16((unsigned short*)(out + size - 16), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
    for (int j = 0; j < cols; j++)
    {
        const unsigned short* p = ptr + (size_t)j * size;
        unsigned short* out = outptr + j * outstride;
        for (int i = 0; i < rows; i++)
        {
            memcpy(out, p, (size_t)size * sizeof(unsigned short));
            p += stride;
            out += size;
        }
    }
}

static void permute_unpack_spatial_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, size_t outcstep, int size, int elempack)
{
    if (elempack == 4)
    {
        permute_unpack4_stride_bf16s_fp16s(ptr, 4, outptr, outcstep, size);
        return;
    }

    if (elempack == 8)
    {
        permute_unpack8_stride_bf16s_fp16s(ptr, 8, outptr, outcstep, size);
        return;
    }
}

// exchange the input channel axis with one output channel group
// w is the remaining spatial extent; cstep includes input channel padding
// the exchanged axis is contiguous; the stride variant has contiguous spatial input
static void permute3d_pack1to4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * outcstep;
        for (int x = 0; x < w; x++)
        {
            vst1_u16(out, vld1_u16(p));
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack1to4_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * outcstep;
        permute_pack4_stride_bf16s_fp16s(p, hstep, out, outwstep, w);
    }
}

// the exchanged axis is contiguous; the stride variant has contiguous spatial input
static void permute3d_pack1to8_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * outcstep;
        for (int x = 0; x < w; x++)
        {
            uint16x8_t _v = vld1q_u16((const unsigned short*)p);
            vst1q_u16((unsigned short*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack1to8_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * outcstep;
        permute_pack8_stride_bf16s_fp16s(p, hstep, out, outwstep, w);
    }
}

// output channels are contiguous; the stride variant has contiguous spatial output
static void permute3d_pack4to1_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4;
        for (int x = 0; x < w; x++)
        {
            vst1_u16(out, vld1_u16(p));
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to1_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * outcstep;
        permute_unpack4_stride_bf16s_fp16s(p, wstep, out, outcstep, w);
    }
}

static void permute3d_pack4to4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * 4;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x4_bf16s_fp16s(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to4_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x4_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to8_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * 8;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x8_bf16s_fp16s(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to8_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 4 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x8_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

// output channels are contiguous; the stride variant has contiguous spatial output
static void permute3d_pack8to1_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8;
        for (int x = 0; x < w; x++)
        {
            uint16x8_t _v = vld1q_u16((const unsigned short*)p);
            vst1q_u16((unsigned short*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack8to1_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8 * outcstep;
        permute_unpack8_stride_bf16s_fp16s(p, wstep, out, outcstep, w);
    }
}

static void permute3d_pack8to4_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8 * 4;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x4_bf16s_fp16s(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack8to4_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x4_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack8to8_stride_bf16s_fp16s(const unsigned short* ptr, unsigned short* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const unsigned short* p = ptr + c * cstep;
        unsigned short* out = outptr + c * 8 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose8x8_stride_bf16s_fp16s(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

// the exchanged input axis is contiguous
static void permute_pack_channels_bf16s_fp16s(const Mat& bottom_blob, const Mat& top_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, size_t wstep, size_t outwstep,
                                              size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (out_elempack == 4)
    {
        permute3d_pack1to4_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outwstep, outcstep);
        return;
    }
    if (out_elempack == 8)
    {
        permute3d_pack1to8_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outwstep, outcstep);
        return;
    }
}

// the remaining input spatial axis is contiguous
static void permute_pack_channels_stride_bf16s_fp16s(const Mat& bottom_blob, const Mat& top_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, size_t hstep,
                                                     size_t outwstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (out_elempack == 4)
    {
        permute3d_pack1to4_stride_bf16s_fp16s(ptr, outptr, size, channels, hstep, cstep, outwstep, outcstep);
        return;
    }
    if (out_elempack == 8)
    {
        permute3d_pack1to8_stride_bf16s_fp16s(ptr, outptr, size, channels, hstep, cstep, outwstep, outcstep);
        return;
    }
}

// the output channel axis is contiguous
static void permute_unpack_channels_bf16s_fp16s(const Mat& bottom_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, size_t wstep, size_t outwstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int elempack = bottom_blob.elempack;

    if (elempack == 4)
    {
        permute3d_pack4to1_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outwstep);
        return;
    }
    if (elempack == 8)
    {
        permute3d_pack8to1_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outwstep);
        return;
    }
}

// the remaining output spatial axis is contiguous
static void permute_unpack_channels_stride_bf16s_fp16s(const Mat& bottom_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, size_t wstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int elempack = bottom_blob.elempack;

    if (elempack == 4)
    {
        permute3d_pack4to1_stride_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outcstep);
        return;
    }
    if (elempack == 8)
    {
        permute3d_pack8to1_stride_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outcstep);
        return;
    }
}

// both channel axes are packed
static void permute_channels_spatial_input_stride_bf16s_fp16s(const Mat& bottom_blob, const Mat& top_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, size_t hstep,
                                                              size_t outwstep, size_t outcstep)
{
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t wstep = elempack;

    if (elempack == 4 && out_elempack == 4)
    {
        if (hstep == 4 && outcstep == 4)
        {
            permute3d_pack4to4_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack4to4_stride_bf16s_fp16s(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
    if (elempack == 4 && out_elempack == 8)
    {
        if (hstep == 4 && outcstep == 8)
        {
            permute3d_pack4to8_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack4to8_stride_bf16s_fp16s(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
    if (elempack == 8 && out_elempack == 4)
    {
        if (hstep == 8 && outcstep == 4)
        {
            permute3d_pack8to4_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack8to4_stride_bf16s_fp16s(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
    if (elempack == 8 && out_elempack == 8)
    {
        permute3d_pack8to8_stride_bf16s_fp16s(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
}

static void permute_channels_axis_input_stride_bf16s_fp16s(const Mat& bottom_blob, const Mat& top_blob, const unsigned short* ptr, unsigned short* outptr, int size, int channels, size_t wstep,
                                                           size_t outwstep, size_t outcstep)
{
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t hstep = elempack;

    if (elempack == 4 && out_elempack == 4)
    {
        if (hstep == 4 && outcstep == 4)
        {
            permute3d_pack4to4_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack4to4_stride_bf16s_fp16s(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
    if (elempack == 4 && out_elempack == 8)
    {
        if (hstep == 4 && outcstep == 8)
        {
            permute3d_pack4to8_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack4to8_stride_bf16s_fp16s(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
    if (elempack == 8 && out_elempack == 4)
    {
        if (hstep == 8 && outcstep == 4)
        {
            permute3d_pack8to4_bf16s_fp16s(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack8to4_stride_bf16s_fp16s(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
    if (elempack == 8 && out_elempack == 8)
    {
        permute3d_pack8to8_stride_bf16s_fp16s(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        return;
    }
}

static void permute_transpose_matrix_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int size, int nT)
{
    if (size == 1 && rows >= 256 && cols >= 256 && rows % 32 == 0 && cols % 32 == 0)
    {
        permute_transpose_blocks_tiled_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols, 1, rows >= 512 && cols >= 512 ? 16 : 32, nT);
        return;
    }

    // scalar matrices keep the full register transpose width
    const int step = size == 1 ? 8 : 1;
    #pragma omp parallel for num_threads(nT)
    for (int j = 0; j < cols; j += step)
    {
        permute_transpose_blocks_bf16s_fp16s(ptr + (size_t)j * size, stride, outptr + j * outstride, outstride, rows, std::min(step, cols - j), size);
    }
}

// transpose independent matrices
static void permute_transpose_matrices_stride_bf16s_fp16s(const unsigned short* ptr, size_t stride, unsigned short* outptr, size_t outstride, int rows, int cols, int size, int planes, size_t step,
                                                          size_t outstep, int nT)
{
    if (planes == 1)
    {
        permute_transpose_matrix_bf16s_fp16s(ptr, stride, outptr, outstride, rows, cols, size, nT);
        return;
    }

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < planes; q++)
    {
        permute_transpose_blocks_bf16s_fp16s(ptr + q * step, stride, outptr + q * outstep, outstride, rows, cols, size);
    }
}

// transpose spatial planes within each input channel group
static void permute_transpose_spatial_planes_stride_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int rows, int cols, size_t stride, size_t outstride, int planes, size_t step, size_t outstep,
                                                                int nT)
{
    (void)nT;

    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        unsigned short* outptr = top_blob.channel(q * elempack / out_elempack);
        for (int z = 0; z < planes; z++)
        {
            permute_transpose_spatial_bf16s_fp16s(ptr + z * step, stride, outptr + z * outstep, outstride, top_blob.cstep, rows, cols, elempack, out_elempack);
        }
    }
}

// transpose h and w while preserving the other two slice axes
static void permute_transpose_hw_pack1_stride_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outcstep, size_t outdstep, int nT)
{
    (void)nT;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const size_t stride = (size_t)w;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        unsigned short* outptr = (unsigned short*)top_blob + q * outcstep;
        for (int z = 0; z < d; z++)
        {
            permute_transpose_pack1_bf16s_fp16s(ptr + z * (size_t)w * h, stride, outptr + z * outdstep, outstride, h, w);
        }
    }
}

// transpose d and w while preserving the other two slice axes
static void permute_transpose_dw_pack1_stride_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outcstep, size_t outhstep, int nT)
{
    (void)nT;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const size_t stride = (size_t)w * h;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        unsigned short* outptr = (unsigned short*)top_blob + q * outcstep;
        for (int y = 0; y < h; y++)
        {
            permute_transpose_pack1_bf16s_fp16s(ptr + y * (size_t)w, stride, outptr + y * outhstep, outstride, d, w);
        }
    }
}

// transpose c and w while preserving the other two slice axes
static void permute_transpose_cw_pack1_stride_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outdstep, size_t outhstep, int nT)
{
    (void)nT;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const size_t stride = bottom_blob.cstep;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        const unsigned short* ptr = (const unsigned short*)bottom_blob + z * (size_t)w * h;
        unsigned short* outptr = (unsigned short*)top_blob + z * outdstep;
        for (int y = 0; y < h; y++)
        {
            permute_transpose_pack1_bf16s_fp16s(ptr + y * (size_t)w, stride, outptr + y * outhstep, outstride, channels, w);
        }
    }
}

static void permute2d_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        permute_transpose_matrix_bf16s_fp16s(bottom_blob, w, top_blob, h, h, w, 1, nT);
        return;
    }

    if (elempack == 4 && out_elempack == 4 && w >= 64 && h >= 64)
    {
        permute_transpose_pack4to4_tiled_bf16s_fp16s((const unsigned short*)bottom_blob, (size_t)w * 4, (unsigned short*)top_blob, (size_t)top_blob.w * 4, h * 4, w, nT);
        return;
    }

    if (elempack == 8 && out_elempack == 4 && w >= 64 && h >= 64)
    {
        permute_transpose_pack8to4_tiled_bf16s_fp16s((const unsigned short*)bottom_blob, (size_t)w * 8, (unsigned short*)top_blob, (size_t)top_blob.w * 4, h * 8, w, nT);
        return;
    }

    #pragma omp parallel for num_threads(nT)
    for (int x = 0; x < w; x += out_elempack)
    {
        const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)x * elempack;
        unsigned short* outptr = top_blob.row<unsigned short>(x / out_elempack);
        permute_transpose2d_bf16s_fp16s(ptr, (size_t)w * elempack, outptr, (size_t)top_blob.w * out_elempack, h * elempack, out_elempack, elempack, out_elempack);
    }
}

static void permute3d_hwc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * elempack;
    const size_t outstride = (size_t)h * out_elempack;
    const size_t step = 0;
    const size_t outstep = 0;
    permute_transpose_spatial_planes_stride_bf16s_fp16s(bottom_blob, top_blob, h, w, stride, outstride, 1, step, outstep, nT);
}

static void permute3d_wch_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        if (w < 64)
        {
            // short blocks include the w == 1 matrix-transpose case
            permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, top_blob.cstep, channels, h, w, nT);
            return;
        }

        #pragma omp parallel for num_threads(nT)
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + q * (size_t)w;
            unsigned short* outptr = top_blob.channel(q);
            for (int y = 0; y < top_blob.h; y++)
            {
                memcpy(outptr + (size_t)y * w, ptr + y * bottom_blob.cstep, (size_t)w * sizeof(unsigned short));
            }
        }
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)q * out_elempack * hstep;
        unsigned short* outptr = (unsigned short*)top_blob.channel(q);
        // exchange c and h, keeping w as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
        else
            permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
    }
}

static void permute3d_cwh_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        // w and h stay adjacent in the output; retain input channel padding
        if (top_blob.cstep == (size_t)w * channels && (size_t)w * h <= INT_MAX)
        {
            permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, channels, channels, w * h, 1, nT);
            return;
        }

        const size_t stride = bottom_blob.cstep;
        const size_t outstride = (size_t)top_blob.w;
        const size_t step = (size_t)w;
        const size_t outstep = top_blob.cstep;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, channels, w, 1, h, step, outstep, nT);
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)q * out_elempack * hstep;
        unsigned short* outptr = (unsigned short*)top_blob.channel(q);
        // exchange c and h, keeping w as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels_bf16s_fp16s(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
        else
            permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
    }
}

static void permute3d_hcw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        // c and h stay adjacent in the input; retain output channel padding
        if (bottom_blob.cstep == (size_t)w * h && (size_t)channels * h <= INT_MAX)
        {
            permute_transpose_matrix_bf16s_fp16s(bottom_blob, w, top_blob, top_blob.cstep, channels * h, w, 1, nT);
            return;
        }

        const size_t stride = (size_t)w;
        const size_t outstride = top_blob.cstep;
        const size_t step = bottom_blob.cstep;
        const size_t outstep = (size_t)top_blob.w;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, h, w, 1, channels, step, outstep, nT);
        return;
    }

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)q * out_elempack * hstep;
        unsigned short* outptr = (unsigned short*)top_blob.channel(q);
        // exchange c and w, keeping h as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
        else
            permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
    }
}

static void permute3d_chw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        // with w or h removed, the remaining operation is one c/spatial transpose
        if (w == 1 || h == 1)
        {
            permute_transpose_matrix_bf16s_fp16s(bottom_blob, bottom_blob.cstep, top_blob, w == 1 ? (size_t)channels : top_blob.cstep, channels, w * h, 1, nT);
            return;
        }

        if (channels == 1)
        {
            permute_transpose_matrix_bf16s_fp16s(bottom_blob, w, top_blob, top_blob.cstep, h, w, 1, nT);
            return;
        }

        const size_t stride = bottom_blob.cstep;
        const size_t outstride = top_blob.cstep;
        const size_t step = (size_t)w;
        const size_t outstep = (size_t)top_blob.w;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, channels, w, 1, h, step, outstep, nT);
        return;
    }

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)q * out_elempack * hstep;
        unsigned short* outptr = (unsigned short*)top_blob.channel(q);
        // exchange c and w, keeping h as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels_bf16s_fp16s(bottom_blob, ptr, outptr, h, channels, wstep, outwstep);
        else
            permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
    }
}

static void permute4d_hwdc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * elempack;
    const size_t outstride = (size_t)h * out_elempack;
    const size_t step = (size_t)w * h * elempack;
    const size_t outstep = (size_t)w * h * out_elempack;
    permute_transpose_spatial_planes_stride_bf16s_fp16s(bottom_blob, top_blob, h, w, stride, outstride, d, step, outstep, nT);
}

static void permute4d_wdhc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == out_elempack)
    {
        const size_t stride = (size_t)w * h * elempack;
        const size_t outstride = (size_t)w * d * elempack;
        const size_t step = bottom_blob.cstep * elempack;
        const size_t outstep = top_blob.cstep * elempack;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, d, h, w * elempack, channels, step, outstep, nT);
        return;
    }

    const size_t hstep = (size_t)w * elempack;
    const size_t dstep = (size_t)w * h * elempack;
    const size_t outhstep = (size_t)w * d * out_elempack;
    const size_t outdstep = (size_t)w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        unsigned short* outptr = top_blob.channel(q * elempack / out_elempack);
        for (int y = 0; y < h; y++)
        {
            const unsigned short* p = ptr + y * hstep;
            unsigned short* out = outptr + y * outhstep;
            for (int z = 0; z < d; z++)
            {
                permute_unpack_spatial_bf16s_fp16s(p + z * dstep, out + z * outdstep, top_blob.cstep, w, elempack);
            }
        }
    }
}

static void permute4d_dwhc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * h * elempack;
    const size_t outstride = (size_t)d * out_elempack;
    const size_t step = (size_t)w * elempack;
    const size_t outstep = (size_t)w * d * out_elempack;
    permute_transpose_spatial_planes_stride_bf16s_fp16s(bottom_blob, top_blob, d, w, stride, outstride, h, step, outstep, nT);
}

static void permute4d_hdwc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * elempack;
    const size_t outstride = (size_t)h * d * out_elempack;
    const size_t step = (size_t)w * h * elempack;
    const size_t outstep = (size_t)h * out_elempack;
    permute_transpose_spatial_planes_stride_bf16s_fp16s(bottom_blob, top_blob, h, w, stride, outstride, d, step, outstep, nT);
}

static void permute4d_dhwc_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * h * elempack;
    const size_t outstride = (size_t)h * d * out_elempack;
    const size_t step = (size_t)w * elempack;
    const size_t outstep = (size_t)d * out_elempack;
    permute_transpose_spatial_planes_stride_bf16s_fp16s(bottom_blob, top_blob, d, w, stride, outstride, h, step, outstep, nT);
}

static void permute4d_whcd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    // w and h stay adjacent on both sides
    Mat bottom_blob_3d(w * h, d, channels, bottom_blob.data, elemsize, elempack);
    bottom_blob_3d.cstep = bottom_blob.cstep;
    Mat top_blob_3d(w * h, channels * elempack, top_blob.c, top_blob.data, out_elemsize, out_elempack);
    top_blob_3d.cstep = top_blob.cstep;
    permute3d_wch_bf16s_fp16s(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_hwcd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w;
        const size_t outcstep = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = top_blob.cstep;
        permute_transpose_hw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outdstep, nT);
        return;
    }

    const int channels = bottom_blob.c;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * elempack;
        const size_t xstep = (size_t)elempack;
        const size_t outxstep = (size_t)top_blob.w * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int x = 0; x < w; x++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                unsigned short* outptr = (unsigned short*)top_blob.channel(q) + x * outxstep;
                // exchange c and d, keeping h as the inner spatial axis
                permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int y = 0; y < h; y++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_wchd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t stride = bottom_blob.cstep;
        const size_t outstride = (size_t)channels * w;
        const size_t step = (size_t)w * h;
        const size_t outstep = top_blob.cstep;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, channels, h, w, d, step, outstep, nT);
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)top_blob.w * top_blob.h * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int y = 0; y < h; y++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
            else
                permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cwhd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    // w and h stay adjacent on both sides
    Mat bottom_blob_3d(w * h, d, channels, bottom_blob.data, elemsize, elempack);
    bottom_blob_3d.cstep = bottom_blob.cstep;
    Mat top_blob_3d(channels * elempack, w * h, top_blob.c, top_blob.data, out_elemsize, out_elempack);
    top_blob_3d.cstep = top_blob.cstep;
    permute3d_cwh_bf16s_fp16s(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_hcwd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outdstep = top_blob.cstep;
        permute_transpose_hw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outdstep, nT);
        return;
    }

    const int channels = bottom_blob.c;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * elempack;
        const size_t xstep = (size_t)elempack;
        const size_t outxstep = (size_t)top_blob.w * top_blob.h * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int x = 0; x < w; x++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                unsigned short* outptr = (unsigned short*)top_blob.channel(q) + x * outxstep;
                // exchange c and d, keeping h as the inner spatial axis
                permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int y = 0; y < h; y++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_chwd_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = top_blob.cstep;
        const size_t outhstep = (size_t)top_blob.w;
        permute_transpose_cw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int channels = bottom_blob.c;
    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * h * elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t ystep = (size_t)w * elempack;
    const size_t outystep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int y = 0; y < h; y++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_bf16s_fp16s(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
            else
                permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_wdch_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t stride = (size_t)w * h;
        const size_t outstride = top_blob.cstep;
        const size_t step = bottom_blob.cstep;
        const size_t outstep = (size_t)d * w;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, d, h, w, channels, step, outstep, nT);
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
            else
                permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_dwch_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w;
        const size_t outcstep = (size_t)top_blob.w * top_blob.h;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_dw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
        return;
    }

    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t hstep = (size_t)w * elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * h * elempack;
        const size_t xstep = (size_t)elempack;
        const size_t outxstep = (size_t)top_blob.w * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int x = 0; x < w; x++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                unsigned short* outptr = (unsigned short*)top_blob.channel(q) + x * outxstep;
                // exchange c and h, keeping d as the inner spatial axis
                permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_wcdh_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t stride = bottom_blob.cstep;
        const size_t outstride = top_blob.cstep;
        const size_t step = (size_t)w * h;
        const size_t outstep = (size_t)channels * w;
        permute_transpose_matrices_stride_bf16s_fp16s(bottom_blob, stride, top_blob, outstride, channels, h, w, d, step, outstep, nT);
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
            else
                permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cwdh_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w;
        const size_t outdstep = (size_t)top_blob.w * top_blob.h;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_cw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_bf16s_fp16s(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
            else
                permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_dcwh_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_dw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
        return;
    }

    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t hstep = (size_t)w * elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * h * elempack;
        const size_t xstep = (size_t)elempack;
        const size_t outxstep = (size_t)top_blob.w * top_blob.h * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int x = 0; x < w; x++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                unsigned short* outptr = (unsigned short*)top_blob.channel(q) + x * outxstep;
                // exchange c and h, keeping d as the inner spatial axis
                permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cdwh_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = (size_t)top_blob.w;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_cw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_bf16s_fp16s(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
            else
                permute_channels_spatial_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_hdcw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    // h and d stay adjacent on both sides
    Mat bottom_blob_3d(w, h * d, channels, bottom_blob.data, elemsize, elempack);
    bottom_blob_3d.cstep = bottom_blob.cstep;
    Mat top_blob_3d(h * d, channels * elempack, top_blob.c, top_blob.data, out_elemsize, out_elempack);
    top_blob_3d.cstep = top_blob.cstep;
    permute3d_hcw_bf16s_fp16s(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_dhcw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outcstep = (size_t)top_blob.w * top_blob.h;
        const size_t outhstep = (size_t)top_blob.w;
        permute_transpose_dw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int channels = bottom_blob.c;
    const size_t hstep = elempack;
    const size_t outcstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * h * elempack;
        const size_t outwstep = out_elempack;
        const size_t ystep = (size_t)w * elempack;
        const size_t outystep = (size_t)top_blob.w * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int y = 0; y < h; y++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
                unsigned short* outptr = (unsigned short*)top_blob.channel(q) + y * outystep;
                // exchange c and w, keeping d as the inner spatial axis
                if (elempack == 1)
                    permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, d, channels, wstep, outwstep, outcstep);
                else
                    permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t wstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else
                permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}

static void permute4d_hcdw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outdstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_hw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outdstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * top_blob.h * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
            else
                permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}

static void permute4d_chdw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    // h and d stay adjacent on both sides
    Mat bottom_blob_3d(w, h * d, channels, bottom_blob.data, elemsize, elempack);
    bottom_blob_3d.cstep = bottom_blob.cstep;
    Mat top_blob_3d(channels * elempack, h * d, top_blob.c, top_blob.data, out_elemsize, out_elempack);
    top_blob_3d.cstep = top_blob.cstep;
    permute3d_chw_bf16s_fp16s(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_dchw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outhstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_dw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int channels = bottom_blob.c;
    const size_t hstep = elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    if (out_elempack == 1)
    {
        const size_t wstep = (size_t)w * h * elempack;
        const size_t outwstep = out_elempack;
        const size_t ystep = (size_t)w * elempack;
        const size_t outystep = (size_t)top_blob.w * top_blob.h * out_elempack;

        #pragma omp parallel for num_threads(nT)
        for (int y = 0; y < h; y++)
        {
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
                unsigned short* outptr = (unsigned short*)top_blob.channel(q) + y * outystep;
                // exchange c and w, keeping d as the inner spatial axis
                if (elempack == 1)
                    permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, d, channels, wstep, outwstep, outcstep);
                else
                    permute_unpack_channels_stride_bf16s_fp16s(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
            }
        }
        return;
    }

    const size_t wstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else
                permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cdhw_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outdstep = (size_t)top_blob.w;
        const size_t outhstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_cw_pack1_stride_bf16s_fp16s(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
        return;
    }

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = (size_t)top_blob.w * top_blob.h * out_elempack;
    const size_t outcstep = out_elempack;
    const size_t zstep = (size_t)w * h * elempack;
    const size_t outzstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int z = 0; z < d; z++)
    {
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            unsigned short* outptr = (unsigned short*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_bf16s_fp16s(bottom_blob, ptr, outptr, h, channels, wstep, outwstep);
            else
                permute_channels_axis_input_stride_bf16s_fp16s(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}
