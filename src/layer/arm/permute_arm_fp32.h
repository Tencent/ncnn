// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

// full register tiles have no size or packing branches

// tile names use input columns x rows
// contiguous tiles take only pointers; stride variants use scalar-element strides

static NCNN_FORCEINLINE void permute_transpose2x4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    float32x4_t _r0 = permute_load_f32x2(ptr);
    float32x4_t _r1 = permute_load_f32x2(ptr + stride);
    float32x4_t _r2 = permute_load_f32x2(ptr + 2 * stride);
    float32x4_t _r3 = permute_load_f32x2(ptr + 3 * stride);
    float32x4_t _t0 = permute_movelh_f32(_r0, _r1);
    float32x4_t _t1 = permute_movelh_f32(_r2, _r3);
    float32x4_t _a0 = vuzp1q_f32(_t0, _t1);
    float32x4_t _b0 = vuzp2q_f32(_t0, _t1);
    vst1q_f32(outptr, _a0);
    vst1q_f32(outptr + outstride, _b0);
}

static NCNN_FORCEINLINE void permute_transpose4x4(const float* ptr, float* outptr)
{
    float32x4_t _r0 = vld1q_f32(ptr);
    float32x4_t _r1 = vld1q_f32(ptr + 4);
    float32x4_t _r2 = vld1q_f32(ptr + 8);
    float32x4_t _r3 = vld1q_f32(ptr + 12);
    transpose4x4_ps(_r0, _r1, _r2, _r3);
    vst1q_f32(outptr, _r0);
    vst1q_f32(outptr + 4, _r1);
    vst1q_f32(outptr + 8, _r2);
    vst1q_f32(outptr + 12, _r3);
}

static NCNN_FORCEINLINE void permute_transpose4x4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    float32x4_t _r0 = vld1q_f32(ptr);
    float32x4_t _r1 = vld1q_f32(ptr + stride);
    float32x4_t _r2 = vld1q_f32(ptr + 2 * stride);
    float32x4_t _r3 = vld1q_f32(ptr + 3 * stride);
    transpose4x4_ps(_r0, _r1, _r2, _r3);
    vst1q_f32(outptr, _r0);
    vst1q_f32(outptr + outstride, _r1);
    vst1q_f32(outptr + 2 * outstride, _r2);
    vst1q_f32(outptr + 3 * outstride, _r3);
}

// final two or one rows: load complete vectors and write only the valid output lanes

static NCNN_FORCEINLINE void permute_transpose4x2_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    float32x4_t _r0 = vld1q_f32(ptr);
    float32x4_t _r1 = vld1q_f32(ptr + stride);
    float32x4_t _t0 = vzip1q_f32(_r0, _r1);
    float32x4_t _t1 = vzip2q_f32(_r0, _r1);
    float32x4_t _v0 = _t0;
    vst1_f32(outptr, vget_low_f32(_v0));
    vst1_f32(outptr + outstride, vget_high_f32(_v0));
    float32x4_t _v1 = _t1;
    vst1_f32(outptr + 2 * outstride, vget_low_f32(_v1));
    vst1_f32(outptr + 3 * outstride, vget_high_f32(_v1));
}

static NCNN_FORCEINLINE void permute_transpose4x1_stride(const float* ptr, float* outptr, size_t outstride)
{
    memcpy(outptr, ptr, sizeof(float));
    memcpy(outptr + outstride, ptr + 1, sizeof(float));
    memcpy(outptr + 2 * outstride, ptr + 2, sizeof(float));
    memcpy(outptr + 3 * outstride, ptr + 3, sizeof(float));
}

static void permute_pack4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int cols)
{
    int j = 0;
    for (; j + 3 < cols; j += 4)
    {
        permute_transpose4x4_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j + 1 < cols; j += 2)
    {
        permute_transpose2x4_stride(ptr + j, stride, outptr + j * outstride, outstride);
    }
    for (; j < cols; j++)
    {
        for (int k = 0; k < 4; k++)
            outptr[j * outstride + k] = ptr[k * stride + j];
    }
}

// unpacked matrix transpose, shared by 2d and channel/spatial permutations
// fixed input width; callers select the packing before traversing the rows
static void permute_unpack4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows)
{
    int i = 0;
    for (; i + 3 < rows; i += 4)
    {
        permute_transpose4x4_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i + 1 < rows; i += 2)
    {
        permute_transpose4x2_stride(ptr + i * stride, stride, outptr + i, outstride);
    }
    for (; i < rows; i++)
    {
        permute_transpose4x1_stride(ptr + i * stride, outptr + i, outstride);
    }
}

static void permute_transpose_pack1(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    if (cols == 1)
    {
        if (stride == 1)
            memcpy(outptr, ptr, (size_t)rows * sizeof(float));
        else
            for (int i = 0; i < rows; i++)
                outptr[i] = ptr[i * stride];
        return;
    }
    if (rows == 1 && outstride == 1)
    {
        memcpy(outptr, ptr, (size_t)cols * sizeof(float));
        return;
    }

    int i = 0;
    for (; i + 3 < rows; i += 4)
    {
        int j = 0;
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x4_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
        }
        for (; j + 1 < cols; j += 2)
        {
            permute_transpose2x4_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
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
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x2_stride(ptr + i * stride + j, stride, outptr + j * outstride + i, outstride);
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
        for (; j + 3 < cols; j += 4)
        {
            permute_transpose4x1_stride(ptr + i * stride + j, outptr + j * outstride + i, outstride);
        }
        for (; j < cols; j++)
        {
            outptr[j * outstride + i] = ptr[i * stride + j];
        }
    }
}

// 2d: packed rows become packed output rows after transposing w and h
static void permute_transpose_pack1to4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int x = 0; x < cols; x += 4)
    {
        for (int y = 0; y < rows; y++)
        {
            const float* p = ptr + y * stride + x;
            float* out = outptr + (x / 4) * outstride + y * 4;
            uint32x4_t _v = vld1q_u32((const unsigned int*)p);
            vst1q_u32((unsigned int*)out, _v);
        }
    }
}

static void permute_transpose_pack4to1(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x++)
        {
            const float* p = ptr + (y / 4) * stride + x * 4;
            float* out = outptr + x * outstride + y;
            uint32x4_t _v = vld1q_u32((const unsigned int*)p);
            vst1q_u32((unsigned int*)out, _v);
        }
    }
}

static void permute_transpose_pack4to4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    for (int y = 0; y < rows; y += 4)
    {
        for (int x = 0; x < cols; x += 4)
        {
            const float* p = ptr + (y / 4) * stride + x * 4;
            float* out = outptr + (x / 4) * outstride + y * 4;
            permute_transpose4x4(p, out);
        }
    }
}

static void permute_transpose_pack4to4_tiled(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int nT)
{
    const int tile_size = 32;
    #pragma omp parallel for num_threads(nT)
    for (int x = 0; x < cols; x += 4)
    {
        const float* p = ptr + (size_t)x * 4;
        float* out = outptr + (size_t)(x / 4) * outstride;
        for (int y = 0; y < rows; y += tile_size * 4)
        {
            const int tile_rows = std::min(tile_size * 4, rows - y);
            permute_transpose_pack4to4(p + (size_t)(y / 4) * stride, stride, out + (size_t)y * 4, outstride, tile_rows, 4);
        }
    }
}

static void permute_transpose2d(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int elempack, int out_elempack)
{
    if (elempack == 1 && out_elempack == 1)
    {
        permute_transpose_pack1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 1 && out_elempack == 4)
    {
        permute_transpose_pack1to4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 4 && out_elempack == 1)
    {
        permute_transpose_pack4to1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }

    if (elempack == 4 && out_elempack == 4)
    {
        permute_transpose_pack4to4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
}

// spatial transpose within one input channel group
// outcstep is used when unpacking
static NCNN_FORCEINLINE void permute_spatial2x2_pack4_stride(const float* ptr, size_t stride, float* outptr, size_t outstride)
{
    uint32x4_t _v0 = vld1q_u32((const unsigned int*)ptr);
    vst1q_u32((unsigned int*)outptr, _v0);
    uint32x4_t _v1 = vld1q_u32((const unsigned int*)(ptr + stride));
    vst1q_u32((unsigned int*)(outptr + 4), _v1);
    uint32x4_t _v2 = vld1q_u32((const unsigned int*)(ptr + 4));
    vst1q_u32((unsigned int*)(outptr + outstride), _v2);
    uint32x4_t _v3 = vld1q_u32((const unsigned int*)(ptr + stride + 4));
    vst1q_u32((unsigned int*)(outptr + outstride + 4), _v3);
}

static void permute_spatial_pack4(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    int i = 0;
    for (; i + 3 < rows; i += 4)
    {
        int j = 0;
        for (; j + 3 < cols; j += 4)
        {
            const float* p = ptr + i * stride + j * 4;
            float* out = outptr + j * outstride + i * 4;

            float32x4_t _r00 = vld1q_f32(p + 0 * stride + 0);
            float32x4_t _r01 = vld1q_f32(p + 0 * stride + 4);
            float32x4_t _r02 = vld1q_f32(p + 0 * stride + 8);
            float32x4_t _r03 = vld1q_f32(p + 0 * stride + 12);
            float32x4_t _r10 = vld1q_f32(p + 1 * stride + 0);
            float32x4_t _r11 = vld1q_f32(p + 1 * stride + 4);
            float32x4_t _r12 = vld1q_f32(p + 1 * stride + 8);
            float32x4_t _r13 = vld1q_f32(p + 1 * stride + 12);
            float32x4_t _r20 = vld1q_f32(p + 2 * stride + 0);
            float32x4_t _r21 = vld1q_f32(p + 2 * stride + 4);
            float32x4_t _r22 = vld1q_f32(p + 2 * stride + 8);
            float32x4_t _r23 = vld1q_f32(p + 2 * stride + 12);
            float32x4_t _r30 = vld1q_f32(p + 3 * stride + 0);
            float32x4_t _r31 = vld1q_f32(p + 3 * stride + 4);
            float32x4_t _r32 = vld1q_f32(p + 3 * stride + 8);
            float32x4_t _r33 = vld1q_f32(p + 3 * stride + 12);

            vst1q_f32(out + 0 * outstride + 0, _r00);
            vst1q_f32(out + 0 * outstride + 4, _r10);
            vst1q_f32(out + 0 * outstride + 8, _r20);
            vst1q_f32(out + 0 * outstride + 12, _r30);
            vst1q_f32(out + 1 * outstride + 0, _r01);
            vst1q_f32(out + 1 * outstride + 4, _r11);
            vst1q_f32(out + 1 * outstride + 8, _r21);
            vst1q_f32(out + 1 * outstride + 12, _r31);
            vst1q_f32(out + 2 * outstride + 0, _r02);
            vst1q_f32(out + 2 * outstride + 4, _r12);
            vst1q_f32(out + 2 * outstride + 8, _r22);
            vst1q_f32(out + 2 * outstride + 12, _r32);
            vst1q_f32(out + 3 * outstride + 0, _r03);
            vst1q_f32(out + 3 * outstride + 4, _r13);
            vst1q_f32(out + 3 * outstride + 8, _r23);
            vst1q_f32(out + 3 * outstride + 12, _r33);
        }
        for (; j < cols; j++)
        {
            const float* p = ptr + i * stride + j * 4;
            float* out = outptr + j * outstride + i * 4;
            vst1q_f32(out + 0, vld1q_f32(p + 0 * stride));
            vst1q_f32(out + 4, vld1q_f32(p + 1 * stride));
            vst1q_f32(out + 8, vld1q_f32(p + 2 * stride));
            vst1q_f32(out + 12, vld1q_f32(p + 3 * stride));
        }
    }
    for (; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            const float* p = ptr + i * stride + j * 4;
            float* out = outptr + j * outstride + i * 4;
            vst1q_f32(out, vld1q_f32(p));
        }
    }
}

static void permute_spatial_pack4to1(const float* ptr, size_t stride, float* outptr, size_t outstride, size_t outcstep, int rows, int cols)
{
    for (int x = 0; x < cols; x++)
        permute_unpack4_stride(ptr + x * 4, stride, outptr + x * outstride, outcstep, rows);
}

static void permute_transpose_blocks(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int size);

static void permute_transpose_blocks_tiled(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int size, int tile_size, int nT)
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
                permute_transpose_blocks(ptr + (size_t)i * stride + (size_t)j * size, stride, outptr + (size_t)j * outstride + (size_t)i * size, outstride, tile_rows, tile_cols, size);
            }
        }
        return;
    }

    permute_transpose_blocks(ptr, stride, outptr, outstride, rows, cols, size);
}

static void permute_transpose_spatial(const float* ptr, size_t stride, float* outptr, size_t outstride, size_t outcstep, int rows, int cols, int elempack, int out_elempack)
{
    if (elempack == out_elempack)
    {
        if (rows == 1 && outstride == (size_t)elempack)
        {
            memcpy(outptr, ptr, (size_t)cols * elempack * sizeof(float));
            return;
        }
        if (cols == 1 && stride == (size_t)elempack)
        {
            memcpy(outptr, ptr, (size_t)rows * elempack * sizeof(float));
            return;
        }
    }

    if (elempack == 1)
    {
        permute_transpose_blocks_tiled(ptr, stride, outptr, outstride, rows, cols, 1, 32, 1);
        return;
    }
    if (elempack == 4 && out_elempack == 4)
    {
        permute_transpose_blocks_tiled(ptr, stride, outptr, outstride, rows, cols, 4, 32, 1);
        return;
    }
    if (elempack == 4 && out_elempack == 1)
    {
        permute_spatial_pack4to1(ptr, stride, outptr, outstride, outcstep, rows, cols);
        return;
    }
}

static void permute_transpose_blocks2(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols)
{
    int j = 0;
    for (; j + 1 < cols; j += 2)
    {
        const float* p = ptr + (size_t)j * 2;
        float* out0 = outptr + j * outstride;
        float* out1 = out0 + outstride;
        int i = 0;
        for (; i + 1 < rows; i += 2)
        {
            float32x4_t _a = vld1q_f32(p);
            float32x4_t _b = vld1q_f32(p + stride);
            vst1q_f32(out0, permute_movelh_f32(_a, _b));
            vst1q_f32(out1, permute_movehl_f32(_b, _a));
            p += stride * 2;
            out0 += 4;
            out1 += 4;
        }
        for (; i < rows; i++)
        {
            memcpy(out0, p, 2 * sizeof(float));
            memcpy(out1, p + 2, 2 * sizeof(float));
            p += stride;
            out0 += 2;
            out1 += 2;
        }
    }
    for (; j < cols; j++)
    {
        const float* p = ptr + (size_t)j * 2;
        float* out = outptr + j * outstride;
        for (int i = 0; i < rows; i++)
        {
            memcpy(out, p, 2 * sizeof(float));
            p += stride;
            out += 2;
        }
    }
}

// transpose rows of contiguous blocks
// size is independent of elempack
// strides include padding; the block contents keep their original order
static void permute_transpose_blocks(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int size)
{
    if (rows == 1 && outstride == (size_t)size)
    {
        memcpy(outptr, ptr, (size_t)cols * size * sizeof(float));
        return;
    }
    if (cols == 1 && stride == (size_t)size)
    {
        memcpy(outptr, ptr, (size_t)rows * size * sizeof(float));
        return;
    }

    if (size == 1)
    {
        permute_transpose_pack1(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (size == 2)
    {
        permute_transpose_blocks2(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (size == 4)
    {
        permute_spatial_pack4(ptr, stride, outptr, outstride, rows, cols);
        return;
    }
    if (size >= 2 && size < 4)
    {
        for (int j = 0; j < cols; j++)
        {
            const float* p = ptr + (size_t)j * size;
            float* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                uint32x2_t _a = vld1_u32((const unsigned int*)p);
                uint32x2_t _b = vld1_u32((const unsigned int*)(p + size - 2));
                vst1_u32((unsigned int*)out, _a);
                vst1_u32((unsigned int*)(out + size - 2), _b);
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
            const float* p = ptr + (size_t)j * size;
            float* out = outptr + j * outstride;
            for (int i = 0; i < rows; i++)
            {
                uint32x4_t _a = vld1q_u32((const unsigned int*)p);
                uint32x4_t _b = vld1q_u32((const unsigned int*)(p + size - 4));
                vst1q_u32((unsigned int*)out, _a);
                vst1q_u32((unsigned int*)(out + size - 4), _b);
                p += stride;
                out += size;
            }
        }
        return;
    }
    for (int j = 0; j < cols; j++)
    {
        const float* p = ptr + (size_t)j * size;
        float* out = outptr + j * outstride;
        for (int i = 0; i < rows; i++)
        {
            memcpy(out, p, (size_t)size * sizeof(float));
            p += stride;
            out += size;
        }
    }
}

static void permute_unpack_spatial(const float* ptr, float* outptr, size_t outcstep, int size, int elempack)
{
    if (elempack == 4)
    {
        permute_unpack4_stride(ptr, 4, outptr, outcstep, size);
        return;
    }
}

// exchange the input channel axis with one output channel group
// w is the remaining spatial extent; cstep includes input channel padding
// the exchanged axis is contiguous; the stride variant has contiguous spatial input
static void permute3d_pack1to4(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * outcstep;
        for (int x = 0; x < w; x++)
        {
            uint32x4_t _v = vld1q_u32((const unsigned int*)p);
            vst1q_u32((unsigned int*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack1to4_stride(const float* ptr, float* outptr, int w, int channels, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * outcstep;
        permute_pack4_stride(p, hstep, out, outwstep, w);
    }
}

// output channels are contiguous; the stride variant has contiguous spatial output
static void permute3d_pack4to1(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4;
        for (int x = 0; x < w; x++)
        {
            uint32x4_t _v = vld1q_u32((const unsigned int*)p);
            vst1q_u32((unsigned int*)out, _v);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to1_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * outcstep;
        permute_unpack4_stride(p, wstep, out, outcstep, w);
    }
}

static void permute3d_pack4to4(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t cstep, size_t outwstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * 4;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x4(p, out);
            p += wstep;
            out += outwstep;
        }
    }
}

static void permute3d_pack4to4_stride(const float* ptr, float* outptr, int w, int channels, size_t wstep, size_t hstep, size_t cstep, size_t outwstep, size_t outcstep)
{
    for (int c = 0; c < channels; c++)
    {
        const float* p = ptr + c * cstep;
        float* out = outptr + c * 4 * outcstep;
        for (int x = 0; x < w; x++)
        {
            permute_transpose4x4_stride(p, hstep, out, outcstep);
            p += wstep;
            out += outwstep;
        }
    }
}

// the exchanged input axis is contiguous
static void permute_pack_channels(const Mat& bottom_blob, const Mat& top_blob, const float* ptr, float* outptr, int size, int channels, size_t wstep, size_t outwstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (out_elempack == 4)
    {
        permute3d_pack1to4(ptr, outptr, size, channels, wstep, cstep, outwstep, outcstep);
        return;
    }
}

// the remaining input spatial axis is contiguous
static void permute_pack_channels_stride(const Mat& bottom_blob, const Mat& top_blob, const float* ptr, float* outptr, int size, int channels, size_t hstep, size_t outwstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (out_elempack == 4)
    {
        permute3d_pack1to4_stride(ptr, outptr, size, channels, hstep, cstep, outwstep, outcstep);
        return;
    }
}

// the output channel axis is contiguous
static void permute_unpack_channels(const Mat& bottom_blob, const float* ptr, float* outptr, int size, int channels, size_t wstep, size_t outwstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int elempack = bottom_blob.elempack;

    if (elempack == 4)
    {
        permute3d_pack4to1(ptr, outptr, size, channels, wstep, cstep, outwstep);
        return;
    }
}

// the remaining output spatial axis is contiguous
static void permute_unpack_channels_stride(const Mat& bottom_blob, const float* ptr, float* outptr, int size, int channels, size_t wstep, size_t outcstep)
{
    const size_t cstep = bottom_blob.cstep * bottom_blob.elempack;
    const int elempack = bottom_blob.elempack;

    if (elempack == 4)
    {
        permute3d_pack4to1_stride(ptr, outptr, size, channels, wstep, cstep, outcstep);
        return;
    }
}

// both channel axes are packed
static void permute_channels_spatial_input_stride(const Mat& bottom_blob, const Mat& top_blob, const float* ptr, float* outptr, int size, int channels, size_t hstep, size_t outwstep, size_t outcstep)
{
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t wstep = elempack;

    if (elempack == 4 && out_elempack == 4)
    {
        if (hstep == 4 && outcstep == 4)
        {
            permute3d_pack4to4(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack4to4_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
}

static void permute_channels_axis_input_stride(const Mat& bottom_blob, const Mat& top_blob, const float* ptr, float* outptr, int size, int channels, size_t wstep, size_t outwstep, size_t outcstep)
{
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t cstep = bottom_blob.cstep * elempack;
    const size_t hstep = elempack;

    if (elempack == 4 && out_elempack == 4)
    {
        if (hstep == 4 && outcstep == 4)
        {
            permute3d_pack4to4(ptr, outptr, size, channels, wstep, cstep, outwstep);
        }
        else
        {
            permute3d_pack4to4_stride(ptr, outptr, size, channels, wstep, hstep, cstep, outwstep, outcstep);
        }
        return;
    }
}

static void permute_transpose_matrix(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int size, int nT)
{
    if (size == 1 && rows >= 256 && cols >= 256 && rows % 32 == 0 && cols % 32 == 0)
    {
        permute_transpose_blocks_tiled(ptr, stride, outptr, outstride, rows, cols, 1, rows >= 512 && cols >= 512 ? 16 : 32, nT);
        return;
    }

    // scalar matrices keep the full register transpose width
    const int step = size == 1 ? 4 : 1;
    #pragma omp parallel for num_threads(nT)
    for (int j = 0; j < cols; j += step)
    {
        permute_transpose_blocks(ptr + (size_t)j * size, stride, outptr + j * outstride, outstride, rows, std::min(step, cols - j), size);
    }
}

// transpose independent matrices
static void permute_transpose_matrices_stride(const float* ptr, size_t stride, float* outptr, size_t outstride, int rows, int cols, int size, int planes, size_t step, size_t outstep, int nT)
{
    if (planes == 1)
    {
        permute_transpose_matrix(ptr, stride, outptr, outstride, rows, cols, size, nT);
        return;
    }

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < planes; q++)
    {
        permute_transpose_blocks(ptr + q * step, stride, outptr + q * outstep, outstride, rows, cols, size);
    }
}

// transpose spatial planes within each input channel group
static void permute_transpose_spatial_planes_stride(const Mat& bottom_blob, Mat& top_blob, int rows, int cols, size_t stride, size_t outstride, int planes, size_t step, size_t outstep, int nT)
{
    (void)nT;

    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q * elempack / out_elempack);
        for (int z = 0; z < planes; z++)
        {
            permute_transpose_spatial(ptr + z * step, stride, outptr + z * outstep, outstride, top_blob.cstep, rows, cols, elempack, out_elempack);
        }
    }
}

// transpose h and w while preserving the other two slice axes
static void permute_transpose_hw_pack1_stride(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outcstep, size_t outdstep, int nT)
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
        const float* ptr = bottom_blob.channel(q);
        float* outptr = (float*)top_blob + q * outcstep;
        for (int z = 0; z < d; z++)
        {
            permute_transpose_pack1(ptr + z * (size_t)w * h, stride, outptr + z * outdstep, outstride, h, w);
        }
    }
}

// transpose d and w while preserving the other two slice axes
static void permute_transpose_dw_pack1_stride(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outcstep, size_t outhstep, int nT)
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
        const float* ptr = bottom_blob.channel(q);
        float* outptr = (float*)top_blob + q * outcstep;
        for (int y = 0; y < h; y++)
        {
            permute_transpose_pack1(ptr + y * (size_t)w, stride, outptr + y * outhstep, outstride, d, w);
        }
    }
}

// transpose c and w while preserving the other two slice axes
static void permute_transpose_cw_pack1_stride(const Mat& bottom_blob, Mat& top_blob, size_t outstride, size_t outdstep, size_t outhstep, int nT)
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
        const float* ptr = (const float*)bottom_blob + z * (size_t)w * h;
        float* outptr = (float*)top_blob + z * outdstep;
        for (int y = 0; y < h; y++)
        {
            permute_transpose_pack1(ptr + y * (size_t)w, stride, outptr + y * outhstep, outstride, channels, w);
        }
    }
}

static void permute2d(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        permute_transpose_matrix(bottom_blob, w, top_blob, h, h, w, 1, nT);
        return;
    }

    if (elempack == 4 && out_elempack == 4 && w >= 64 && h >= 64)
    {
        permute_transpose_pack4to4_tiled((const float*)bottom_blob, (size_t)w * 4, (float*)top_blob, (size_t)top_blob.w * 4, h * 4, w, nT);
        return;
    }

    #pragma omp parallel for num_threads(nT)
    for (int x = 0; x < w; x += out_elempack)
    {
        const float* ptr = (const float*)bottom_blob + (size_t)x * elempack;
        float* outptr = top_blob.row<float>(x / out_elempack);
        permute_transpose2d(ptr, (size_t)w * elempack, outptr, (size_t)top_blob.w * out_elempack, h * elempack, out_elempack, elempack, out_elempack);
    }
}

static void permute3d_hwc(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    const size_t stride = (size_t)w * elempack;
    const size_t outstride = (size_t)h * out_elempack;
    const size_t step = 0;
    const size_t outstep = 0;
    permute_transpose_spatial_planes_stride(bottom_blob, top_blob, h, w, stride, outstride, 1, step, outstep, nT);
}

static void permute3d_wch(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        if (w < 32)
        {
            // short blocks include the w == 1 matrix-transpose case
            permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, top_blob.cstep, channels, h, w, nT);
            return;
        }

        #pragma omp parallel for num_threads(nT)
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob + q * (size_t)w;
            float* outptr = top_blob.channel(q);
            for (int y = 0; y < top_blob.h; y++)
            {
                memcpy(outptr + (size_t)y * w, ptr + y * bottom_blob.cstep, (size_t)w * sizeof(float));
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
        const float* ptr = (const float*)bottom_blob + (size_t)q * out_elempack * hstep;
        float* outptr = (float*)top_blob.channel(q);
        // exchange c and h, keeping w as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels_stride(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
        else
            permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
    }
}

static void permute3d_cwh(const Mat& bottom_blob, Mat& top_blob, int nT)
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
            permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, channels, channels, w * h, 1, nT);
            return;
        }

        const size_t stride = bottom_blob.cstep;
        const size_t outstride = (size_t)top_blob.w;
        const size_t step = (size_t)w;
        const size_t outstep = top_blob.cstep;
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, channels, w, 1, h, step, outstep, nT);
        return;
    }

    const size_t wstep = elempack;
    const size_t hstep = (size_t)w * elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const float* ptr = (const float*)bottom_blob + (size_t)q * out_elempack * hstep;
        float* outptr = (float*)top_blob.channel(q);
        // exchange c and h, keeping w as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
        else
            permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
    }
}

static void permute3d_hcw(const Mat& bottom_blob, Mat& top_blob, int nT)
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
            permute_transpose_matrix(bottom_blob, w, top_blob, top_blob.cstep, channels * h, w, 1, nT);
            return;
        }

        const size_t stride = (size_t)w;
        const size_t outstride = top_blob.cstep;
        const size_t step = bottom_blob.cstep;
        const size_t outstep = (size_t)top_blob.w;
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, h, w, 1, channels, step, outstep, nT);
        return;
    }

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = out_elempack;
    const size_t outcstep = (size_t)top_blob.w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const float* ptr = (const float*)bottom_blob + (size_t)q * out_elempack * hstep;
        float* outptr = (float*)top_blob.channel(q);
        // exchange c and w, keeping h as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels_stride(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
        else
            permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
    }
}

static void permute3d_chw(const Mat& bottom_blob, Mat& top_blob, int nT)
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
            permute_transpose_matrix(bottom_blob, bottom_blob.cstep, top_blob, w == 1 ? (size_t)channels : top_blob.cstep, channels, w * h, 1, nT);
            return;
        }

        if (channels == 1)
        {
            permute_transpose_matrix(bottom_blob, w, top_blob, top_blob.cstep, h, w, 1, nT);
            return;
        }

        const size_t stride = bottom_blob.cstep;
        const size_t outstride = top_blob.cstep;
        const size_t step = (size_t)w;
        const size_t outstep = (size_t)top_blob.w;
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, channels, w, 1, h, step, outstep, nT);
        return;
    }

    const size_t wstep = (size_t)w * elempack;
    const size_t hstep = elempack;
    const size_t outwstep = (size_t)top_blob.w * out_elempack;
    const size_t outcstep = out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < top_blob.c; q++)
    {
        const float* ptr = (const float*)bottom_blob + (size_t)q * out_elempack * hstep;
        float* outptr = (float*)top_blob.channel(q);
        // exchange c and w, keeping h as the inner spatial axis
        if (elempack == 1)
            permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        else if (out_elempack == 1)
            permute_unpack_channels(bottom_blob, ptr, outptr, h, channels, wstep, outwstep);
        else
            permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
    }
}

static void permute4d_hwdc(const Mat& bottom_blob, Mat& top_blob, int nT)
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
    permute_transpose_spatial_planes_stride(bottom_blob, top_blob, h, w, stride, outstride, d, step, outstep, nT);
}

static void permute4d_wdhc(const Mat& bottom_blob, Mat& top_blob, int nT)
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
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, d, h, w * elempack, channels, step, outstep, nT);
        return;
    }

    const size_t hstep = (size_t)w * elempack;
    const size_t dstep = (size_t)w * h * elempack;
    const size_t outhstep = (size_t)w * d * out_elempack;
    const size_t outdstep = (size_t)w * out_elempack;

    #pragma omp parallel for num_threads(nT)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q * elempack / out_elempack);
        for (int y = 0; y < h; y++)
        {
            const float* p = ptr + y * hstep;
            float* out = outptr + y * outhstep;
            for (int z = 0; z < d; z++)
            {
                permute_unpack_spatial(p + z * dstep, out + z * outdstep, top_blob.cstep, w, elempack);
            }
        }
    }
}

static void permute4d_dwhc(const Mat& bottom_blob, Mat& top_blob, int nT)
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
    permute_transpose_spatial_planes_stride(bottom_blob, top_blob, d, w, stride, outstride, h, step, outstep, nT);
}

static void permute4d_hdwc(const Mat& bottom_blob, Mat& top_blob, int nT)
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
    permute_transpose_spatial_planes_stride(bottom_blob, top_blob, h, w, stride, outstride, d, step, outstep, nT);
}

static void permute4d_dhwc(const Mat& bottom_blob, Mat& top_blob, int nT)
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
    permute_transpose_spatial_planes_stride(bottom_blob, top_blob, d, w, stride, outstride, h, step, outstep, nT);
}

static void permute4d_whcd(const Mat& bottom_blob, Mat& top_blob, int nT)
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
    permute3d_wch(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_hwcd(const Mat& bottom_blob, Mat& top_blob, int nT)
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
        permute_transpose_hw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outdstep, nT);
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
                const float* ptr = (const float*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + x * outxstep;
                // exchange c and d, keeping h as the inner spatial axis
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
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
            const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_wchd(const Mat& bottom_blob, Mat& top_blob, int nT)
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
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, channels, h, w, d, step, outstep, nT);
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
            const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cwhd(const Mat& bottom_blob, Mat& top_blob, int nT)
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
    permute3d_cwh(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_hcwd(const Mat& bottom_blob, Mat& top_blob, int nT)
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
        permute_transpose_hw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outdstep, nT);
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
                const float* ptr = (const float*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + x * outxstep;
                // exchange c and d, keeping h as the inner spatial axis
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
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
            const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_chwd(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = top_blob.cstep;
        const size_t outhstep = (size_t)top_blob.w;
        permute_transpose_cw_pack1_stride(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
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
            const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + y * outystep;
            // exchange c and d, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_wdch(const Mat& bottom_blob, Mat& top_blob, int nT)
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
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, d, h, w, channels, step, outstep, nT);
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
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_dwch(const Mat& bottom_blob, Mat& top_blob, int nT)
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
        permute_transpose_dw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
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
                const float* ptr = (const float*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + x * outxstep;
                // exchange c and h, keeping d as the inner spatial axis
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
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
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_wcdh(const Mat& bottom_blob, Mat& top_blob, int nT)
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
        permute_transpose_matrices_stride(bottom_blob, stride, top_blob, outstride, channels, h, w, d, step, outstep, nT);
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
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, w, channels, wstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cwdh(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w;
        const size_t outdstep = (size_t)top_blob.w * top_blob.h;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_cw_pack1_stride(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
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
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_dcwh(const Mat& bottom_blob, Mat& top_blob, int nT)
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
        permute_transpose_dw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
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
                const float* ptr = (const float*)bottom_blob + x * xstep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + x * outxstep;
                // exchange c and h, keeping d as the inner spatial axis
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
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
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cdwh(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = (size_t)top_blob.w * top_blob.h;
        const size_t outdstep = (size_t)top_blob.w;
        const size_t outhstep = top_blob.cstep;
        permute_transpose_cw_pack1_stride(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
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
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and h, keeping w as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels(bottom_blob, ptr, outptr, w, channels, wstep, outwstep);
            else
                permute_channels_spatial_input_stride(bottom_blob, top_blob, ptr, outptr, w, channels, hstep, outwstep, outcstep);
        }
    }
}

static void permute4d_hdcw(const Mat& bottom_blob, Mat& top_blob, int nT)
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
    permute3d_hcw(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_dhcw(const Mat& bottom_blob, Mat& top_blob, int nT)
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
        permute_transpose_dw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
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
                const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + y * outystep;
                // exchange c and w, keeping d as the inner spatial axis
                if (elempack == 1)
                    permute_pack_channels(bottom_blob, top_blob, ptr, outptr, d, channels, wstep, outwstep, outcstep);
                else
                    permute_unpack_channels_stride(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
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
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else
                permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}

static void permute4d_hcdw(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outcstep = (size_t)top_blob.w;
        const size_t outdstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_hw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outdstep, nT);
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
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels_stride(bottom_blob, ptr, outptr, h, channels, wstep, outcstep);
            else
                permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}

static void permute4d_chdw(const Mat& bottom_blob, Mat& top_blob, int nT)
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
    permute3d_chw(bottom_blob_3d, top_blob_3d, nT);
}

static void permute4d_dchw(const Mat& bottom_blob, Mat& top_blob, int nT)
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
        permute_transpose_dw_pack1_stride(bottom_blob, top_blob, outstride, outcstep, outhstep, nT);
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
                const float* ptr = (const float*)bottom_blob + y * ystep + (size_t)q * out_elempack * hstep;
                float* outptr = (float*)top_blob.channel(q) + y * outystep;
                // exchange c and w, keeping d as the inner spatial axis
                if (elempack == 1)
                    permute_pack_channels(bottom_blob, top_blob, ptr, outptr, d, channels, wstep, outwstep, outcstep);
                else
                    permute_unpack_channels_stride(bottom_blob, ptr, outptr, d, channels, wstep, outcstep);
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
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else
                permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}

static void permute4d_cdhw(const Mat& bottom_blob, Mat& top_blob, int nT)
{
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 1 && out_elempack == 1)
    {
        const size_t outstride = top_blob.cstep;
        const size_t outdstep = (size_t)top_blob.w;
        const size_t outhstep = (size_t)top_blob.w * top_blob.h;
        permute_transpose_cw_pack1_stride(bottom_blob, top_blob, outstride, outdstep, outhstep, nT);
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
            const float* ptr = (const float*)bottom_blob + z * zstep + (size_t)q * out_elempack * hstep;
            float* outptr = (float*)top_blob.channel(q) + z * outzstep;
            // exchange c and w, keeping h as the inner spatial axis
            if (elempack == 1)
                permute_pack_channels(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
            else if (out_elempack == 1)
                permute_unpack_channels(bottom_blob, ptr, outptr, h, channels, wstep, outwstep);
            else
                permute_channels_axis_input_stride(bottom_blob, top_blob, ptr, outptr, h, channels, wstep, outwstep, outcstep);
        }
    }
}
