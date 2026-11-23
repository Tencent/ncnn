// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "permute_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include <string.h>

#include "cpu.h"

namespace ncnn {

static const int permute_axes_2d[2][2] =
{
    {0, 1},
    {1, 0}
};

static const int permute_axes_3d[6][3] =
{
    {0, 1, 3},
    {1, 0, 3},
    {0, 3, 1},
    {3, 0, 1},
    {1, 3, 0},
    {3, 1, 0}
};

static const int permute_axes_4d[24][4] =
{
    {0, 1, 2, 3},
    {1, 0, 2, 3},
    {0, 2, 1, 3},
    {2, 0, 1, 3},
    {1, 2, 0, 3},
    {2, 1, 0, 3},
    {0, 1, 3, 2},
    {1, 0, 3, 2},
    {0, 3, 1, 2},
    {3, 0, 1, 2},
    {1, 3, 0, 2},
    {3, 1, 0, 2},
    {0, 2, 3, 1},
    {2, 0, 3, 1},
    {0, 3, 2, 1},
    {3, 0, 2, 1},
    {2, 3, 0, 1},
    {3, 2, 0, 1},
    {1, 2, 3, 0},
    {2, 1, 3, 0},
    {1, 3, 2, 0},
    {3, 1, 2, 0},
    {2, 3, 1, 0},
    {3, 2, 1, 0}
};

static NCNN_FORCEINLINE size_t permute_input_offset(const Mat& bottom_blob, const int* pos, int scalar_size)
{
    const int elempack = bottom_blob.elempack;
    const int dims = bottom_blob.dims;
    const size_t elemsize = (size_t)scalar_size * elempack;
    if (dims == 2)
    {
        const int h = pos[1];
        const int q = h / elempack;
        const int lane = h % elempack;
        return q * bottom_blob.w * elemsize + (size_t)pos[0] * elemsize + (size_t)lane * scalar_size;
    }

    const int c = pos[3];
    const int q = c / elempack;
    const int lane = c % elempack;

    size_t offset = q * bottom_blob.cstep * elemsize;
    if (dims == 3)
    {
        offset += ((size_t)pos[1] * bottom_blob.w + pos[0]) * elemsize;
    }
    else
    {
        offset += (((size_t)pos[2] * bottom_blob.h + pos[1]) * bottom_blob.w + pos[0]) * elemsize;
    }
    offset += (size_t)lane * scalar_size;

    return offset;
}

static NCNN_FORCEINLINE size_t permute_output_offset(const Mat& top_blob, const int* pos, int inner, int scalar_size)
{
    const int elempack = top_blob.elempack;
    const int dims = top_blob.dims;
    const size_t elemsize = (size_t)scalar_size * elempack;
    const int q = inner / elempack;
    const int lane = inner % elempack;

    if (dims == 2)
    {
        return (size_t)q * top_blob.w * elemsize + (size_t)pos[0] * elemsize + (size_t)lane * scalar_size;
    }

    size_t offset = (size_t)q * top_blob.cstep * elemsize;
    if (dims == 3)
    {
        offset += ((size_t)pos[1] * top_blob.w + pos[0]) * elemsize;
    }
    else
    {
        offset += (((size_t)pos[2] * top_blob.h + pos[1]) * top_blob.w + pos[0]) * elemsize;
    }
    offset += (size_t)lane * scalar_size;

    return offset;
}

#if __ARM_NEON && __aarch64__
static NCNN_FORCEINLINE void permute_transpose4x4_ps(float32x4_t& r0, float32x4_t& r1, float32x4_t& r2, float32x4_t& r3)
{
    float32x4_t _t0 = vtrn1q_f32(r0, r1);
    float32x4_t _t1 = vtrn2q_f32(r0, r1);
    float32x4_t _t2 = vtrn1q_f32(r2, r3);
    float32x4_t _t3 = vtrn2q_f32(r2, r3);

    r0 = vcombine_f32(vget_low_f32(_t0), vget_low_f32(_t2));
    r1 = vcombine_f32(vget_low_f32(_t1), vget_low_f32(_t3));
    r2 = vcombine_f32(vget_high_f32(_t0), vget_high_f32(_t2));
    r3 = vcombine_f32(vget_high_f32(_t1), vget_high_f32(_t3));
}

static NCNN_FORCEINLINE float32x4_t permute_load_c4(const Mat& bottom_blob, int cbase, int h, int w)
{
    const int elempack = bottom_blob.elempack;
    if (elempack == 4)
    {
        const float* ptr = (const float*)bottom_blob.channel(cbase / 4) + (h * bottom_blob.w + w) * 4;
        return vld1q_f32(ptr);
    }

    float32x4_t v;
    v = vsetq_lane_f32(((const float*)bottom_blob.channel(cbase))[h * bottom_blob.w + w], v, 0);
    v = vsetq_lane_f32(((const float*)bottom_blob.channel(cbase + 1))[h * bottom_blob.w + w], v, 1);
    v = vsetq_lane_f32(((const float*)bottom_blob.channel(cbase + 2))[h * bottom_blob.w + w], v, 2);
    v = vsetq_lane_f32(((const float*)bottom_blob.channel(cbase + 3))[h * bottom_blob.w + w], v, 3);
    return v;
}

static int permute2d_ps(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (w % 4 != 0 || h % 4 != 0)
        return -1;

    if (elempack == 1 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y += 4)
        {
            for (int x = 0; x < w; x += 4)
            {
                float32x4_t _r0 = vld1q_f32((const float*)bottom_blob.row(y + 0) + x);
                float32x4_t _r1 = vld1q_f32((const float*)bottom_blob.row(y + 1) + x);
                float32x4_t _r2 = vld1q_f32((const float*)bottom_blob.row(y + 2) + x);
                float32x4_t _r3 = vld1q_f32((const float*)bottom_blob.row(y + 3) + x);
                permute_transpose4x4_ps(_r0, _r1, _r2, _r3);
                permute_transpose4x4_ps(_r0, _r1, _r2, _r3);

                float* outptr = (float*)top_blob.row(x / 4) + y * 4;
                vst1q_f32(outptr + 0, _r0);
                vst1q_f32(outptr + 4, _r1);
                vst1q_f32(outptr + 8, _r2);
                vst1q_f32(outptr + 12, _r3);
            }
        }
        return 0;
    }

    if (elempack == 1 && out_elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y += 4)
        {
            for (int x = 0; x < w; x += 4)
            {
                float32x4_t _r0 = vld1q_f32((const float*)bottom_blob.row(y + 0) + x);
                float32x4_t _r1 = vld1q_f32((const float*)bottom_blob.row(y + 1) + x);
                float32x4_t _r2 = vld1q_f32((const float*)bottom_blob.row(y + 2) + x);
                float32x4_t _r3 = vld1q_f32((const float*)bottom_blob.row(y + 3) + x);
                permute_transpose4x4_ps(_r0, _r1, _r2, _r3);

                vst1q_f32((float*)top_blob.row(x + 0) + y, _r0);
                vst1q_f32((float*)top_blob.row(x + 1) + y, _r1);
                vst1q_f32((float*)top_blob.row(x + 2) + y, _r2);
                vst1q_f32((float*)top_blob.row(x + 3) + y, _r3);
            }
        }
        return 0;
    }

    if (elempack == 4 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const float* ptr = (const float*)bottom_blob.row(y);
            for (int x = 0; x < w; x += 4)
            {
                float32x4x4_t _r = vld4q_f32(ptr + x * 4);
                permute_transpose4x4_ps(_r.val[0], _r.val[1], _r.val[2], _r.val[3]);
                permute_transpose4x4_ps(_r.val[0], _r.val[1], _r.val[2], _r.val[3]);

                float* outptr = (float*)top_blob.row(x / 4) + y * 16;
                vst1q_f32(outptr + 0, _r.val[0]);
                vst1q_f32(outptr + 4, _r.val[1]);
                vst1q_f32(outptr + 8, _r.val[2]);
                vst1q_f32(outptr + 12, _r.val[3]);
            }
        }
        return 0;
    }

    return -1;
}

static int permute3d_ps(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt);

static int permute4d_ps(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    if (order_type != 6 && order_type != 9 && order_type != 18 && order_type != 21)
        return -1;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t out_elemsize = top_blob.elemsize;

    if (order_type == 6)
    {
        Mat bottom_blob_3d(w * h, d, channels, bottom_blob.data, elemsize, elempack);
        bottom_blob_3d.cstep = bottom_blob.cstep;
        Mat top_blob_3d(w * h, channels * elempack, top_blob.c, top_blob.data, out_elemsize, out_elempack);
        top_blob_3d.cstep = top_blob.cstep;
        return permute3d_ps(bottom_blob_3d, top_blob_3d, 2, opt);
    }

    if (order_type == 9)
    {
        Mat bottom_blob_3d(w * h, d, channels, bottom_blob.data, elemsize, elempack);
        bottom_blob_3d.cstep = bottom_blob.cstep;
        Mat top_blob_3d(channels * elempack, w * h, top_blob.c, top_blob.data, out_elemsize, out_elempack);
        top_blob_3d.cstep = top_blob.cstep;
        return permute3d_ps(bottom_blob_3d, top_blob_3d, 3, opt);
    }

    if (order_type == 18)
    {
        Mat bottom_blob_3d(w, h * d, channels, bottom_blob.data, elemsize, elempack);
        bottom_blob_3d.cstep = bottom_blob.cstep;
        Mat top_blob_3d(h * d, channels * elempack, top_blob.c, top_blob.data, out_elemsize, out_elempack);
        top_blob_3d.cstep = top_blob.cstep;
        return permute3d_ps(bottom_blob_3d, top_blob_3d, 4, opt);
    }

    Mat bottom_blob_3d(w, h * d, channels, bottom_blob.data, elemsize, elempack);
    bottom_blob_3d.cstep = bottom_blob.cstep;
    Mat top_blob_3d(channels * elempack, h * d, top_blob.c, top_blob.data, out_elemsize, out_elempack);
    top_blob_3d.cstep = top_blob.cstep;
    return permute3d_ps(bottom_blob_3d, top_blob_3d, 5, opt);
}

static int permute3d_ps(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int logical_channels = channels * elempack;
    const int outw = top_blob.w;

    if (order_type == 1 && elempack == out_elempack && (elempack == 1 || elempack == 4) && w % 4 == 0 && h % 4 == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = (const float*)bottom_blob.channel(q);
            float* outptr = (float*)top_blob.channel(q);

            if (elempack == 1)
            {
                for (int y = 0; y < h; y += 4)
                {
                    for (int x = 0; x < w; x += 4)
                    {
                        float32x4_t _r0 = vld1q_f32(ptr + y * w + x);
                        float32x4_t _r1 = vld1q_f32(ptr + (y + 1) * w + x);
                        float32x4_t _r2 = vld1q_f32(ptr + (y + 2) * w + x);
                        float32x4_t _r3 = vld1q_f32(ptr + (y + 3) * w + x);
                        permute_transpose4x4_ps(_r0, _r1, _r2, _r3);

                        vst1q_f32(outptr + (x + 0) * outw + y, _r0);
                        vst1q_f32(outptr + (x + 1) * outw + y, _r1);
                        vst1q_f32(outptr + (x + 2) * outw + y, _r2);
                        vst1q_f32(outptr + (x + 3) * outw + y, _r3);
                    }
                }
            }
            else
            {
                for (int y = 0; y < h; y += 4)
                {
                    for (int x = 0; x < w; x += 4)
                    {
                        float32x4_t _r[4][4];
                        for (int i = 0; i < 4; i++)
                        {
                            float32x4x4_t _v = vld4q_f32(ptr + (y + i) * w * 4 + x * 4);
                            _r[i][0] = _v.val[0];
                            _r[i][1] = _v.val[1];
                            _r[i][2] = _v.val[2];
                            _r[i][3] = _v.val[3];
                        }

                        float32x4_t _t[4][4];
                        for (int c = 0; c < 4; c++)
                        {
                            _t[c][0] = _r[0][c];
                            _t[c][1] = _r[1][c];
                            _t[c][2] = _r[2][c];
                            _t[c][3] = _r[3][c];
                            permute_transpose4x4_ps(_t[c][0], _t[c][1], _t[c][2], _t[c][3]);
                        }

                        for (int j = 0; j < 4; j++)
                        {
                            float32x4_t _u0 = _t[0][j];
                            float32x4_t _u1 = _t[1][j];
                            float32x4_t _u2 = _t[2][j];
                            float32x4_t _u3 = _t[3][j];
                            permute_transpose4x4_ps(_u0, _u1, _u2, _u3);

                            float* out = outptr + ((x + j) * outw + y) * 4;
                            vst1q_f32(out, _u0);
                            vst1q_f32(out + 4, _u1);
                            vst1q_f32(out + 8, _u2);
                            vst1q_f32(out + 12, _u3);
                        }
                    }
                }
            }
        }

        return 0;
    }

    if ((order_type == 2 || order_type == 3) && h % 4 == 0 && logical_channels % 4 == 0 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int c = 0; c < logical_channels; c += 4)
        {
            for (int y = 0; y < h; y += 4)
            {
                for (int x = 0; x < w; x++)
                {
                    float32x4_t _r0 = permute_load_c4(bottom_blob, c, y + 0, x);
                    float32x4_t _r1 = permute_load_c4(bottom_blob, c, y + 1, x);
                    float32x4_t _r2 = permute_load_c4(bottom_blob, c, y + 2, x);
                    float32x4_t _r3 = permute_load_c4(bottom_blob, c, y + 3, x);
                    permute_transpose4x4_ps(_r0, _r1, _r2, _r3);

                    float* outq = (float*)top_blob.channel(y / 4);
                    for (int l = 0; l < 4; l++)
                    {
                        float32x4_t _v = l == 0 ? _r0 : l == 1 ? _r1
                                         : l == 2   ? _r2
                                         : _r3;
                        float* outptr = order_type == 2 ? outq + ((c + l) * top_blob.w + x) * 4 : outq + (x * top_blob.w + c + l) * 4;
                        vst1q_f32(outptr, _v);
                    }
                }
            }
        }

        return 0;
    }

    if ((order_type == 4 || order_type == 5) && w % 4 == 0 && logical_channels % 4 == 0 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            for (int c = 0; c < logical_channels; c += 4)
            {
                for (int x = 0; x < w; x += 4)
                {
                    float32x4_t _r0 = permute_load_c4(bottom_blob, c, y, x + 0);
                    float32x4_t _r1 = permute_load_c4(bottom_blob, c, y, x + 1);
                    float32x4_t _r2 = permute_load_c4(bottom_blob, c, y, x + 2);
                    float32x4_t _r3 = permute_load_c4(bottom_blob, c, y, x + 3);
                    permute_transpose4x4_ps(_r0, _r1, _r2, _r3);

                    float* outq = (float*)top_blob.channel(x / 4);
                    for (int l = 0; l < 4; l++)
                    {
                        float32x4_t _v = l == 0 ? _r0 : l == 1 ? _r1
                                         : l == 2   ? _r2
                                         : _r3;
                        float* outptr = order_type == 4 ? outq + ((c + l) * top_blob.w + y) * 4 : outq + (y * top_blob.w + c + l) * 4;
                        vst1q_f32(outptr, _v);
                    }
                }
            }
        }

        return 0;
    }

    return -1;
}
#endif // __ARM_NEON && __aarch64__

Permute_arm::Permute_arm()
{
#if __ARM_NEON
    support_packing = true;
    support_any_packing = true;
#endif // __ARM_NEON
}

int Permute_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;

    if (dims == 1 || order_type == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    const int in_w = bottom_blob.w;
    const int in_h = bottom_blob.h;
    const int in_d = bottom_blob.d;
    const int in_c = bottom_blob.c;
    const int in_elempack = bottom_blob.elempack;

    const int scalar_bits = bottom_blob.elembits();
    if (scalar_bits != 16 && scalar_bits != 32)
        return -100;

    const int scalar_size = scalar_bits / 8;

    int in_ext[4] = {in_w, in_h, in_d, in_c * in_elempack};
    if (dims == 2)
        in_ext[1] *= in_elempack;
    int out_ext[4] = {1, 1, 1, 1};

    const int* axes;
    if (dims == 2)
        axes = permute_axes_2d[order_type];
    else if (dims == 3)
        axes = permute_axes_3d[order_type];
    else
        axes = permute_axes_4d[order_type];

    for (int i = 0; i < dims; i++)
    {
        out_ext[i] = in_ext[axes[i]];
    }

    const int last_axis = axes[dims - 1];
    const int input_last_axis = dims == 2 ? 1 : 3;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        if (last_axis == input_last_axis)
        {
            out_elempack = in_elempack;
        }
        else
        {
            const int axis_size = out_ext[dims - 1];
            out_elempack = axis_size % 4 == 0 ? 4 : 1;
        }
    }

    const size_t out_elemsize = (size_t)scalar_size * out_elempack;

    if (dims == 2)
        top_blob.create(out_ext[0], out_ext[1] / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (dims == 3)
        top_blob.create(out_ext[0], out_ext[1], out_ext[2] / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (dims == 4)
        top_blob.create(out_ext[0], out_ext[1], out_ext[2], out_ext[3] / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

#if __ARM_NEON && __aarch64__
    if (dims == 2 && order_type == 1 && scalar_size == 4)
    {
        int ret = permute2d_ps(bottom_blob, top_blob, opt);
        if (ret == 0)
            return 0;
    }
    if (dims == 3 && scalar_size == 4)
    {
        int ret = permute3d_ps(bottom_blob, top_blob, order_type, opt);
        if (ret == 0)
            return 0;
    }
    if (dims == 4 && scalar_size == 4)
    {
        int ret = permute4d_ps(bottom_blob, top_blob, order_type, opt);
        if (ret == 0)
            return 0;
    }
#endif // __ARM_NEON && __aarch64__

    int outer_count = 1;
    for (int i = 0; i < dims - 1; i++)
    {
        outer_count *= out_ext[i];
    }

    const int inner_ext = out_ext[dims - 1];
    const unsigned char* bottom_ptr = (const unsigned char*)bottom_blob.data;
    unsigned char* top_ptr = (unsigned char*)top_blob.data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < outer_count; r++)
    {
        int out_pos[4] = {0, 0, 0, 0};
        int in_pos[4] = {0, 0, 0, 0};

        int v = r;
        for (int i = 0; i < dims - 1; i++)
        {
            out_pos[i] = v % out_ext[i];
            v /= out_ext[i];
            in_pos[axes[i]] = out_pos[i];
        }

        in_pos[last_axis] = 0;

        const size_t in_elemsize = (size_t)scalar_size * in_elempack;
        const size_t out_elemsize = (size_t)scalar_size * out_elempack;
        const size_t out_cstep_bytes = top_blob.cstep * out_elemsize;

        if (dims == 2)
        {
            for (int j = 0; j < inner_ext; j++)
            {
                in_pos[last_axis] = j;
                const size_t src_offset = permute_input_offset(bottom_blob, in_pos, scalar_size);
                const size_t dst_offset = permute_output_offset(top_blob, out_pos, j, scalar_size);
                memcpy(top_ptr + dst_offset, bottom_ptr + src_offset, scalar_size);
            }
            continue;
        }

        if (last_axis != 3)
        {
            size_t in_step = in_elemsize;
            if (last_axis == 1)
                in_step = (size_t)in_w * in_elemsize;
            if (last_axis == 2)
                in_step = (size_t)in_w * in_h * in_elemsize;

            const unsigned char* inptr = bottom_ptr + permute_input_offset(bottom_blob, in_pos, scalar_size);
            unsigned char* outptr = top_ptr + permute_output_offset(top_blob, out_pos, 0, scalar_size);

            if (out_elempack == 1)
            {
                if (scalar_size == 4)
                {
                    for (int j = 0; j < inner_ext; j++)
                    {
                        *(float*)outptr = *(const float*)inptr;
                        inptr += in_step;
                        outptr += out_cstep_bytes;
                    }
                }
                else
                {
                    for (int j = 0; j < inner_ext; j++)
                    {
                        *(unsigned short*)outptr = *(const unsigned short*)inptr;
                        inptr += in_step;
                        outptr += out_cstep_bytes;
                    }
                }
            }
            else
            {
                int j = 0;
#if __ARM_NEON
                for (; j + 3 < inner_ext; j += 4)
                {
                    if (scalar_size == 4)
                    {
                        float32x4_t _p;
                        _p = vsetq_lane_f32(*(const float*)inptr, _p, 0);
                        _p = vsetq_lane_f32(*(const float*)(inptr + in_step), _p, 1);
                        _p = vsetq_lane_f32(*(const float*)(inptr + in_step * 2), _p, 2);
                        _p = vsetq_lane_f32(*(const float*)(inptr + in_step * 3), _p, 3);
                        vst1q_f32((float*)outptr, _p);
                    }
                    else
                    {
                        uint16x4_t _p;
                        _p = vset_lane_u16(*(const unsigned short*)inptr, _p, 0);
                        _p = vset_lane_u16(*(const unsigned short*)(inptr + in_step), _p, 1);
                        _p = vset_lane_u16(*(const unsigned short*)(inptr + in_step * 2), _p, 2);
                        _p = vset_lane_u16(*(const unsigned short*)(inptr + in_step * 3), _p, 3);
                        vst1_u16((unsigned short*)outptr, _p);
                    }
                    inptr += in_step * 4;
                    outptr += out_cstep_bytes;
                }
#endif // __ARM_NEON
                for (; j < inner_ext; j++)
                {
                    const unsigned char* outp = outptr + (j % out_elempack) * scalar_size;
                    memcpy((void*)outp, inptr, scalar_size);
                    inptr += in_step;
                }
            }
        }
        else
        {
            const unsigned char* inbase = bottom_ptr + permute_input_offset(bottom_blob, in_pos, scalar_size);
            unsigned char* outbase = top_ptr + permute_output_offset(top_blob, out_pos, 0, scalar_size);

            if (in_elempack == out_elempack && in_elempack > 1)
            {
                for (int j = 0; j < inner_ext; j += out_elempack)
                {
                    const size_t src_offset = (size_t)(j / in_elempack) * bottom_blob.cstep * in_elemsize;
                    const size_t dst_offset = (size_t)(j / out_elempack) * out_cstep_bytes;
                    memcpy(outbase + dst_offset, inbase + src_offset, out_elemsize);
                }
            }
            else
            {
                const size_t spatial_offset = permute_input_offset(bottom_blob, in_pos, scalar_size);
                for (int j = 0; j < inner_ext; j++)
                {
                    const size_t src_offset = (size_t)(j / in_elempack) * bottom_blob.cstep * in_elemsize + (j % in_elempack) * scalar_size;
                    const size_t dst_offset = (size_t)(j / out_elempack) * out_cstep_bytes + (j % out_elempack) * scalar_size;
                    memcpy(outbase + dst_offset, bottom_ptr + spatial_offset + src_offset, scalar_size);
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
