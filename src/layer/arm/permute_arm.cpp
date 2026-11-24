// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "permute_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include <algorithm>
#include <limits.h>
#include <string.h>

#include "arm_usability.h"
#include "cpu.h"
#include "permute_arm_sse2.h"
#include "permute_arm_sse2_u16.h"

namespace ncnn {

namespace permute_arm_fp32 {

#define __SSE2__ 1
#include "permute_arm_fp32.h"
#undef __SSE2__

} // namespace permute_arm_fp32

namespace permute_arm_bf16s_fp16s_kernel {

#define __m128i  permute_u16_m128i
#define __SSE2__ 1
#include "permute_arm_bf16s_fp16s.h"
#undef __SSE2__
#undef __m128i

} // namespace permute_arm_bf16s_fp16s_kernel

int permute_arm_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, int nT)
{
    using namespace permute_arm_bf16s_fp16s_kernel;

    if (bottom_blob.elempack > 4 || top_blob.elempack > 4)
        return -1;

    if (bottom_blob.dims == 2)
    {
        permute2d_bf16s_fp16s(bottom_blob, top_blob, nT);
        return 0;
    }

    if (bottom_blob.dims == 3)
    {
        switch (order_type)
        {
        case 1: permute3d_hwc_bf16s_fp16s(bottom_blob, top_blob, nT); break;
        case 2: permute3d_wch_bf16s_fp16s(bottom_blob, top_blob, nT); break;
        case 3: permute3d_cwh_bf16s_fp16s(bottom_blob, top_blob, nT); break;
        case 4: permute3d_hcw_bf16s_fp16s(bottom_blob, top_blob, nT); break;
        case 5: permute3d_chw_bf16s_fp16s(bottom_blob, top_blob, nT); break;
        }
        return 0;
    }

    switch (order_type)
    {
    case 1: permute4d_hwdc_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 2: permute4d_wdhc_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 3: permute4d_dwhc_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 4: permute4d_hdwc_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 5: permute4d_dhwc_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 6: permute4d_whcd_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 7: permute4d_hwcd_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 8: permute4d_wchd_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 9: permute4d_cwhd_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 10: permute4d_hcwd_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 11: permute4d_chwd_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 12: permute4d_wdch_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 13: permute4d_dwch_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 14: permute4d_wcdh_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 15: permute4d_cwdh_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 16: permute4d_dcwh_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 17: permute4d_cdwh_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 18: permute4d_hdcw_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 19: permute4d_dhcw_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 20: permute4d_hcdw_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 21: permute4d_chdw_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 22: permute4d_dchw_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    case 23: permute4d_cdhw_bf16s_fp16s(bottom_blob, top_blob, nT); break;
    }

    return 0;
}

static const int permute_axes_2d[2][2] = {
    {0, 1},
    {1, 0}
};

static const int permute_axes_3d[6][3] = {
    {0, 1, 3},
    {1, 0, 3},
    {0, 3, 1},
    {3, 0, 1},
    {1, 3, 0},
    {3, 1, 0}
};

static const int permute_axes_4d[24][4] = {
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

static NCNN_FORCEINLINE uint16x4_t permute_load_c4_u16(const Mat& bottom_blob, int cbase, int h, int w)
{
    const int elempack = bottom_blob.elempack;
    if (elempack == 4)
    {
        const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(cbase / 4) + (h * bottom_blob.w + w) * 4;
        return vld1_u16(ptr);
    }

    uint16x4_t v;
    v = vset_lane_u16(((const unsigned short*)bottom_blob.channel(cbase))[h * bottom_blob.w + w], v, 0);
    v = vset_lane_u16(((const unsigned short*)bottom_blob.channel(cbase + 1))[h * bottom_blob.w + w], v, 1);
    v = vset_lane_u16(((const unsigned short*)bottom_blob.channel(cbase + 2))[h * bottom_blob.w + w], v, 2);
    v = vset_lane_u16(((const unsigned short*)bottom_blob.channel(cbase + 3))[h * bottom_blob.w + w], v, 3);
    return v;
}

static int permute2d_u16(const Mat& bottom_blob, Mat& top_blob)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;

    if (elempack == 4 && out_elempack == 4)
    {
        if (w % 4 != 0)
            return -1;

        const int in_h = h * 4;
        const int out_w = in_h;
        const unsigned short* ptr = (const unsigned short*)bottom_blob.data;
        unsigned short* outptr = (unsigned short*)top_blob.data;

        for (int y = 0; y < w; y += 4)
        {
            for (int x = 0; x < in_h; x += 4)
            {
                uint16x4_t _r0 = vld1_u16(ptr + (x / 4) * w * 4 + (y + 0) * 4);
                uint16x4_t _r1 = vld1_u16(ptr + (x / 4) * w * 4 + (y + 1) * 4);
                uint16x4_t _r2 = vld1_u16(ptr + (x / 4) * w * 4 + (y + 2) * 4);
                uint16x4_t _r3 = vld1_u16(ptr + (x / 4) * w * 4 + (y + 3) * 4);
                transpose4x4_u16(_r0, _r1, _r2, _r3);

                unsigned short* out = outptr + (y / 4) * out_w * 4 + x * 4;
                vst1_u16(out + 0, _r0);
                vst1_u16(out + 4, _r1);
                vst1_u16(out + 8, _r2);
                vst1_u16(out + 12, _r3);
            }
        }

        return 0;
    }

    if (elempack != 1)
        return -1;

    const int out_w = h;
    const int out_h = w;

    const unsigned short* ptr = (const unsigned short*)bottom_blob.data;
    unsigned short* outptr = (unsigned short*)top_blob.data;

    if (out_elempack == 1)
    {
        if (h % 4 != 0 || w % 4 != 0)
            return -1;

        for (int y = 0; y < h; y += 4)
        {
            for (int x = 0; x < w; x += 4)
            {
                uint16x4_t _r0 = vld1_u16(ptr + (y + 0) * w + x);
                uint16x4_t _r1 = vld1_u16(ptr + (y + 1) * w + x);
                uint16x4_t _r2 = vld1_u16(ptr + (y + 2) * w + x);
                uint16x4_t _r3 = vld1_u16(ptr + (y + 3) * w + x);
                transpose4x4_u16(_r0, _r1, _r2, _r3);

                vst1_u16(outptr + (x + 0) * out_w + y, _r0);
                vst1_u16(outptr + (x + 1) * out_w + y, _r1);
                vst1_u16(outptr + (x + 2) * out_w + y, _r2);
                vst1_u16(outptr + (x + 3) * out_w + y, _r3);
            }
        }

        return 0;
    }

    if (out_elempack == 4)
    {
        if (h % 4 != 0 || w % 4 != 0)
            return -1;

        for (int y = 0; y < h; y += 4)
        {
            for (int x = 0; x < w; x += 4)
            {
                uint16x4_t _r0 = vld1_u16(ptr + (y + 0) * w + x);
                uint16x4_t _r1 = vld1_u16(ptr + (y + 1) * w + x);
                uint16x4_t _r2 = vld1_u16(ptr + (y + 2) * w + x);
                uint16x4_t _r3 = vld1_u16(ptr + (y + 3) * w + x);
                transpose4x4_u16(_r0, _r1, _r2, _r3);
                transpose4x4_u16(_r0, _r1, _r2, _r3);

                unsigned short* out = outptr + (x / 4) * out_w * 4 + y * 4;
                vst1_u16(out + 0, _r0);
                vst1_u16(out + 4, _r1);
                vst1_u16(out + 8, _r2);
                vst1_u16(out + 12, _r3);
            }
        }

        return 0;
    }

    return -1;
}

static int permute2d(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r0, _r1, _r2, _r3);

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
                transpose4x4_ps(_r0, _r1, _r2, _r3);

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
                transpose4x4_ps(_r.val[0], _r.val[1], _r.val[2], _r.val[3]);
                transpose4x4_ps(_r.val[0], _r.val[1], _r.val[2], _r.val[3]);

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

static int permute3d(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt);

static int permute4d(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
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
        return permute3d(bottom_blob_3d, top_blob_3d, 2, opt);
    }

    if (order_type == 9)
    {
        Mat bottom_blob_3d(w * h, d, channels, bottom_blob.data, elemsize, elempack);
        bottom_blob_3d.cstep = bottom_blob.cstep;
        Mat top_blob_3d(channels * elempack, w * h, top_blob.c, top_blob.data, out_elemsize, out_elempack);
        top_blob_3d.cstep = top_blob.cstep;
        return permute3d(bottom_blob_3d, top_blob_3d, 3, opt);
    }

    if (order_type == 18)
    {
        Mat bottom_blob_3d(w, h * d, channels, bottom_blob.data, elemsize, elempack);
        bottom_blob_3d.cstep = bottom_blob.cstep;
        Mat top_blob_3d(h * d, channels * elempack, top_blob.c, top_blob.data, out_elemsize, out_elempack);
        top_blob_3d.cstep = top_blob.cstep;
        return permute3d(bottom_blob_3d, top_blob_3d, 4, opt);
    }

    Mat bottom_blob_3d(w, h * d, channels, bottom_blob.data, elemsize, elempack);
    bottom_blob_3d.cstep = bottom_blob.cstep;
    Mat top_blob_3d(channels * elempack, h * d, top_blob.c, top_blob.data, out_elemsize, out_elempack);
    top_blob_3d.cstep = top_blob.cstep;
    return permute3d(bottom_blob_3d, top_blob_3d, 5, opt);
}

static int permute3d(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const int logical_channels = channels * elempack;
    const int outw = top_blob.w;
    const int elembits = bottom_blob.elembits();

    if (elembits == 16 && order_type == 1 && elempack == 1 && out_elempack == 1 && w % 4 == 0 && h % 4 == 0)
    {
        const int cols = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q);
            unsigned short* outptr = (unsigned short*)top_blob.channel(q);

            for (int y = 0; y < h; y += 4)
            {
                for (int x = 0; x < cols; x += 4)
                {
                    uint16x4_t _r0 = vld1_u16(ptr + (y + 0) * cols + x);
                    uint16x4_t _r1 = vld1_u16(ptr + (y + 1) * cols + x);
                    uint16x4_t _r2 = vld1_u16(ptr + (y + 2) * cols + x);
                    uint16x4_t _r3 = vld1_u16(ptr + (y + 3) * cols + x);
                    transpose4x4_u16(_r0, _r1, _r2, _r3);

                    vst1_u16(outptr + (x + 0) * h + y, _r0);
                    vst1_u16(outptr + (x + 1) * h + y, _r1);
                    vst1_u16(outptr + (x + 2) * h + y, _r2);
                    vst1_u16(outptr + (x + 3) * h + y, _r3);
                }
            }
        }

        return 0;
    }

    if (elembits == 16 && opt.num_threads == 1 && (order_type == 2 || order_type == 3) && (elempack == 1 || elempack == 4) && h % 4 == 0 && logical_channels % 4 == 0 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int c = 0; c < logical_channels; c += 4)
        {
            for (int y = 0; y < h; y += 4)
            {
                for (int x = 0; x < w; x++)
                {
                    uint16x4_t _r0 = permute_load_c4_u16(bottom_blob, c, y + 0, x);
                    uint16x4_t _r1 = permute_load_c4_u16(bottom_blob, c, y + 1, x);
                    uint16x4_t _r2 = permute_load_c4_u16(bottom_blob, c, y + 2, x);
                    uint16x4_t _r3 = permute_load_c4_u16(bottom_blob, c, y + 3, x);
                    transpose4x4_u16(_r0, _r1, _r2, _r3);

                    unsigned short* outbase = (unsigned short*)top_blob.channel(y / 4);
                    for (int l = 0; l < 4; l++)
                    {
                        uint16x4_t _v = l == 0 ? _r0 : l == 1 ? _r1
                                        : l == 2   ? _r2
                                        : _r3;
                        unsigned short* outptr = order_type == 2 ? outbase + ((c + l) * top_blob.w + x) * 4 : outbase + (x * top_blob.w + c + l) * 4;
                        vst1_u16(outptr, _v);
                    }
                }
            }
        }

        return 0;
    }

    if (elembits == 16 && opt.num_threads == 1 && (order_type == 4 || order_type == 5) && elempack == 4 && out_elempack == 4 && w % 4 == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int x = 0; x < w; x += 4)
        {
            for (int y = 0; y < h; y++)
            {
                for (int c = 0; c < logical_channels; c += 4)
                {
                    const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(c / 4);
                    uint16x4x4_t _v = vld4_u16(ptr + (y * w + x) * 4);

                    unsigned short* outbase = (unsigned short*)top_blob.channel(x / 4);
                    for (int l = 0; l < 4; l++)
                    {
                        unsigned short* outptr = order_type == 4 ? outbase + ((c + l) * top_blob.w + y) * 4 : outbase + (y * top_blob.w + c + l) * 4;
                        vst1_u16(outptr, _v.val[l]);
                    }
                }
            }
        }

        return 0;
    }

    if (elembits == 16 && opt.num_threads == 1 && (order_type == 4 || order_type == 5) && (elempack == 1 || elempack == 4) && w % 4 == 0 && logical_channels % 4 == 0 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int x = 0; x < w; x += 4)
        {
            for (int y = 0; y < h; y++)
            {
                for (int c = 0; c < logical_channels; c += 4)
                {
                    uint16x4_t _r0 = permute_load_c4_u16(bottom_blob, c, y, x + 0);
                    uint16x4_t _r1 = permute_load_c4_u16(bottom_blob, c, y, x + 1);
                    uint16x4_t _r2 = permute_load_c4_u16(bottom_blob, c, y, x + 2);
                    uint16x4_t _r3 = permute_load_c4_u16(bottom_blob, c, y, x + 3);
                    transpose4x4_u16(_r0, _r1, _r2, _r3);

                    unsigned short* outbase = (unsigned short*)top_blob.channel(x / 4);
                    for (int l = 0; l < 4; l++)
                    {
                        uint16x4_t _v = l == 0 ? _r0 : l == 1 ? _r1
                                        : l == 2   ? _r2
                                        : _r3;
                        unsigned short* outptr = order_type == 4 ? outbase + ((c + l) * top_blob.w + y) * 4 : outbase + (y * top_blob.w + c + l) * 4;
                        vst1_u16(outptr, _v);
                    }
                }
            }
        }

        return 0;
    }

    if (elembits == 16 && order_type == 1 && elempack == 4 && out_elempack == 4 && w % 4 == 0 && h % 4 == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob.channel(q);
            unsigned short* outptr = (unsigned short*)top_blob.channel(q);

            for (int y = 0; y < h; y += 4)
            {
                for (int x = 0; x < w; x += 4)
                {
                    uint16x4_t _r[4][4];
                    for (int i = 0; i < 4; i++)
                    {
                        uint16x4x4_t _v = vld4_u16(ptr + (y + i) * w * 4 + x * 4);
                        _r[i][0] = _v.val[0];
                        _r[i][1] = _v.val[1];
                        _r[i][2] = _v.val[2];
                        _r[i][3] = _v.val[3];
                    }

                    uint16x4_t _t[4][4];
                    for (int c = 0; c < 4; c++)
                    {
                        _t[c][0] = _r[0][c];
                        _t[c][1] = _r[1][c];
                        _t[c][2] = _r[2][c];
                        _t[c][3] = _r[3][c];
                        transpose4x4_u16(_t[c][0], _t[c][1], _t[c][2], _t[c][3]);
                    }

                    for (int j = 0; j < 4; j++)
                    {
                        uint16x4_t _u0 = _t[0][j];
                        uint16x4_t _u1 = _t[1][j];
                        uint16x4_t _u2 = _t[2][j];
                        uint16x4_t _u3 = _t[3][j];
                        transpose4x4_u16(_u0, _u1, _u2, _u3);

                        unsigned short* out = outptr + ((x + j) * outw + y) * 4;
                        vst1_u16(out, _u0);
                        vst1_u16(out + 4, _u1);
                        vst1_u16(out + 8, _u2);
                        vst1_u16(out + 12, _u3);
                    }
                }
            }
        }

        return 0;
    }

    if (elembits == 32 && order_type == 1 && elempack == out_elempack && (elempack == 1 || elempack == 4) && w % 4 == 0 && h % 4 == 0)
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
                        transpose4x4_ps(_r0, _r1, _r2, _r3);

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
                            transpose4x4_ps(_t[c][0], _t[c][1], _t[c][2], _t[c][3]);
                        }

                        for (int j = 0; j < 4; j++)
                        {
                            float32x4_t _u0 = _t[0][j];
                            float32x4_t _u1 = _t[1][j];
                            float32x4_t _u2 = _t[2][j];
                            float32x4_t _u3 = _t[3][j];
                            transpose4x4_ps(_u0, _u1, _u2, _u3);

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

    if (elembits == 32 && (order_type == 2 || order_type == 3) && (elempack == 1 || elempack == 4) && h % 4 == 0 && logical_channels % 4 == 0 && out_elempack == 4)
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
                    transpose4x4_ps(_r0, _r1, _r2, _r3);

                    float* outbase = (float*)top_blob.channel(y / 4);
                    for (int l = 0; l < 4; l++)
                    {
                        float32x4_t _v = l == 0 ? _r0 : l == 1 ? _r1
                                         : l == 2   ? _r2
                                         : _r3;
                        float* outptr = order_type == 2 ? outbase + ((c + l) * top_blob.w + x) * 4 : outbase + (x * top_blob.w + c + l) * 4;
                        vst1q_f32(outptr, _v);
                    }
                }
            }
        }

        return 0;
    }

    if (elembits == 32 && (order_type == 4 || order_type == 5) && elempack == 4 && out_elempack == 4 && w % 4 == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int x = 0; x < w; x += 4)
        {
            for (int y = 0; y < h; y++)
            {
                for (int c = 0; c < logical_channels; c += 4)
                {
                    const float* ptr = (const float*)bottom_blob.channel(c / 4);
                    float32x4x4_t _v = vld4q_f32(ptr + (y * w + x) * 4);

                    float* outbase = (float*)top_blob.channel(x / 4);
                    for (int l = 0; l < 4; l++)
                    {
                        float* outptr = order_type == 4 ? outbase + ((c + l) * top_blob.w + y) * 4 : outbase + (y * top_blob.w + c + l) * 4;
                        vst1q_f32(outptr, _v.val[l]);
                    }
                }
            }
        }

        return 0;
    }

    if (elembits == 32 && (order_type == 4 || order_type == 5) && (elempack == 1 || elempack == 4) && w % 4 == 0 && logical_channels % 4 == 0 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int x = 0; x < w; x += 4)
        {
            for (int y = 0; y < h; y++)
            {
                for (int c = 0; c < logical_channels; c += 4)
                {
                    float32x4_t _r0 = permute_load_c4(bottom_blob, c, y, x + 0);
                    float32x4_t _r1 = permute_load_c4(bottom_blob, c, y, x + 1);
                    float32x4_t _r2 = permute_load_c4(bottom_blob, c, y, x + 2);
                    float32x4_t _r3 = permute_load_c4(bottom_blob, c, y, x + 3);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);

                    float* outbase = (float*)top_blob.channel(x / 4);
                    for (int l = 0; l < 4; l++)
                    {
                        float32x4_t _v = l == 0 ? _r0 : l == 1 ? _r1
                                         : l == 2   ? _r2
                                         : _r3;
                        float* outptr = order_type == 4 ? outbase + ((c + l) * top_blob.w + y) * 4 : outbase + (y * top_blob.w + c + l) * 4;
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

static void permute_axes(const Mat& bottom_blob, Mat& top_blob, const int* axes, int dims, const Option& opt)
{
    const int in_w = bottom_blob.w;
    const int in_h = bottom_blob.h;
    const int in_d = bottom_blob.d;
    const int in_c = bottom_blob.c;
    const int in_elempack = bottom_blob.elempack;
    const int scalar_size = bottom_blob.elembits() / 8;

    int in_shape[4] = {in_w, in_h, in_d, in_c * in_elempack};
    if (dims == 2)
        in_shape[1] *= in_elempack;

    int out_shape[4] = {1, 1, 1, 1};
    for (int i = 0; i < dims; i++)
    {
        out_shape[i] = in_shape[axes[i]];
    }

    const int last_axis = axes[dims - 1];
    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        const int input_last_axis = dims == 2 ? 1 : 3;
        if (last_axis == input_last_axis)
        {
            out_elempack = in_elempack;
        }
        else
        {
            out_elempack = out_shape[dims - 1] % 4 == 0 ? 4 : 1;
        }
    }

    int outer_size = 1;
    for (int i = 0; i < dims - 1; i++)
    {
        outer_size *= out_shape[i];
    }

    const int inner_size = out_shape[dims - 1];
    const unsigned char* bottom_ptr = (const unsigned char*)bottom_blob.data;
    unsigned char* top_ptr = (unsigned char*)top_blob.data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < outer_size; r++)
    {
        int out_pos[4] = {0, 0, 0, 0};
        int in_pos[4] = {0, 0, 0, 0};

        int v = r;
        for (int i = 0; i < dims - 1; i++)
        {
            out_pos[i] = v % out_shape[i];
            v /= out_shape[i];
            in_pos[axes[i]] = out_pos[i];
        }

        in_pos[last_axis] = 0;

        const size_t in_elemsize = (size_t)scalar_size * in_elempack;
        const size_t out_elemsize = (size_t)scalar_size * out_elempack;
        const size_t out_cstep_bytes = top_blob.cstep * out_elemsize;

        if (dims == 2)
        {
            for (int j = 0; j < inner_size; j++)
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
                    for (int j = 0; j < inner_size; j++)
                    {
                        *(float*)outptr = *(const float*)inptr;
                        inptr += in_step;
                        outptr += out_cstep_bytes;
                    }
                }
                else
                {
                    for (int j = 0; j < inner_size; j++)
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
                for (; j + 3 < inner_size; j += 4)
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
                for (; j < inner_size; j++)
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
                for (int j = 0; j < inner_size; j += out_elempack)
                {
                    const size_t src_offset = (size_t)(j / in_elempack) * bottom_blob.cstep * in_elemsize;
                    const size_t dst_offset = (size_t)(j / out_elempack) * out_cstep_bytes;
                    memcpy(outbase + dst_offset, inbase + src_offset, out_elemsize);
                }
            }
            else
            {
                const size_t spatial_offset = permute_input_offset(bottom_blob, in_pos, scalar_size);
                for (int j = 0; j < inner_size; j++)
                {
                    const size_t src_offset = (size_t)(j / in_elempack) * bottom_blob.cstep * in_elemsize + (j % in_elempack) * scalar_size;
                    const size_t dst_offset = (size_t)(j / out_elempack) * out_cstep_bytes + (j % out_elempack) * scalar_size;
                    memcpy(outbase + dst_offset, bottom_ptr + spatial_offset + src_offset, scalar_size);
                }
            }
        }
    }
}

template<int AXIS, int A0, int A1, int A2, int A3>
static NCNN_FORCEINLINE int permute_input_coord(int x, int y, int z, int q)
{
    return A0 == AXIS ? x : A1 == AXIS ? y
           : A2 == AXIS   ? z
           : q;
}

template<typename T, int A0, int A1, int A2, int A3>
static void permute4d_pack1(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outd = top_blob.d;
    const int outc = top_blob.c;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < outc * outd * outh; i++)
    {
        const int q = i / (outd * outh);
        const int z = i / outh % outd;
        const int y = i % outh;
        T* outptr = (T*)top_blob.channel(q) + (z * outh + y) * outw;

        for (int x = 0; x < outw; x++)
        {
            const int iw = permute_input_coord<0, A0, A1, A2, A3>(x, y, z, q);
            const int ih = permute_input_coord<1, A0, A1, A2, A3>(x, y, z, q);
            const int id = permute_input_coord<2, A0, A1, A2, A3>(x, y, z, q);
            const int ic = permute_input_coord<3, A0, A1, A2, A3>(x, y, z, q);

            const T* ptr = (const T*)bottom_blob.channel(ic);
            outptr[x] = ptr[((size_t)id * h + ih) * w + iw];
        }
    }
}

static void permute4d_pack1_fp32(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;

    if (order_type == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int z = 0; z < d; z++)
            {
                const Mat m = bottom_blob.channel(q).depth(z);

                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        *outptr++ = m.row(j)[i];
                    }
                }
            }
        }
    }

    if (order_type == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int z = 0; z < h; z++)
            {
                for (int i = 0; i < d; i++)
                {
                    const float* ptr = bottom_blob.channel(q).depth(i).row(z);

                    for (int j = 0; j < w; j++)
                    {
                        *outptr++ = ptr[j];
                    }
                }
            }
        }
    }

    if (order_type == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int z = 0; z < h; z++)
            {
                for (int i = 0; i < w; i++)
                {
                    for (int j = 0; j < d; j++)
                    {
                        *outptr++ = m.depth(j).row(z)[i];
                    }
                }
            }
        }
    }

    if (order_type == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int z = 0; z < w; z++)
            {
                for (int i = 0; i < d; i++)
                {
                    const Mat m = bottom_blob.channel(q).depth(i);

                    for (int j = 0; j < h; j++)
                    {
                        *outptr++ = m.row(j)[z];
                    }
                }
            }
        }
    }

    if (order_type == 5)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int z = 0; z < w; z++)
            {
                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < d; j++)
                    {
                        *outptr++ = m.depth(j).row(i)[z];
                    }
                }
            }
        }
    }
}

static void permute3d_hwc(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if __ARM_NEON && __aarch64__
    const int scalar_size = bottom_blob.elembits() / 8;
    if ((scalar_size == 4 || scalar_size == 2) && permute3d(bottom_blob, top_blob, 1, opt) == 0)
        return;
#endif // __ARM_NEON && __aarch64__
    permute_axes(bottom_blob, top_blob, permute_axes_3d[1], 3, opt);
}

static void permute3d_wch(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if __ARM_NEON && __aarch64__
    const int scalar_size = bottom_blob.elembits() / 8;
    if ((scalar_size == 4 || scalar_size == 2) && permute3d(bottom_blob, top_blob, 2, opt) == 0)
        return;
#endif // __ARM_NEON && __aarch64__
    permute_axes(bottom_blob, top_blob, permute_axes_3d[2], 3, opt);
}

static void permute3d_cwh(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if __ARM_NEON && __aarch64__
    const int scalar_size = bottom_blob.elembits() / 8;
    if ((scalar_size == 4 || scalar_size == 2) && permute3d(bottom_blob, top_blob, 3, opt) == 0)
        return;
#endif // __ARM_NEON && __aarch64__
    permute_axes(bottom_blob, top_blob, permute_axes_3d[3], 3, opt);
}

static void permute3d_hcw(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if __ARM_NEON && __aarch64__
    const int scalar_size = bottom_blob.elembits() / 8;
    if ((scalar_size == 4 || scalar_size == 2) && permute3d(bottom_blob, top_blob, 4, opt) == 0)
        return;
#endif // __ARM_NEON && __aarch64__
    permute_axes(bottom_blob, top_blob, permute_axes_3d[4], 3, opt);
}

static void permute3d_chw(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if __ARM_NEON && __aarch64__
    const int scalar_size = bottom_blob.elembits() / 8;
    if ((scalar_size == 4 || scalar_size == 2) && permute3d(bottom_blob, top_blob, 5, opt) == 0)
        return;
#endif // __ARM_NEON && __aarch64__
    permute_axes(bottom_blob, top_blob, permute_axes_3d[5], 3, opt);
}

static void permute4d_hwdc(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elembits() == 32 && bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        permute4d_pack1_fp32(bottom_blob, top_blob, 1, opt);
        return;
    }
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 1, 0, 2, 3>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 1, 0, 2, 3>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[1], 4, opt);
}

static void permute4d_wdhc(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elembits() == 32 && bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        permute4d_pack1_fp32(bottom_blob, top_blob, 2, opt);
        return;
    }
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 0, 2, 1, 3>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 0, 2, 1, 3>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[2], 4, opt);
}

static void permute4d_dwhc(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elembits() == 32 && bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        permute4d_pack1_fp32(bottom_blob, top_blob, 3, opt);
        return;
    }
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 2, 0, 1, 3>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 2, 0, 1, 3>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[3], 4, opt);
}

static void permute4d_hdwc(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elembits() == 32 && bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        permute4d_pack1_fp32(bottom_blob, top_blob, 4, opt);
        return;
    }
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 1, 2, 0, 3>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 1, 2, 0, 3>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[4], 4, opt);
}

static void permute4d_dhwc(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elembits() == 32 && bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        permute4d_pack1_fp32(bottom_blob, top_blob, 5, opt);
        return;
    }
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 2, 1, 0, 3>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 2, 1, 0, 3>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[5], 4, opt);
}

static void permute4d_whcd(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 0, 1, 3, 2>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 0, 1, 3, 2>(bottom_blob, top_blob, opt);
        return;
    }
#if __ARM_NEON && __aarch64__
    const int scalar_size = bottom_blob.elembits() / 8;
    if (scalar_size == 4 && permute4d(bottom_blob, top_blob, 6, opt) == 0)
        return;
#endif // __ARM_NEON && __aarch64__
    permute_axes(bottom_blob, top_blob, permute_axes_4d[6], 4, opt);
}

static void permute4d_hwcd(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 1, 0, 3, 2>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 1, 0, 3, 2>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[7], 4, opt);
}

static void permute4d_wchd(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 0, 3, 1, 2>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 0, 3, 1, 2>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[8], 4, opt);
}

static void permute4d_cwhd(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 3, 0, 1, 2>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 3, 0, 1, 2>(bottom_blob, top_blob, opt);
        return;
    }
#if __ARM_NEON && __aarch64__
    const int scalar_size = bottom_blob.elembits() / 8;
    if (scalar_size == 4 && permute4d(bottom_blob, top_blob, 9, opt) == 0)
        return;
#endif // __ARM_NEON && __aarch64__
    permute_axes(bottom_blob, top_blob, permute_axes_4d[9], 4, opt);
}

static void permute4d_hcwd(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 1, 3, 0, 2>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 1, 3, 0, 2>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[10], 4, opt);
}

static void permute4d_chwd(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 3, 1, 0, 2>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 3, 1, 0, 2>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[11], 4, opt);
}

static void permute4d_wdch(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 0, 2, 3, 1>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 0, 2, 3, 1>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[12], 4, opt);
}

static void permute4d_dwch(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 2, 0, 3, 1>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 2, 0, 3, 1>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[13], 4, opt);
}

static void permute4d_wcdh(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 0, 3, 2, 1>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 0, 3, 2, 1>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[14], 4, opt);
}

static void permute4d_cwdh(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 3, 0, 2, 1>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 3, 0, 2, 1>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[15], 4, opt);
}

static void permute4d_dcwh(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 2, 3, 0, 1>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 2, 3, 0, 1>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[16], 4, opt);
}

static void permute4d_cdwh(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 3, 2, 0, 1>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 3, 2, 0, 1>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[17], 4, opt);
}

static void permute4d_hdcw(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 1, 2, 3, 0>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 1, 2, 3, 0>(bottom_blob, top_blob, opt);
        return;
    }
#if __ARM_NEON && __aarch64__
    const int scalar_size = bottom_blob.elembits() / 8;
    if (scalar_size == 4 && permute4d(bottom_blob, top_blob, 18, opt) == 0)
        return;
#endif // __ARM_NEON && __aarch64__
    permute_axes(bottom_blob, top_blob, permute_axes_4d[18], 4, opt);
}

static void permute4d_dhcw(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 2, 1, 3, 0>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 2, 1, 3, 0>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[19], 4, opt);
}

static void permute4d_hcdw(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 1, 3, 2, 0>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 1, 3, 2, 0>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[20], 4, opt);
}

static void permute4d_chdw(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 3, 1, 2, 0>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 3, 1, 2, 0>(bottom_blob, top_blob, opt);
        return;
    }
#if __ARM_NEON && __aarch64__
    const int scalar_size = bottom_blob.elembits() / 8;
    if (scalar_size == 4 && permute4d(bottom_blob, top_blob, 21, opt) == 0)
        return;
#endif // __ARM_NEON && __aarch64__
    permute_axes(bottom_blob, top_blob, permute_axes_4d[21], 4, opt);
}

static void permute4d_dchw(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 2, 3, 1, 0>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 2, 3, 1, 0>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[22], 4, opt);
}

static void permute4d_cdhw(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    if (bottom_blob.elempack == 1 && top_blob.elempack == 1)
    {
        if (bottom_blob.elembits() == 32)
            permute4d_pack1<float, 3, 2, 1, 0>(bottom_blob, top_blob, opt);
        else
            permute4d_pack1<unsigned short, 3, 2, 1, 0>(bottom_blob, top_blob, opt);
        return;
    }
    permute_axes(bottom_blob, top_blob, permute_axes_4d[23], 4, opt);
}

Permute_arm::Permute_arm()
{
#if __ARM_NEON
    support_packing = true;
    support_any_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Permute_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;

    if (dims == 1 || order_type == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if ((dims == 2 && order_type != 1) || (dims == 3 && (order_type < 1 || order_type > 5)) || (dims == 4 && (order_type < 1 || order_type > 23)))
        return -1;

    const int scalar_bits = bottom_blob.elembits();
    if (scalar_bits != 16 && scalar_bits != 32)
        return -100;

    const int scalar_size = scalar_bits / 8;
    const int in_w = bottom_blob.w;
    const int in_h = bottom_blob.h;
    const int in_d = bottom_blob.d;
    const int in_c = bottom_blob.c;
    const int in_elempack = bottom_blob.elempack;

    int in_shape[4] = {in_w, in_h, in_d, in_c * in_elempack};
    if (dims == 2)
        in_shape[1] *= in_elempack;

    const int* axes = dims == 2 ? permute_axes_2d[order_type] : dims == 3 ? permute_axes_3d[order_type]
                      : permute_axes_4d[order_type];

    int out_shape[4] = {1, 1, 1, 1};
    for (int i = 0; i < dims; i++)
    {
        out_shape[i] = in_shape[axes[i]];
    }

    const int last_axis = axes[dims - 1];
    const int input_last_axis = dims == 2 ? 1 : 3;
    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        if (last_axis == input_last_axis)
            out_elempack = in_elempack;
        else
            out_elempack = out_shape[dims - 1] % 4 == 0 ? 4 : 1;
    }

    const size_t out_elemsize = (size_t)scalar_size * out_elempack;
    if (dims == 2)
        top_blob.create(out_shape[0], out_shape[1] / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (dims == 3)
        top_blob.create(out_shape[0], out_shape[1], out_shape[2] / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (dims == 4)
        top_blob.create(out_shape[0], out_shape[1], out_shape[2], out_shape[3] / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    if (scalar_size == 2 && permute_arm_bf16s_fp16s(bottom_blob, top_blob, order_type, opt.num_threads) == 0)
    {
        return 0;
    }

    if (scalar_size == 4)
    {
        if (dims == 2)
        {
            permute_arm_fp32::permute2d(bottom_blob, top_blob, opt.num_threads);
            return 0;
        }

        if (dims == 3)
        {
            switch (order_type)
            {
            case 1: permute_arm_fp32::permute3d_hwc(bottom_blob, top_blob, opt.num_threads); break;
            case 2: permute_arm_fp32::permute3d_wch(bottom_blob, top_blob, opt.num_threads); break;
            case 3: permute_arm_fp32::permute3d_cwh(bottom_blob, top_blob, opt.num_threads); break;
            case 4: permute_arm_fp32::permute3d_hcw(bottom_blob, top_blob, opt.num_threads); break;
            case 5: permute_arm_fp32::permute3d_chw(bottom_blob, top_blob, opt.num_threads); break;
            }
            return 0;
        }

        switch (order_type)
        {
        case 1: permute_arm_fp32::permute4d_hwdc(bottom_blob, top_blob, opt.num_threads); break;
        case 2: permute_arm_fp32::permute4d_wdhc(bottom_blob, top_blob, opt.num_threads); break;
        case 3: permute_arm_fp32::permute4d_dwhc(bottom_blob, top_blob, opt.num_threads); break;
        case 4: permute_arm_fp32::permute4d_hdwc(bottom_blob, top_blob, opt.num_threads); break;
        case 5: permute_arm_fp32::permute4d_dhwc(bottom_blob, top_blob, opt.num_threads); break;
        case 6: permute_arm_fp32::permute4d_whcd(bottom_blob, top_blob, opt.num_threads); break;
        case 7: permute_arm_fp32::permute4d_hwcd(bottom_blob, top_blob, opt.num_threads); break;
        case 8: permute_arm_fp32::permute4d_wchd(bottom_blob, top_blob, opt.num_threads); break;
        case 9: permute_arm_fp32::permute4d_cwhd(bottom_blob, top_blob, opt.num_threads); break;
        case 10: permute_arm_fp32::permute4d_hcwd(bottom_blob, top_blob, opt.num_threads); break;
        case 11: permute_arm_fp32::permute4d_chwd(bottom_blob, top_blob, opt.num_threads); break;
        case 12: permute_arm_fp32::permute4d_wdch(bottom_blob, top_blob, opt.num_threads); break;
        case 13: permute_arm_fp32::permute4d_dwch(bottom_blob, top_blob, opt.num_threads); break;
        case 14: permute_arm_fp32::permute4d_wcdh(bottom_blob, top_blob, opt.num_threads); break;
        case 15: permute_arm_fp32::permute4d_cwdh(bottom_blob, top_blob, opt.num_threads); break;
        case 16: permute_arm_fp32::permute4d_dcwh(bottom_blob, top_blob, opt.num_threads); break;
        case 17: permute_arm_fp32::permute4d_cdwh(bottom_blob, top_blob, opt.num_threads); break;
        case 18: permute_arm_fp32::permute4d_hdcw(bottom_blob, top_blob, opt.num_threads); break;
        case 19: permute_arm_fp32::permute4d_dhcw(bottom_blob, top_blob, opt.num_threads); break;
        case 20: permute_arm_fp32::permute4d_hcdw(bottom_blob, top_blob, opt.num_threads); break;
        case 21: permute_arm_fp32::permute4d_chdw(bottom_blob, top_blob, opt.num_threads); break;
        case 22: permute_arm_fp32::permute4d_dchw(bottom_blob, top_blob, opt.num_threads); break;
        case 23: permute_arm_fp32::permute4d_cdhw(bottom_blob, top_blob, opt.num_threads); break;
        }
        return 0;
    }

    if (dims == 2)
    {
#if __ARM_NEON && __aarch64__
        if (scalar_size == 4 && permute2d(bottom_blob, top_blob, opt) == 0)
            return 0;
        if (scalar_size == 2 && permute2d_u16(bottom_blob, top_blob) == 0)
            return 0;
#endif // __ARM_NEON && __aarch64__
        permute_axes(bottom_blob, top_blob, axes, dims, opt);
        return 0;
    }

    if (dims == 3)
    {
        switch (order_type)
        {
        case 1: permute3d_hwc(bottom_blob, top_blob, opt); break;
        case 2: permute3d_wch(bottom_blob, top_blob, opt); break;
        case 3: permute3d_cwh(bottom_blob, top_blob, opt); break;
        case 4: permute3d_hcw(bottom_blob, top_blob, opt); break;
        case 5: permute3d_chw(bottom_blob, top_blob, opt); break;
        }
        return 0;
    }

    switch (order_type)
    {
    case 1: permute4d_hwdc(bottom_blob, top_blob, opt); break;
    case 2: permute4d_wdhc(bottom_blob, top_blob, opt); break;
    case 3: permute4d_dwhc(bottom_blob, top_blob, opt); break;
    case 4: permute4d_hdwc(bottom_blob, top_blob, opt); break;
    case 5: permute4d_dhwc(bottom_blob, top_blob, opt); break;
    case 6: permute4d_whcd(bottom_blob, top_blob, opt); break;
    case 7: permute4d_hwcd(bottom_blob, top_blob, opt); break;
    case 8: permute4d_wchd(bottom_blob, top_blob, opt); break;
    case 9: permute4d_cwhd(bottom_blob, top_blob, opt); break;
    case 10: permute4d_hcwd(bottom_blob, top_blob, opt); break;
    case 11: permute4d_chwd(bottom_blob, top_blob, opt); break;
    case 12: permute4d_wdch(bottom_blob, top_blob, opt); break;
    case 13: permute4d_dwch(bottom_blob, top_blob, opt); break;
    case 14: permute4d_wcdh(bottom_blob, top_blob, opt); break;
    case 15: permute4d_cwdh(bottom_blob, top_blob, opt); break;
    case 16: permute4d_dcwh(bottom_blob, top_blob, opt); break;
    case 17: permute4d_cdwh(bottom_blob, top_blob, opt); break;
    case 18: permute4d_hdcw(bottom_blob, top_blob, opt); break;
    case 19: permute4d_dhcw(bottom_blob, top_blob, opt); break;
    case 20: permute4d_hcdw(bottom_blob, top_blob, opt); break;
    case 21: permute4d_chdw(bottom_blob, top_blob, opt); break;
    case 22: permute4d_dchw(bottom_blob, top_blob, opt); break;
    case 23: permute4d_cdhw(bottom_blob, top_blob, opt); break;
    }

    return 0;
}

} // namespace ncnn
