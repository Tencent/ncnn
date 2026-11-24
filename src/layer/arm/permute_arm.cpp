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
#define __AVX__  1
#include "permute_arm_bf16s_fp16s.h"
#undef __AVX__
#undef __SSE2__
#undef __m128i

} // namespace permute_arm_bf16s_fp16s_kernel

int permute_arm_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, int order_type, int nT)
{
    using namespace permute_arm_bf16s_fp16s_kernel;

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

    if (scalar_size == 2)
        return permute_arm_bf16s_fp16s(bottom_blob, top_blob, order_type, opt.num_threads);

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

} // namespace ncnn
