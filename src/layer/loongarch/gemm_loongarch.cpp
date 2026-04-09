// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_loongarch.h"

namespace ncnn {

Gemm_loongarch::Gemm_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
}

static int unpack_or_cast_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    if (src.empty())
    {
        dst = src;
        return 0;
    }

    Mat unpacked = src;
    if (src.elempack != 1)
    {
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;

        convert_packing(src, unpacked, 1, opt_unpack);
        if (unpacked.empty())
            return -100;
    }

#if NCNN_BF16
    if (unpacked.elembits() == 16)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = opt.workspace_allocator;

        cast_bfloat16_to_float32(unpacked, dst, opt_cast);
        if (dst.empty())
            return -100;
        return 0;
    }
#endif

    dst = unpacked;
    return 0;
}

static int assign_or_copy_top_blob(const Mat& src, Mat& dst)
{
    if (dst.empty())
    {
        dst = src;
        return 0;
    }

    if (dst.dims != src.dims || dst.w != src.w || dst.h != src.h || dst.d != src.d || dst.c != src.c || dst.elemsize != src.elemsize || dst.elempack != src.elempack)
        return -100;

    memcpy((void*)dst, (const void*)src, src.total() * src.elemsize);
    return 0;
}

int Gemm_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        std::vector<Mat> bottom_blobs_unpacked(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            int ret = unpack_or_cast_to_float32(bottom_blobs[i], bottom_blobs_unpacked[i], opt);
            if (ret != 0)
                return ret;
        }

        Mat A_data_unpacked = A_data;
        Mat B_data_unpacked = B_data;
        Mat C_data_unpacked = C_data;

        if (constantA)
        {
            int ret = unpack_or_cast_to_float32(A_data, A_data_unpacked, opt);
            if (ret != 0)
                return ret;
        }
        if (constantB)
        {
            int ret = unpack_or_cast_to_float32(B_data, B_data_unpacked, opt);
            if (ret != 0)
                return ret;
        }
        if (constantC && constant_broadcast_type_C != -1)
        {
            int ret = unpack_or_cast_to_float32(C_data, C_data_unpacked, opt);
            if (ret != 0)
                return ret;
        }

        Gemm gemm;
        gemm.alpha = alpha;
        gemm.beta = beta;
        gemm.transA = transA;
        gemm.transB = transB;
        gemm.constantA = constantA;
        gemm.constantB = constantB;
        gemm.constantC = constantC;
        gemm.constantM = constantM;
        gemm.constantN = constantN;
        gemm.constantK = constantK;
        gemm.constant_broadcast_type_C = constant_broadcast_type_C;
        gemm.output_N1M = output_N1M;
        gemm.output_elempack = 1;
        gemm.output_elemtype = output_elemtype;
        gemm.output_transpose = output_transpose;
        gemm.int8_scale_term = int8_scale_term;
        gemm.constant_TILE_M = constant_TILE_M;
        gemm.constant_TILE_N = constant_TILE_N;
        gemm.constant_TILE_K = constant_TILE_K;
        gemm.A_data = A_data_unpacked;
        gemm.B_data = B_data_unpacked;
        gemm.C_data = C_data_unpacked;
#if NCNN_INT8
        gemm.A_data_int8_scales = A_data_int8_scales;
        gemm.B_data_int8_scale = B_data_int8_scale;
#endif

        Option opt_int8 = opt;
        opt_int8.use_packing_layout = false;

        std::vector<Mat> top_blobs_unpacked(1);
        int ret = gemm.forward(bottom_blobs_unpacked, top_blobs_unpacked, opt_int8);
        if (ret != 0)
            return ret;

        return assign_or_copy_top_blob(top_blobs_unpacked[0], top_blobs[0]);
    }
#endif

    std::vector<Mat> bottom_blobs_fp32(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        int ret = unpack_or_cast_to_float32(bottom_blobs[i], bottom_blobs_fp32[i], opt);
        if (ret != 0)
            return ret;
    }

    Mat A_data_fp32 = A_data;
    Mat B_data_fp32 = B_data;
    Mat C_data_fp32 = C_data;

    if (constantA)
    {
        int ret = unpack_or_cast_to_float32(A_data, A_data_fp32, opt);
        if (ret != 0)
            return ret;
    }
    if (constantB)
    {
        int ret = unpack_or_cast_to_float32(B_data, B_data_fp32, opt);
        if (ret != 0)
            return ret;
    }
    if (constantC && constant_broadcast_type_C != -1)
    {
        int ret = unpack_or_cast_to_float32(C_data, C_data_fp32, opt);
        if (ret != 0)
            return ret;
    }

    const Mat& A0 = constantA ? A_data_fp32 : bottom_blobs_fp32[0];
    const Mat& B0 = constantB ? B_data_fp32 : constantA ? bottom_blobs_fp32[0] : bottom_blobs_fp32[1];

    const int M = transA == 0 ? (A0.dims == 3 ? A0.c : A0.h) : A0.w;
    const int N = transB == 0 ? B0.w : (B0.dims == 3 ? B0.c : B0.h);

    int out_elempack = 1;
#if __loongarch_sx
    if (output_elempack == 0 && opt.use_packing_layout)
    {
        const int outh = output_transpose ? N : M;
        out_elempack = outh % 4 == 0 ? 4 : 1;
    }
    if (output_elempack == 4)
        out_elempack = 4;
#endif // __loongarch_sx

    Gemm gemm;
    gemm.alpha = alpha;
    gemm.beta = beta;
    gemm.transA = transA;
    gemm.transB = transB;
    gemm.constantA = constantA;
    gemm.constantB = constantB;
    gemm.constantC = constantC;
    gemm.constantM = constantM;
    gemm.constantN = constantN;
    gemm.constantK = constantK;
    gemm.constant_broadcast_type_C = constant_broadcast_type_C;
    gemm.output_N1M = output_N1M;
    gemm.output_elempack = 1;
    gemm.output_elemtype = 1;
    gemm.output_transpose = output_transpose;
    gemm.int8_scale_term = 0;
    gemm.constant_TILE_M = constant_TILE_M;
    gemm.constant_TILE_N = constant_TILE_N;
    gemm.constant_TILE_K = constant_TILE_K;
    gemm.A_data = A_data_fp32;
    gemm.B_data = B_data_fp32;
    gemm.C_data = C_data_fp32;

    std::vector<Mat> top_blobs_unpacked(1);
    int ret = gemm.forward(bottom_blobs_fp32, top_blobs_unpacked, opt);
    if (ret != 0)
        return ret;

    Mat top_blob_final;
    if (out_elempack == 4)
    {
        convert_packing(top_blobs_unpacked[0], top_blob_final, 4, opt);
        if (top_blob_final.empty())
            return -100;
    }
    else
    {
        top_blob_final = top_blobs_unpacked[0];
    }

    return assign_or_copy_top_blob(top_blob_final, top_blobs[0]);
}

} // namespace ncnn
