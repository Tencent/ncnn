// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GEMM_POWERPC_SPU_H
#define LAYER_GEMM_POWERPC_SPU_H

#include "mat.h"

namespace ncnn {

// pack4 gemm on the ps3 SPEs. A and BT are elempack 4, top is elempack 4 with
// top(i, j) at ((i/4)*N + j)*4 + i%4, matching gemm_transB_pack4_powerpc.
// a_stride_bytes is the byte distance from the start of one K-segment of a
// packed A row to the next: the K vec4 of a row are split into contiguous
// segments of a_stride_bytes/16 vec4 each that tile the K axis, and segment s
// starts s*a_stride_bytes after segment 0. 0 means the row is one contiguous
// run of K vec4, which is what every existing caller passes. K must be a
// multiple of the segment length.
// returns 0 when the SPEs produced the result, -1 when they are unavailable so
// the caller can fall back to the ppe kernel.
int gemm_transB_pack4_spu(const Mat& A, const Mat& BT, Mat& top_blob, float alpha, float beta, float c0, unsigned int a_stride_bytes);

} // namespace ncnn

#endif // LAYER_GEMM_POWERPC_SPU_H
