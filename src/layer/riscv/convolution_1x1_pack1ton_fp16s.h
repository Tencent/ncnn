// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void conv1x1s1_sgemm_pack1ton_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    const int size = w * h;

    Mat bottom_im2col = bottom_blob;
    bottom_im2col.w = size;
    bottom_im2col.h = 1;

    im2col_sgemm_pack1ton_fp16sa_rvv(bottom_im2col, top_blob, kernel, _bias, opt);
}
