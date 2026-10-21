// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_stridedslice_oom(const ncnn::Mat& a)
{
    ncnn::Mat starts(1);
    ncnn::Mat ends(1);
    ncnn::Mat axes(1);
    ncnn::Mat steps(1);
    ((int*)starts)[0] = 0;
    ((int*)ends)[0] = 0;
    ((int*)axes)[0] = -1;
    ((int*)steps)[0] = 2;

    ncnn::ParamDict pd;
    pd.set(1, 1);
    pd.set(9, starts);
    pd.set(10, ends);
    pd.set(11, axes);
    pd.set(12, steps);

    int ret = test_layer_oom("StridedSlice", pd, std::vector<ncnn::Mat>(), a);
    if (ret != 0)
        fprintf(stderr, "test_stridedslice_oom failed dims=%d shape=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    return ret;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_stridedslice_oom(RandomMat(17))
           || test_stridedslice_oom(RandomMat(17, 9))
           || test_stridedslice_oom(RandomMat(17, 9, 8))
           || test_stridedslice_oom(RandomMat(17, 9, 5, 8));
}
