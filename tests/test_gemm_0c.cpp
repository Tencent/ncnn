// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_gemm_0.h"

int main()
{
    SRAND(7767517);

    int mnk[][3] = {
        {12, 12, 23},
        {12, 31, 12},
        {23, 12, 12},
        {1, 1, 47},
        {1, 35, 1},
        {47, 1, 1}
    };

    int mnk_count = sizeof(mnk) / sizeof(int) / 3;

    for (int i = 0; i < mnk_count; i++)
    {
        int M = mnk[i][0];
        int N = mnk[i][1];
        int K = mnk[i][2];

        int ret = test_gemm_0(M, N, K);

        if (ret != 0)
            return ret;
    }

    return 0;
}
