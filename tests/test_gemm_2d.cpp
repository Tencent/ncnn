// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_gemm_2.h"

int main()
{
    SRAND(7767517);

    int mnk[][3] = {
        {24, 24, 47},
        {24, 35, 24},
        {47, 24, 24},
        {1, 35, 47},
        {23, 31, 1},
        {23, 1, 23},
        {23, 31, 23}
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
