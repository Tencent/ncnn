// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_gemm_2.h"

int main()
{
    SRAND(7767517);

    int mnk[][3] = {
        {31, 7, 3},
        {28, 20, 7},
        {32, 32, 9},
        {44, 19, 7},
        {47, 35, 48},
        {47, 48, 47},
        {48, 35, 47}
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
