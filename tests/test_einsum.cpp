// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "testutil.h"

static int test_einsum(const std::vector<ncnn::Mat>& a, const std::string& equation)
{
    ncnn::Mat equation_mat(equation.size());
    for (size_t i = 0; i < equation.size(); i++)
    {
        ((int*)equation_mat)[i] = equation[i];
    }

    ncnn::ParamDict pd;
    pd.set(0, equation_mat);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Einsum", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_einsum failed a[0].dims=%d a[0]=(%d %d %d) equation=%s\n", a[0].dims, a[0].w, a[0].h, a[0].c, equation.c_str());
    }

    return ret;
}

static int test_einsum_0()
{
    std::vector<ncnn::Mat> a(1);
    a[0] = RandomMat(32, 32);

    return test_einsum(a, "ii");
}

static int test_einsum_1()
{
    std::vector<ncnn::Mat> a(1);
    a[0] = RandomMat(27, 32);

    return test_einsum(a, "ij->i") || test_einsum(a, "ji->i");
}

static int test_einsum_2()
{
    std::vector<ncnn::Mat> a(1);
    a[0] = RandomMat(17, 14, 32);

    return 0
           || test_einsum(a, "ijk->i")
           || test_einsum(a, "jik->i")
           || test_einsum(a, "jki->i")
           || test_einsum(a, "ikj->ij")
           || test_einsum(a, "kij->ij")
           || test_einsum(a, "ijk->ij");
}

static int test_einsum_3()
{
    std::vector<ncnn::Mat> a(1);
    a[0] = RandomMat(17, 14, 9, 32);

    return 0
           || test_einsum(a, "jkli->i")
           || test_einsum(a, "jkil->i")
           || test_einsum(a, "jikl->i")
           || test_einsum(a, "ijkl->i")
           || test_einsum(a, "iklj->ij")
           || test_einsum(a, "klij->ij")
           || test_einsum(a, "kijl->ij")
           || test_einsum(a, "ijkl->ij")
           || test_einsum(a, "ijlk->ijk")
           || test_einsum(a, "lijk->ijk")
           || test_einsum(a, "ijkl->ijk");
}

static int test_einsum_4()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(12, 28);
    a[1] = RandomMat(12);

    return test_einsum(a, "ij,j->i");
}

static int test_einsum_5()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(14);
    a[1] = RandomMat(14, 7, 16);

    return test_einsum(a, "k,ijk->ij");
}

static int test_einsum_6()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(27);
    a[1] = RandomMat(32);

    return test_einsum(a, "i,j->ij");
}

static int test_einsum_7()
{
    std::vector<ncnn::Mat> a(4);
    a[0] = RandomMat(7);
    a[1] = RandomMat(2);
    a[2] = RandomMat(11);
    a[3] = RandomMat(16);

    return test_einsum(a, "i,j,k,l->ijkl");
}

static int test_einsum_8()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(5, 2, 3);
    a[1] = RandomMat(4, 5, 3);

    return test_einsum(a, "ijl,ilk->ijk");
}

static int test_einsum_9()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(4, 5, 3);
    a[1] = RandomMat(5, 2, 3);

    return test_einsum(a, "ilk,ijl->ijk");
}

static int test_einsum_10()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(15, 12);
    a[1] = RandomMat(24, 15, 13);
    a[2] = RandomMat(24, 12);

    return test_einsum(a, "ik,jkl,il->ij");
}

static int test_einsum_11()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(7, 5, 3, 2);
    a[1] = RandomMat(5, 17, 3, 11);

    return test_einsum(a, "imnj,kmln->ijkl");
}

int main()
{
    SRAND(7767517);

    return 0
           || test_einsum_0()
           || test_einsum_1()
           || test_einsum_2()
           || test_einsum_3()
           || test_einsum_4()
           || test_einsum_5()
           || test_einsum_6()
           || test_einsum_7()
           || test_einsum_8()
           || test_einsum_9()
           || test_einsum_10()
           || test_einsum_11();
}
