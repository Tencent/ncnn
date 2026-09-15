// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

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

static int test_einsum_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::Einsum);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if ((ret == 0) != valid)
    {
        fprintf(stderr, "Einsum load_param returned %d, expected %s\n", ret, valid ? "success" : "failure");
        return -1;
    }

    return 0;
}

static ncnn::Mat param_int_array(int size, int value)
{
    ncnn::Mat m(size);
    int* p = m;
    for (int i = 0; i < size; i++)
        p[i] = value;
    return m;
}

static ncnn::ParamDict equation_params(const char* equation)
{
    ncnn::Mat m = param_int_array((int)strlen(equation), 0);
    int* p = m;
    for (int i = 0; i < m.w; i++)
        p[i] = equation[i];
    ncnn::ParamDict pd;
    pd.set(0, m);
    return pd;
}

static int test_einsum_load_param()
{
    const char* invalid[] = {"", "->i", "i,->i", ",i->i", "i,,j->ij", "i->", "i->ij", "i->ii", "i->j", "ijklm->i", "i->i->i"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++)
    {
        if (test_einsum_load_param_case(equation_params(invalid[i]), false) != 0)
            return -1;
    }

    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::Einsum);
    if (!layer)
        return 0;
    int ret = layer->load_param(equation_params("ij,j->i"));
    ret |= layer->load_param(equation_params("ii"));
    std::vector<ncnn::Mat> inputs(1);
    inputs[0].create(2, 2);
    for (int i = 0; i < 4; i++)
        inputs[0][i] = (float)(i + 1);
    std::vector<ncnn::Mat> outputs(1);
    ncnn::Option opt;
    if (ret == 0)
        ret = layer->forward(inputs, outputs, opt);
    if (ret == 0 && (outputs[0].empty() || outputs[0][0] != 5.f))
        ret = -1;
    ret |= layer->load_param(equation_params("ij->i"));
    if (ret == 0)
        ret = layer->forward(inputs, outputs, opt);
    if (ret == 0 && (outputs[0].w != 2 || outputs[0][0] != 3.f || outputs[0][1] != 7.f))
        ret = -1;
    delete layer;
    ncnn::ParamDict pd = equation_params("ij->i");
    ncnn::Mat m = pd.get(0, ncnn::Mat());
    int* p = m;
    p[0] = 'i' + 256; // cannot silently narrow to a valid token
    return ret || test_einsum_load_param_case(pd, false);
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
           || test_einsum_11()
           || test_einsum_load_param();
}
