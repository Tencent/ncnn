// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

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

static ncnn::ParamDict equation_params(const char* equation)
{
    ncnn::Mat m((int)strlen(equation));
    int* p = m;
    for (int i = 0; i < m.w; i++)
        p[i] = equation[i];

    ncnn::ParamDict pd;
    pd.set(0, m);
    return pd;
}

static int test_einsum_load_param_equation(const char* equation, int expected_ret)
{
    int ret = test_layer_param(ncnn::LayerType::Einsum, equation_params(equation), expected_ret);
    if (ret != 0)
    {
        fprintf(stderr, "test_einsum_load_param failed equation=%s\n", equation);
    }

    return ret;
}

static int test_einsum_load_param()
{
    return 0
           || test_einsum_load_param_equation("ij->ji", -1)
           || test_einsum_load_param_equation("ji->ij", 0)
           || test_einsum_load_param_equation("", -1)
           || test_einsum_load_param_equation("->i", -1)
           || test_einsum_load_param_equation("i,->i", -1)
           || test_einsum_load_param_equation(",i->i", -1)
           || test_einsum_load_param_equation("i,,j->ij", -1)
           || test_einsum_load_param_equation("i->", -1)
           || test_einsum_load_param_equation("i->ij", -1)
           || test_einsum_load_param_equation("i->ii", -1)
           || test_einsum_load_param_equation("i->j", -1)
           || test_einsum_load_param_equation("ijklm->i", -1)
           || test_einsum_load_param_equation("i->i->i", -1);
}

static int test_einsum_load_param_char()
{
    ncnn::ParamDict pd = equation_params("ij->i");
    ncnn::Mat m = pd.get(0, ncnn::Mat());
    int* p = m;
    p[0] = 'i' + 256;

    int ret = test_layer_param(ncnn::LayerType::Einsum, pd, -1);
    if (ret != 0)
    {
        fprintf(stderr, "test_einsum_load_param_char failed value=%d\n", p[0]);
    }

    return ret;
}

static int test_einsum_reload_case(ncnn::Layer* layer, const char* equation, const std::vector<ncnn::Mat>& a, const ncnn::Mat& expected)
{
    int ret = layer->load_param(equation_params(equation));
    if (ret != 0)
    {
        fprintf(stderr, "test_einsum_reload load_param failed equation=%s ret=%d\n", equation, ret);
        return ret;
    }

    std::vector<ncnn::Mat> b(1);
    ncnn::Option opt;
    ret = layer->forward(a, b, opt);
    if (ret == 0)
        ret = CompareMat(expected, b[0], 0.f);
    if (ret != 0)
    {
        fprintf(stderr, "test_einsum_reload failed equation=%s ret=%d\n", equation, ret);
    }

    return ret;
}

static int test_einsum_reload()
{
    std::vector<ncnn::Mat> a(1);
    a[0].create(2, 2);
    for (int i = 0; i < 4; i++)
        a[0][i] = (float)(i + 1);

    ncnn::Mat trace(1);
    trace[0] = 5.f;
    ncnn::Mat sum(2);
    sum[0] = 3.f;
    sum[1] = 7.f;

    // reuse one layer to check that loading replaces the previous equation
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::Einsum);
    if (!layer)
        return -1;

    int ret = layer->load_param(equation_params("ij,j->i"));
    if (ret != 0)
    {
        fprintf(stderr, "test_einsum_reload initial load failed ret=%d\n", ret);
        delete layer;
        return ret;
    }

    ret = 0
          || test_einsum_reload_case(layer, "ii", a, trace)
          || test_einsum_reload_case(layer, "ij->i", a, sum);
    delete layer;

    return ret;
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
           || test_einsum_load_param()
           || test_einsum_load_param_char()
           || test_einsum_reload();
}
