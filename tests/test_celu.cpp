// License

#include "layer/celu.h"
#include "testutil.h"

static int test_celu(const ncnn::Mat& a, float alpha)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::CELU>("CELU", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_celu failed a.dims=%d a=(%d %d %d) alpha=%f\n", a.dims, a.w, a.h, a.c, alpha);
    }

    return ret;
}

static int test_celu_0()
{
    return 0
           || test_celu(RandomMat(5, 7, 24), 1.f)
           || test_celu(RandomMat(7, 9, 12), 1.f)
           || test_celu(RandomMat(3, 5, 13), 1.f);
}

static int test_celu_1()
{
    return 0
           || test_celu(RandomMat(15, 24), 1.f)
           || test_celu(RandomMat(17, 12), 1.f)
           || test_celu(RandomMat(19, 15), 1.f);
}

static int test_celu_2()
{
    return 0
           || test_celu(RandomMat(128), 1.f)
           || test_celu(RandomMat(124), 1.f)
           || test_celu(RandomMat(127), 1.f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_celu_0()
           || test_celu_1()
           || test_celu_2();
}
