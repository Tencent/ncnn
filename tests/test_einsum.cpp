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

#if NCNN_VALIDATION
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
#endif // NCNN_VALIDATION

// independent reference implementation, used as an oracle for the layer.
// it enumerates every index tuple, multiplies the gathered operand values and
// scatters the product into the output, which is a different shape of
// computation than the layer itself, so it can disagree instead of repeating
// the layer's own arithmetic.

static int einsum_ref_dim_size(const ncnn::Mat& m, int s)
{
    if (m.dims == 1)
        return m.w;
    if (m.dims == 2)
        return s == 0 ? m.h : m.w;
    if (m.dims == 3)
        return s == 0 ? m.c : (s == 1 ? m.h : m.w);
    if (m.dims == 4)
        return s == 0 ? m.c : (s == 1 ? m.d : (s == 2 ? m.h : m.w));

    return 0;
}

static float einsum_ref_get(const ncnn::Mat& m, const int* idx)
{
    if (m.dims == 1)
        return m[idx[0]];
    if (m.dims == 2)
        return m.row(idx[0])[idx[1]];
    if (m.dims == 3)
        return m.channel(idx[0]).row(idx[1])[idx[2]];

    return m.channel(idx[0]).depth(idx[1]).row(idx[2])[idx[3]];
}

static void einsum_ref_add(ncnn::Mat& m, const int* idx, float v)
{
    if (m.dims == 1)
    {
        m[idx[0]] += v;
        return;
    }
    if (m.dims == 2)
    {
        m.row(idx[0])[idx[1]] += v;
        return;
    }
    if (m.dims == 3)
    {
        m.channel(idx[0]).row(idx[1])[idx[2]] += v;
        return;
    }

    m.channel(idx[0]).depth(idx[1]).row(idx[2])[idx[3]] += v;
}

static float einsum_ref_leaf(const std::vector<ncnn::Mat>& a, const std::vector<std::string>& tokens, const int* char_index)
{
    float v = 1.f;

    for (size_t b = 0; b < a.size(); b++)
    {
        int idx[4];
        for (size_t s = 0; s < tokens[b].size(); s++)
        {
            idx[s] = char_index[tokens[b][s] - 'i'];
        }

        v *= einsum_ref_get(a[b], idx);
    }

    return v;
}

static void einsum_ref_enumerate(const std::vector<ncnn::Mat>& a, const std::vector<std::string>& tokens, const std::string& out_token, const std::vector<int>& used, const int* sizes, int* char_index, size_t d, ncnn::Mat& out)
{
    if (d == used.size())
    {
        const float v = einsum_ref_leaf(a, tokens, char_index);

        int idx[4];
        for (size_t s = 0; s < out_token.size(); s++)
        {
            idx[s] = char_index[out_token[s] - 'i'];
        }

        einsum_ref_add(out, idx, v);
        return;
    }

    const int c = used[d];
    for (int i = 0; i < sizes[c]; i++)
    {
        char_index[c] = i;
        einsum_ref_enumerate(a, tokens, out_token, used, sizes, char_index, d + 1, out);
    }
}

// returns 0 and fills out when the oracle understands the case, -1 when the case
// is out of the oracle contract and the caller should skip it
static int einsum_reference(const std::vector<ncnn::Mat>& a, const char* equation, ncnn::Mat& out)
{
    const std::string eq(equation);

    if (eq == "ii")
    {
        if (a.size() != 1 || a[0].dims != 2 || a[0].w != a[0].h)
            return -1;

        out.create(1);
        if (out.empty())
            return -1;

        float sum = 0.f;
        for (int i = 0; i < a[0].h; i++)
        {
            sum += a[0].row(i)[i];
        }
        out[0] = sum;
        return 0;
    }

    const size_t arrow = eq.find("->");
    if (arrow == std::string::npos)
        return -1;

    const std::string lhs = eq.substr(0, arrow);
    const std::string rhs = eq.substr(arrow + 2);
    if (rhs.empty() || rhs.size() > 4)
        return -1;

    std::vector<std::string> tokens;
    {
        std::string cur;
        for (size_t i = 0; i < lhs.size(); i++)
        {
            if (lhs[i] == ',')
            {
                tokens.push_back(cur);
                cur.clear();
            }
            else
            {
                cur.push_back(lhs[i]);
            }
        }
        tokens.push_back(cur);
    }

    if (tokens.size() != a.size())
        return -1;

    int sizes[16];
    bool used_flag[16];
    for (int c = 0; c < 16; c++)
    {
        sizes[c] = 1;
        used_flag[c] = false;
    }

    for (size_t b = 0; b < tokens.size(); b++)
    {
        if (tokens[b].empty() || tokens[b].size() > 4)
            return -1;
        if ((int)tokens[b].size() != a[b].dims)
            return -1;

        for (size_t s = 0; s < tokens[b].size(); s++)
        {
            const int c = tokens[b][s] - 'i';
            if (c < 0 || c >= 16)
                return -1;

            const int sz = einsum_ref_dim_size(a[b], (int)s);
            if (used_flag[c] && sizes[c] != sz)
                return -1; // the operands disagree on this dimension, out of contract

            sizes[c] = sz;
            used_flag[c] = true;
        }
    }

    for (size_t s = 0; s < rhs.size(); s++)
    {
        const int c = rhs[s] - 'i';
        if (c < 0 || c >= 16 || !used_flag[c])
            return -1;
    }

    std::vector<int> used;
    for (int c = 0; c < 16; c++)
    {
        if (used_flag[c])
            used.push_back(c);
    }

    // the layer emits the output token positions as w h d c from the innermost
    // position, so the third position is the depth only for a 4 dim output
    const int ow = sizes[rhs[rhs.size() - 1] - 'i'];
    const int oh = rhs.size() >= 2 ? sizes[rhs[rhs.size() - 2] - 'i'] : 1;
    const int ohd = rhs.size() >= 3 ? sizes[rhs[rhs.size() - 3] - 'i'] : 1;
    const int oc = rhs.size() >= 4 ? sizes[rhs[rhs.size() - 4] - 'i'] : 1;

    if (rhs.size() == 1)
        out.create(ow);
    else if (rhs.size() == 2)
        out.create(ow, oh);
    else if (rhs.size() == 3)
        out.create(ow, oh, ohd);
    else
        out.create(ow, oh, ohd, oc);

    if (out.empty())
        return -1;

    out.fill(0.f);

    int char_index[16];
    for (int c = 0; c < 16; c++)
    {
        char_index[c] = 0;
    }

    einsum_ref_enumerate(a, tokens, rhs, used, sizes, char_index, 0, out);

    return 0;
}

static int test_einsum_oracle(const std::vector<ncnn::Mat>& a, const char* equation, int num_threads)
{
    ncnn::Mat ref;
    if (einsum_reference(a, equation, ref) != 0)
    {
        // every case in this battery is expected to be inside the contract, a
        // skip here means the case is malformed and must not pass silently
        fprintf(stderr, "test_einsum_oracle case is out of oracle contract equation=%s\n", equation);
        return -1;
    }

    ncnn::Layer* layer = ncnn::create_layer(ncnn::LayerType::Einsum);
    if (!layer)
        return -1;

    int ret = layer->load_param(equation_params(equation));
    if (ret != 0)
    {
        fprintf(stderr, "test_einsum_oracle load_param failed equation=%s ret=%d\n", equation, ret);
        delete layer;
        return ret;
    }

    std::vector<ncnn::Mat> b(1);
    ncnn::Option opt;
    opt.num_threads = num_threads;

    ret = layer->forward(a, b, opt);
    delete layer;

    if (ret == 0)
        ret = CompareMat(ref, b[0], 0.001f);

    if (ret != 0)
    {
        fprintf(stderr, "test_einsum_oracle failed equation=%s num_threads=%d\n", equation, num_threads);
    }

    return ret;
}

// counts workspace allocations so a test can prove which path ran. without it a
// silent fall back to the generic implementation would still compare equal
class WorkspaceCountingAllocator : public ncnn::PoolAllocator
{
public:
    WorkspaceCountingAllocator()
    {
        count = 0;
    }

    virtual void* fastMalloc(size_t size)
    {
        count++;
        return ncnn::PoolAllocator::fastMalloc(size);
    }

public:
    int count;
};

static int test_einsum_oracle_path(const std::vector<ncnn::Mat>& a, const char* equation, bool expect_fast)
{
    ncnn::Mat ref;
    if (einsum_reference(a, equation, ref) != 0)
    {
        fprintf(stderr, "test_einsum_oracle_path case is out of oracle contract equation=%s\n", equation);
        return -1;
    }

    ncnn::Layer* layer = ncnn::create_layer(ncnn::LayerType::Einsum);
    if (!layer)
        return -1;

    int ret = layer->load_param(equation_params(equation));

    std::vector<ncnn::Mat> b(1);
    ncnn::Option opt;
    opt.num_threads = 4;
    WorkspaceCountingAllocator workspace_allocator;
    opt.workspace_allocator = &workspace_allocator;

    if (ret == 0)
        ret = layer->forward(a, b, opt);
    delete layer;

    if (ret != 0)
        return ret;

    const bool fast = workspace_allocator.count > 0;
    if (fast != expect_fast)
    {
        fprintf(stderr, "test_einsum_oracle_path unexpected path equation=%s expect_fast=%d workspace_allocations=%d\n",
                equation, (int)expect_fast, workspace_allocator.count);
        return -1;
    }

    return CompareMat(ref, b[0], 0.001f);
}

static int test_einsum_oracle_paths()
{
    // packed path: a contraction with a long reduction
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(4, 3, 256);
        a[1] = RandomMat(256, 5);

        int ret = 0
                  || test_einsum_oracle_path(a, "ljk,il->ijk", true)
                  || test_einsum_oracle_path(a, "ljk,il->ijk", true);
        if (ret != 0)
            return ret;
    }

    // generic path: a reduction over an index that appears in one operand only
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(5, 1);
        a[1] = RandomMat(4, 1);

        int ret = test_einsum_oracle_path(a, "ij,ik->ij", false);
        if (ret != 0)
            return ret;
    }

    // generic path: an elementwise product, every index is a batch index
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(8);
        a[1] = RandomMat(8);

        int ret = test_einsum_oracle_path(a, "i,i->i", false);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_einsum_oracle_one_operand()
{
    // single operand equations stay on the generic path, they are here to pin
    // the token to axis mapping and the output layout independently of the
    // packed path
    std::vector<ncnn::Mat> a(1);
    a[0] = RandomMat(5, 4);

    int ret = 0
              || test_einsum_oracle(a, "ij->i", 1)
              || test_einsum_oracle(a, "ji->i", 1)
              || test_einsum_oracle(a, "ji->ij", 4);

    std::vector<ncnn::Mat> a3(1);
    a3[0] = RandomMat(4, 3, 2);

    ret = ret
          || test_einsum_oracle(a3, "jki->i", 1)
          || test_einsum_oracle(a3, "kij->ij", 4)
          || test_einsum_oracle(a3, "ijk->ijk", 1);

    std::vector<ncnn::Mat> a4(1);
    a4[0] = RandomMat(4, 3, 2, 2);

    ret = ret
          || test_einsum_oracle(a4, "jkli->i", 4)
          || test_einsum_oracle(a4, "klij->ij", 4)
          || test_einsum_oracle(a4, "ijkl->ijkl", 1);

    return ret;
}

static int test_einsum_oracle_two_operands()
{
    // trace
    {
        std::vector<ncnn::Mat> a(1);
        a[0] = RandomMat(4, 4);

        int ret = test_einsum_oracle(a, "ii", 1);
        if (ret != 0)
            return ret;
    }

    // matrix vector, surface large enough that the packed path is taken
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(300, 4);
        a[1] = RandomMat(300);

        int ret = test_einsum_oracle(a, "ij,j->i", 1) || test_einsum_oracle(a, "ij,j->i", 4);
        if (ret != 0)
            return ret;
    }

    // outer product, K is one so the surface has to be larger to pass the
    // packed path threshold
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(20);
        a[1] = RandomMat(20);

        int ret = test_einsum_oracle(a, "i,j->ij", 1) || test_einsum_oracle(a, "i,j->ij", 4);
        if (ret != 0)
            return ret;
    }

    // elementwise product: every index is shared by both operands and present in
    // the output, so this stays on the generic path by design
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(5);
        a[1] = RandomMat(5);

        int ret = test_einsum_oracle(a, "i,i->i", 1);
        if (ret != 0)
            return ret;
    }

    // reduction over an index that appears in one operand only, also generic
    // path by design
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(5, 1);
        a[1] = RandomMat(4, 1);

        int ret = test_einsum_oracle(a, "ij,ik->ij", 1);
        if (ret != 0)
            return ret;
    }

    // two contracted indices at once
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(40, 6, 3, 2);
        a[1] = RandomMat(6, 40);

        int ret = 0
                  || test_einsum_oracle(a, "ijkl,lk->ij", 1)
                  || test_einsum_oracle(a, "ijkl,lk->ij", 4);
        if (ret != 0)
            return ret;
    }

    // batched attention style contraction, a[0] is (i m j k), a[1] is (l i m)
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(16, 12, 64, 3);
        a[1] = RandomMat(64, 3, 4);

        int ret = test_einsum_oracle(a, "imjk,lim->ijkl", 1) || test_einsum_oracle(a, "imjk,lim->ijkl", 4);
        if (ret != 0)
            return ret;
    }

    // weight times activation style contraction, a[0] is (l j k), a[1] is (i l)
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(4, 3, 256);
        a[1] = RandomMat(256, 5);

        int ret = test_einsum_oracle(a, "ljk,il->ijk", 1) || test_einsum_oracle(a, "ljk,il->ijk", 4);
        if (ret != 0)
            return ret;
    }

    // every index is shared by both operands and present in the output, so the
    // batch carry runs with more than one batch index while the reduction is
    // still large enough for the packed path
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(1200, 3, 2);
        a[1] = RandomMat(1200, 3, 2);

        int ret = 0
                  || test_einsum_oracle(a, "ijm,ijm->ij", 1)
                  || test_einsum_oracle(a, "ijm,ijm->ij", 4);
        if (ret != 0)
            return ret;
    }

    // 4d operand whose channel stride is padded (w * h * d * 4 is not a
    // multiple of 16), so the stride model has to use cstep
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(7, 5, 63, 2);
        a[1] = RandomMat(63, 2, 4);

        int ret = 0
                  || test_einsum_oracle(a, "imjk,lim->ijkl", 1)
                  || test_einsum_oracle(a, "imjk,lim->ijkl", 4);
        if (ret != 0)
            return ret;
    }

    // 3d output whose w * h is not a multiple of 4, so the output channel
    // stride is padded as well
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(5, 3, 64);
        a[1] = RandomMat(64, 4);

        int ret = 0
                  || test_einsum_oracle(a, "ljk,il->ijk", 1)
                  || test_einsum_oracle(a, "ljk,il->ijk", 4);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_einsum_oracle_many_operands()
{
    {
        std::vector<ncnn::Mat> a(3);
        a[0] = RandomMat(3, 4);
        a[1] = RandomMat(2, 3, 3);
        a[2] = RandomMat(2, 4);

        int ret = test_einsum_oracle(a, "ik,jkl,il->ij", 1) || test_einsum_oracle(a, "ik,jkl,il->ij", 4);
        if (ret != 0)
            return ret;
    }

    {
        std::vector<ncnn::Mat> a(4);
        a[0] = RandomMat(3);
        a[1] = RandomMat(2);
        a[2] = RandomMat(4);
        a[3] = RandomMat(2);

        int ret = test_einsum_oracle(a, "i,j,k,l->ijkl", 1) || test_einsum_oracle(a, "i,j,k,l->ijkl", 4);
        if (ret != 0)
            return ret;
    }

    return 0;
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
           || test_einsum_reload()
           || test_einsum_oracle_one_operand()
           || test_einsum_oracle_two_operands()
           || test_einsum_oracle_many_operands()
           || test_einsum_oracle_paths()
#if NCNN_VALIDATION
           || test_einsum_load_param()
           || test_einsum_load_param_char()
#endif // NCNN_VALIDATION
           ;
}
