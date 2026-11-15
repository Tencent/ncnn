// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <utility>

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

// Enumerate index tuples independently of the packed GEMM layout.
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

static size_t einsum_ref_offset(const ncnn::Mat& m, const std::string& token, const int* index)
{
    int coords[4] = {0};
    for (size_t p = 0; p < token.size(); p++)
        coords[p] = index[token[p] - 'i'];

    if (m.dims == 1)
        return coords[0];
    if (m.dims == 2)
        return (size_t)coords[0] * m.w + coords[1];
    if (m.dims == 3)
        return coords[0] * m.cstep + (size_t)coords[1] * m.w + coords[2];

    return coords[0] * m.cstep + (size_t)coords[1] * m.h * m.w + (size_t)coords[2] * m.w + coords[3];
}

static int einsum_reference(const std::vector<ncnn::Mat>& a, const char* equation, ncnn::Mat& out)
{
    // Fixed two-operand cases with explicit output; enumerate the union of indices.
    const std::string eq(equation);
    const size_t comma = eq.find(',');
    const size_t arrow = eq.find("->");
    if (a.size() != 2 || comma == std::string::npos || arrow == std::string::npos || comma >= arrow)
        return -1;
    const std::string tokens[2] = {eq.substr(0, comma), eq.substr(comma + 1, arrow - comma - 1)};
    const std::string rhs = eq.substr(arrow + 2);
    if (rhs.empty() || rhs.size() > 4)
        return -1;

    int sizes[16] = {0};
    for (int b = 0; b < 2; b++)
    {
        if ((int)tokens[b].size() != a[b].dims || a[b].dims < 1 || a[b].dims > 4)
            return -1;
        for (int p = 0; p < a[b].dims; p++)
        {
            const int c = tokens[b][p] - 'i';
            if (c < 0 || c >= 16)
                return -1;
            const int size = einsum_ref_dim_size(a[b], p);
            if (size <= 0 || (sizes[c] != 0 && sizes[c] != size))
                return -1;
            sizes[c] = size;
        }
    }

    int shape[4] = {1, 1, 1, 1};
    for (size_t p = 0; p < rhs.size(); p++)
    {
        const int c = rhs[p] - 'i';
        if (c < 0 || c >= 16 || sizes[c] == 0)
            return -1;
        shape[rhs.size() - 1 - p] = sizes[c];
    }
    if (rhs.size() == 1)
        out.create(shape[0]);
    else if (rhs.size() == 2)
        out.create(shape[0], shape[1]);
    else if (rhs.size() == 3)
        out.create(shape[0], shape[1], shape[2]);
    else
        out.create(shape[0], shape[1], shape[2], shape[3]);
    if (out.empty())
        return -1;
    out.fill(0.f);

    size_t total = 1;
    for (int c = 0; c < 16; c++)
        if (sizes[c] != 0)
            total *= sizes[c];
    const float* aptr = a[0];
    const float* bptr = a[1];
    float* outptr = out;
    for (size_t i = 0; i < total; i++)
    {
        size_t remaining = i;
        int index[16] = {0};
        for (int c = 15; c >= 0; c--)
        {
            if (sizes[c] == 0)
                continue;
            index[c] = (int)(remaining % sizes[c]);
            remaining /= sizes[c];
        }
        outptr[einsum_ref_offset(out, rhs, index)] += aptr[einsum_ref_offset(a[0], tokens[0], index)] * bptr[einsum_ref_offset(a[1], tokens[1], index)];
    }

    return 0;
}

static int test_einsum_oracle(const std::vector<ncnn::Mat>& a, const char* equation, int num_threads)
{
    ncnn::Mat ref;
    if (einsum_reference(a, equation, ref) != 0)
    {
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

// Detect silent fallback: both paths produce the same numerical result.
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

class PeakBlobAllocator : public ncnn::Allocator
{
public:
    PeakBlobAllocator()
    {
        live_bytes = 0;
        peak_live_bytes = 0;
    }

    virtual void* fastMalloc(size_t size)
    {
        void* ptr = ncnn::fastMalloc(size);
        if (!ptr)
            return 0;

        allocations.push_back(std::make_pair(ptr, size));
        live_bytes += size;
        if (live_bytes > peak_live_bytes)
            peak_live_bytes = live_bytes;

        return ptr;
    }

    virtual void fastFree(void* ptr)
    {
        for (std::vector<std::pair<void*, size_t> >::iterator it = allocations.begin(); it != allocations.end(); ++it)
        {
            if (it->first == ptr)
            {
                live_bytes -= it->second;
                allocations.erase(it);
                break;
            }
        }

        ncnn::fastFree(ptr);
    }

public:
    size_t live_bytes;
    size_t peak_live_bytes;
    std::vector<std::pair<void*, size_t> > allocations;
};

static int test_einsum_oracle_path(const std::vector<ncnn::Mat>& a, const char* equation, bool expect_fast, int num_threads = 4)
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
    opt.num_threads = num_threads;
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

    int compare_ret = CompareMat(ref, b[0], 0.001f);
    if (compare_ret != 0)
    {
        fprintf(stderr, "test_einsum_oracle_path comparison failed equation=%s\n", equation);
    }

    return compare_ret;
}

static int test_einsum_oracle_paths()
{
    ncnn::Layer* gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
    const bool gemm_available = gemm != 0;
    delete gemm;

    // Packed path with padded output channel stride, in both thread configurations.
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(5, 3, 64);
        a[1] = RandomMat(64, 4);

        int ret = test_einsum_oracle_path(a, "ljk,il->ijk", gemm_available, 1) || test_einsum_oracle_path(a, "ljk,il->ijk", gemm_available, 4);
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

    // Keep the temporary GEMM result bounded for direct and scattered outputs.
    {
        const char* equations[] = {"i,j->ij", "ik,j->ijk"};
        for (int i = 0; i < 2; i++)
        {
            std::vector<ncnn::Mat> a(2);
            if (i == 0)
            {
                a[0] = RandomMat(512);
                a[1] = RandomMat(512);
            }
            else
            {
                a[0] = RandomMat(16, 16);
                a[1] = RandomMat(1024);
            }

            ncnn::Mat ref;
            if (einsum_reference(a, equations[i], ref) != 0)
            {
                fprintf(stderr, "test_einsum_oracle_paths case is out of oracle contract equation=%s\n", equations[i]);
                return -1;
            }

            ncnn::Layer* layer = ncnn::create_layer(ncnn::LayerType::Einsum);
            if (!layer)
            {
                fprintf(stderr, "test_einsum_oracle_paths failed to create Einsum layer equation=%s\n", equations[i]);
                return -1;
            }

            int ret = layer->load_param(equation_params(equations[i]));
            PeakBlobAllocator blob_allocator;
            std::vector<ncnn::Mat> b(1);
            ncnn::Option opt;
            opt.num_threads = 1;
            opt.blob_allocator = &blob_allocator;

            if (ret == 0)
                ret = layer->forward(a, b, opt);
            delete layer;

            if (ret != 0)
            {
                fprintf(stderr, "test_einsum_oracle_paths forward failed equation=%s ret=%d\n", equations[i], ret);
                return ret;
            }

            const size_t output_bytes = ref.total() * ref.elemsize;
            if (blob_allocator.peak_live_bytes >= output_bytes * 3 / 2)
            {
                fprintf(stderr, "test_einsum_oracle_paths output scratch too large equation=%s output_bytes=%lu peak_blob_bytes=%lu\n",
                        equations[i], (unsigned long)output_bytes, (unsigned long)blob_allocator.peak_live_bytes);
                return -1;
            }

            ret = CompareMat(ref, b[0], 0.001f);
            if (ret != 0)
            {
                fprintf(stderr, "test_einsum_oracle_paths comparison failed equation=%s\n", equations[i]);
                return ret;
            }
        }
    }

    return 0;
}

static int test_einsum_oracle_two_operands()
{
    // matrix vector, surface large enough that the packed path is taken
    {
        std::vector<ncnn::Mat> a(2);
        a[0] = RandomMat(300, 4);
        a[1] = RandomMat(300);

        int ret = test_einsum_oracle(a, "ij,j->i", 1) || test_einsum_oracle(a, "ij,j->i", 4);
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

    // Multiple shared output indices exercise the batch carry.
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
           || test_einsum_oracle_two_operands()
           || test_einsum_oracle_paths()
#if NCNN_VALIDATION
           || test_einsum_load_param()
           || test_einsum_load_param_char()
#endif // NCNN_VALIDATION
           ;
}
