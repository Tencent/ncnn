// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "einsum.h"

#include "layer_type.h"

#include <limits.h>

namespace ncnn {

Einsum::Einsum()
{
    one_blob_only = false;
    support_inplace = false;
}

int Einsum::load_param(const ParamDict& pd)
{
    Mat equation_mat = pd.get(0, Mat());

#if NCNN_VALIDATION
    {
        const int equation_mat_type = pd.type(0);
        if (equation_mat_type != 0 && equation_mat_type != 4 && equation_mat_type != 5)
            return -1;

        if ((equation_mat.dims != 0 || equation_mat.w != 0 || equation_mat.data) && (equation_mat.dims != 1 || equation_mat.w < 0 || equation_mat.elempack != 1 || equation_mat.elemsize != 4u || (equation_mat.w > 0 && !equation_mat.data)))
            return -1;
    }

    if (equation_mat.empty())
        return -1;
#endif // NCNN_VALIDATION

    // validate character values before narrowing to char
    const int* p = equation_mat;
    std::string equation;
    equation.resize(equation_mat.w);
    for (int i = 0; i < equation_mat.w; i++)
    {
#if NCNN_VALIDATION
        if ((p[i] < 'i' || p[i] > 'x') && p[i] != ',' && p[i] != '-' && p[i] != '>')
            return -1;
#endif // NCNN_VALIDATION
        equation[i] = (char)p[i];
    }

    if (equation == "ii")
    {
        lhs_tokens.clear();
        rhs_token = "ii";
        return 0;
    }

    // keep parsed tokens local until the equation is valid
    std::vector<std::string> tokens;
    std::string token;
#if NCNN_VALIDATION
    bool seen[16] = {false};
#endif // NCNN_VALIDATION
    int arrow = -1;
    for (int i = 0; i < equation_mat.w; i++)
    {
        const char ch = equation[i];
        if (ch == ',' || ch == '-')
        {
#if NCNN_VALIDATION
            if (token.empty() || token.size() > 4)
                return -1;
#endif // NCNN_VALIDATION
            tokens.push_back(token);
            token.clear();
            if (ch == '-')
            {
#if NCNN_VALIDATION
                if (i + 1 >= equation_mat.w || equation[i + 1] != '>')
                    return -1;
#endif // NCNN_VALIDATION
                arrow = i;
                break;
            }
        }
        else
        {
#if NCNN_VALIDATION
            if (ch < 'i' || ch > 'x')
                return -1;
#endif // NCNN_VALIDATION
            token.push_back(ch);
#if NCNN_VALIDATION
            seen[ch - 'i'] = true;
#endif // NCNN_VALIDATION
        }
    }

#if NCNN_VALIDATION
    if (arrow < 0)
        return -1;
#endif // NCNN_VALIDATION

    const int output_dims = equation_mat.w - arrow - 2;
#if NCNN_VALIDATION
    if (output_dims < 1 || output_dims > 4)
        return -1;
#endif // NCNN_VALIDATION

    // the implementation emits dimensions in the canonical i,j,k,l order
    std::string output;
    output.resize(output_dims);
    for (int i = 0; i < output_dims; i++)
    {
#if NCNN_VALIDATION
        if (equation[arrow + 2 + i] != 'i' + i || !seen[i])
            return -1;
#endif // NCNN_VALIDATION
        output[i] = equation[arrow + 2 + i];
    }

    lhs_tokens = tokens;
    rhs_token = output;

    return 0;
}

static float get_indexed_value(const Mat& m, const std::string& token, std::vector<int>& indexes)
{
    const int dims = m.dims;

    if (dims == 1)
    {
        int x = indexes[token[0] - 'i'];
        return m[x];
    }

    if (dims == 2)
    {
        int y = indexes[token[0] - 'i'];
        int x = indexes[token[1] - 'i'];
        return m.row(y)[x];
    }

    if (dims == 3)
    {
        int c = indexes[token[0] - 'i'];
        int y = indexes[token[1] - 'i'];
        int x = indexes[token[2] - 'i'];
        return m.channel(c).row(y)[x];
    }

    if (dims == 4)
    {
        int c = indexes[token[0] - 'i'];
        int z = indexes[token[1] - 'i'];
        int y = indexes[token[2] - 'i'];
        int x = indexes[token[3] - 'i'];
        return m.channel(c).depth(z).row(y)[x];
    }

    // should never reach here
    return 0;
}

static float sum_dim(const std::vector<int>& dim_sizes, int d, const std::vector<Mat>& bottom_blobs, const std::vector<std::string>& tokens, std::vector<int>& indexes)
{
    if (d == (int)dim_sizes.size())
    {
        float v = 1.f;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            v *= get_indexed_value(bottom_blobs[b], tokens[b], indexes);
        }

        return v;
    }

    float sum = 0.f;

    for (int i = 0; i < dim_sizes[d]; i++)
    {
        indexes[d] = i;

        sum += sum_dim(dim_sizes, d + 1, bottom_blobs, tokens, indexes);
    }

    return sum;
}

// Pack plain two-operand contractions as (rows, K) blocks for Gemm.
// Unsupported equations retain the generic implementation.

static int dim_size_at(const Mat& m, int p)
{
    if (m.dims == 1)
        return m.w;
    if (m.dims == 2)
        return p == 0 ? m.h : m.w;
    if (m.dims == 3)
        return p == 0 ? m.c : (p == 1 ? m.h : m.w);

    return p == 0 ? m.c : (p == 1 ? m.d : (p == 2 ? m.h : m.w));
}

static size_t stride_at(const Mat& m, int p)
{
    if (m.dims == 1)
        return 1;
    if (m.dims == 2)
        return p == 0 ? (size_t)m.w : 1;
    if (m.dims == 3)
        return p == 0 ? m.cstep : (p == 1 ? (size_t)m.w : 1);

    return p == 0 ? m.cstep : (p == 1 ? (size_t)m.h * m.w : (p == 2 ? (size_t)m.w : 1));
}

// one odometer step over the listed dimensions, keeping the operand element
// offset in sync so the packing loops never recompute indices per element
static void advance_index(const int* list_char, const size_t* list_stride, const int* dim_sizes, int list_count, int* char_index, size_t& off)
{
    for (int d = list_count - 1; d >= 0; d--)
    {
        const int c = list_char[d];
        const int v = char_index[c] + 1;
        if (v < dim_sizes[c])
        {
            char_index[c] = v;
            off += list_stride[d];
            return;
        }

        char_index[c] = 0;
        off -= (size_t)(dim_sizes[c] - 1) * list_stride[d];
    }
}

static void advance_char(const int* chars, const int* dim_sizes, int count, int* char_index)
{
    for (int d = count - 1; d >= 0; d--)
    {
        const int c = chars[d];
        if (++char_index[c] < dim_sizes[c])
            return;

        char_index[c] = 0;
    }
}

struct TwoBlobContraction
{
    int dim_sizes[16];
    int token_pos_a[16];
    int token_pos_b[16];
    int batch_chars[16];
    int aout_chars[16];
    int bout_chars[16];
    int kchars[16];
    int batch_count;
    int aout_count;
    int bout_count;
    int kcount;
    size_t M;
    size_t N;
    size_t K;
    size_t batch;
};

static bool make_contraction_plan(const std::vector<Mat>& bottom_blobs, const std::vector<std::string>& lhs_tokens, const std::string& rhs_token, TwoBlobContraction& plan)
{
    if (lhs_tokens.size() != 2)
        return false;

    const Mat& A = bottom_blobs[0];
    const Mat& B = bottom_blobs[1];

    if (A.elemsize != 4u || B.elemsize != 4u)
        return false;
    if (A.elempack != 1 || B.elempack != 1)
        return false;
    if (A.dims < 1 || A.dims > 4 || B.dims < 1 || B.dims > 4)
        return false;
    if ((int)lhs_tokens[0].size() != A.dims || (int)lhs_tokens[1].size() != B.dims)
        return false;
    if (rhs_token.empty() || rhs_token.size() > 4)
        return false;

    bool in_out[16] = {false};
    for (int c = 0; c < 16; c++)
    {
        plan.dim_sizes[c] = 1;
        plan.token_pos_a[c] = -1;
        plan.token_pos_b[c] = -1;
    }

    int* positions[2] = {plan.token_pos_a, plan.token_pos_b};
    for (int b = 0; b < 2; b++)
    {
        for (int p = 0; p < bottom_blobs[b].dims; p++)
        {
            const int c = lhs_tokens[b][p] - 'i';
            if (c < 0 || c >= 16 || positions[b][c] >= 0)
                return false; // Diagonals stay on the generic path.

            const int size = dim_size_at(bottom_blobs[b], p);
            if (size <= 0 || (b == 1 && plan.token_pos_a[c] >= 0 && plan.dim_sizes[c] != size))
                return false;

            positions[b][c] = p;
            plan.dim_sizes[c] = size;
        }
    }

    for (size_t p = 0; p < rhs_token.size(); p++)
    {
        const int c = rhs_token[p] - 'i';
        if (c < 0 || c >= 16)
            return false;
        if (in_out[c])
            return false; // a repeated output index is not a canonical equation
        if (plan.token_pos_a[c] < 0 && plan.token_pos_b[c] < 0)
            return false; // an output index has to come from an operand

        in_out[c] = true;
    }

    plan.batch_count = 0;
    plan.aout_count = 0;
    plan.bout_count = 0;
    plan.kcount = 0;

    for (int c = 0; c < 16; c++)
    {
        const bool in_a = plan.token_pos_a[c] >= 0;
        const bool in_b = plan.token_pos_b[c] >= 0;
        if (!in_a && !in_b)
            continue;

        if (in_a && in_b)
        {
            if (in_out[c])
                plan.batch_chars[plan.batch_count++] = c;
            else
                plan.kchars[plan.kcount++] = c;

            continue;
        }

        if (!in_out[c])
            return false; // a reduction over one operand only, the generic path handles it

        if (in_a)
            plan.aout_chars[plan.aout_count++] = c;
        else
            plan.bout_chars[plan.bout_count++] = c;
    }

    plan.M = 1;
    plan.N = 1;
    plan.K = 1;
    plan.batch = 1;
    for (int d = 0; d < plan.aout_count; d++)
        plan.M *= plan.dim_sizes[plan.aout_chars[d]];
    for (int d = 0; d < plan.bout_count; d++)
        plan.N *= plan.dim_sizes[plan.bout_chars[d]];
    for (int d = 0; d < plan.kcount; d++)
        plan.K *= plan.dim_sizes[plan.kchars[d]];
    for (int d = 0; d < plan.batch_count; d++)
        plan.batch *= plan.dim_sizes[plan.batch_chars[d]];

    if (plan.M * plan.K > (size_t)INT_MAX || plan.N * plan.K > (size_t)INT_MAX || plan.M * plan.N > (size_t)INT_MAX)
        return false;

    // Small slices do not amortize packing and per-slice GEMM setup.
    if (plan.M * plan.N * plan.K < 256)
        return false;

    if (plan.M == 1 && plan.N == 1 && plan.K < 1024)
        return false;

    return true;
}

static void pack_contraction_operand(const Mat& src, Mat& packed, const int* free_chars, int free_count, const int* token_pos, const TwoBlobContraction& plan, const int* batch_index)
{
    // Free dimensions precede contracted dimensions: destination is row * K + k.
    int chars[4];
    size_t strides[4];
    const int count = free_count + plan.kcount;
    for (int d = 0; d < count; d++)
    {
        const int c = d < free_count ? free_chars[d] : plan.kchars[d - free_count];
        chars[d] = c;
        strides[d] = stride_at(src, token_pos[c]);
    }

    size_t offset = 0;
    for (int d = 0; d < plan.batch_count; d++)
    {
        const int c = plan.batch_chars[d];
        offset += (size_t)batch_index[c] * stride_at(src, token_pos[c]);
    }

    int index[16] = {0};
    const float* ptr = src;
    float* outptr = packed;
    const size_t total = (size_t)packed.w * packed.h;
    for (size_t i = 0; i < total; i++)
    {
        outptr[i] = ptr[offset];
        if (i + 1 < total)
            advance_index(chars, strides, plan.dim_sizes, count, index, offset);
    }
}

static void scatter_contraction_tile(const Mat& product, Mat& output, const std::string& rhs_token, const TwoBlobContraction& plan, const size_t* stride_out, const int* batch_index, size_t m_start, size_t n_start)
{
    int index[16];
    std::copy(batch_index, batch_index + 16, index);
    const float* ptr = product;
    float* outptr = output;
    for (int m = 0; m < product.h; m++)
    {
        size_t mm = m_start + m;
        for (int d = plan.aout_count - 1; d >= 0; d--)
        {
            const int c = plan.aout_chars[d];
            index[c] = (int)(mm % plan.dim_sizes[c]);
            mm /= plan.dim_sizes[c];
        }

        for (int n = 0; n < product.w; n++)
        {
            size_t nn = n_start + n;
            for (int d = plan.bout_count - 1; d >= 0; d--)
            {
                const int c = plan.bout_chars[d];
                index[c] = (int)(nn % plan.dim_sizes[c]);
                nn /= plan.dim_sizes[c];
            }

            size_t offset = 0;
            for (size_t p = 0; p < rhs_token.size(); p++)
                offset += (size_t)index[rhs_token[p] - 'i'] * stride_out[p];

            outptr[offset] = ptr[(size_t)m * product.w + n];
        }
    }
}

static int compute_contraction_gemm(Layer* gemm, const std::vector<Mat>& inputs, Mat& W, Mat& X, Mat& output, const std::string& rhs_token, const TwoBlobContraction& plan, const Option& opt)
{
    // Reuse the final output for the identity layout; other layouts scatter bounded tiles.
    const bool direct_output = plan.batch == 1 && rhs_token.size() == 2 && plan.aout_count == 1 && plan.bout_count == 1 && rhs_token[0] == (char)('i' + plan.aout_chars[0]) && rhs_token[1] == (char)('i' + plan.bout_chars[0]) && output.dims == 2 && output.h == (int)plan.M && output.w == (int)plan.N;
    const size_t max_gemm_tile_size = 256;
    const size_t tile_m = direct_output ? plan.M : std::min(plan.M, max_gemm_tile_size);
    const size_t tile_n = direct_output ? plan.N : std::min(plan.N, max_gemm_tile_size);

    size_t stride_out[4];
    for (size_t p = 0; p < rhs_token.size(); p++)
        stride_out[p] = stride_at(output, (int)p);

    std::vector<Mat> bottoms(2);
    std::vector<Mat> tops(1);
    if (direct_output)
        tops[0] = output;

    int batch_index[16] = {0};
    for (size_t bi = 0; bi < plan.batch; bi++)
    {
        pack_contraction_operand(inputs[0], W, plan.aout_chars, plan.aout_count, plan.token_pos_a, plan, batch_index);
        pack_contraction_operand(inputs[1], X, plan.bout_chars, plan.bout_count, plan.token_pos_b, plan, batch_index);

        for (size_t m = 0; m < plan.M; m += tile_m)
        {
            const int rows = (int)std::min(tile_m, plan.M - m);
            bottoms[0] = W.row_range((int)m, rows);
            for (size_t n = 0; n < plan.N; n += tile_n)
            {
                const int cols = (int)std::min(tile_n, plan.N - n);
                bottoms[1] = X.row_range((int)n, cols);
                const int ret = gemm->forward(bottoms, tops, opt);
                if (ret != 0)
                    return ret;

                const Mat& product = tops[0];
                if (product.dims != 2 || product.w != cols || product.h != rows || product.elemsize != 4u)
                    return -1;

                if (!direct_output)
                    scatter_contraction_tile(product, output, rhs_token, plan, stride_out, batch_index, m, n);
            }
        }

        if (bi + 1 < plan.batch)
            advance_char(plan.batch_chars, plan.dim_sizes, plan.batch_count, batch_index);
    }

    return 0;
}

// Return 0 when handled, 1 for the generic path, negative on allocation error.
static int forward_two_blob_contraction(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const std::vector<std::string>& lhs_tokens, const std::string& rhs_token, const Option& opt)
{
    TwoBlobContraction plan;
    if (!make_contraction_plan(bottom_blobs, lhs_tokens, rhs_token, plan))
        return 1;

    // Resolve Gemm before allocating: selective builds may omit it, and the
    // workspace pool retains freed packing buffers for the Net lifetime.
    Layer* gemm = ncnn::create_layer_cpu(LayerType::Gemm);
    if (!gemm)
        return 1;

    {
        ParamDict pd;
        pd.set(2, 0);   // transA
        pd.set(3, 1);   // transB, both operands are packed as (rows, K)
        pd.set(4, 0);   // constantA
        pd.set(5, 0);   // constantB
        pd.set(6, 1);   // constantC
        pd.set(10, -1); // constant_broadcast_type_C = null
        pd.set(12, 1);  // output_elempack

        gemm->load_param(pd);
    }

    Option gemm_opt = opt;
    // Small per-slice GEMMs lose to OpenMP barrier overhead when run across threads.
    if ((plan.M == 1 && plan.N == 1) || plan.M * plan.N * plan.K < 65536)
        gemm_opt.num_threads = 1;

    gemm->load_model(ModelBinFromMatArray(0));
    gemm->create_pipeline(gemm_opt);

    // Pack as (rows, K) for transB=1. Allocation failure can still fall back
    // without touching the output.
    Mat W((int)plan.K, (int)plan.M, 4u, opt.workspace_allocator);
    Mat X((int)plan.K, (int)plan.N, 4u, opt.workspace_allocator);
    if (W.empty() || X.empty())
    {
        gemm->destroy_pipeline(gemm_opt);
        delete gemm;
        return 1;
    }

    Mat& top_blob = top_blobs[0];
    {
        const int out_dims = (int)rhs_token.size();
        if (out_dims == 1)
            top_blob.create(plan.dim_sizes[rhs_token[0] - 'i'], 4u, opt.blob_allocator);
        if (out_dims == 2)
            top_blob.create(plan.dim_sizes[rhs_token[1] - 'i'], plan.dim_sizes[rhs_token[0] - 'i'], 4u, opt.blob_allocator);
        if (out_dims == 3)
            top_blob.create(plan.dim_sizes[rhs_token[2] - 'i'], plan.dim_sizes[rhs_token[1] - 'i'], plan.dim_sizes[rhs_token[0] - 'i'], 4u, opt.blob_allocator);
        if (out_dims == 4)
            top_blob.create(plan.dim_sizes[rhs_token[3] - 'i'], plan.dim_sizes[rhs_token[2] - 'i'], plan.dim_sizes[rhs_token[1] - 'i'], plan.dim_sizes[rhs_token[0] - 'i'], 4u, opt.blob_allocator);
    }
    if (top_blob.empty())
    {
        gemm->destroy_pipeline(gemm_opt);
        delete gemm;
        return -100;
    }

    const int ret = compute_contraction_gemm(gemm, bottom_blobs, W, X, top_blob, rhs_token, plan, gemm_opt);

    gemm->destroy_pipeline(gemm_opt);
    delete gemm;

    return ret == 0 ? 0 : 1;
}

int Einsum::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // assert bottom_blobs.size() == lhs_tokens.size()
    // assert top_blobs.size() == 1

    size_t elemsize = bottom_blobs[0].elemsize;

    if (lhs_tokens.empty() && rhs_token == "ii")
    {
        // assert bottom_blobs.size() == 1
        // assert bottom_blob.dims == 2
        // assert bottom_blob.w == bottom_blob.h

        // trace
        Mat& top_blob = top_blobs[0];
        top_blob.create(1, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const Mat& bottom_blob = bottom_blobs[0];

        float sum = 0.f;

        for (int i = 0; i < bottom_blob.h; i++)
        {
            sum += bottom_blob.row(i)[i];
        }

        top_blob[0] = sum;

        return 0;
    }

    if (bottom_blobs.size() == 2)
    {
        const int ret = forward_two_blob_contraction(bottom_blobs, top_blobs, lhs_tokens, rhs_token, opt);
        if (ret != 1)
            return ret;
    }

    // resolve dimension sizes
    std::vector<int> dim_sizes(16, 1); // map ijklmnopqrstuvwx -> dim_size
    int dim_sizes_count = 0;

    for (size_t b = 0; b < bottom_blobs.size(); b++)
    {
        const std::string& lhs_token = lhs_tokens[b];
        const Mat& bottom_blob = bottom_blobs[b];
        const int in_dims = bottom_blob.dims;

        for (int s = 0; s < in_dims; s++)
        {
            int dim_size = 1;
            if (in_dims == 1) dim_size = bottom_blob.w;
            if (in_dims == 2 && s == 0) dim_size = bottom_blob.h;
            if (in_dims == 2 && s == 1) dim_size = bottom_blob.w;
            if (in_dims == 3 && s == 0) dim_size = bottom_blob.c;
            if (in_dims == 3 && s == 1) dim_size = bottom_blob.h;
            if (in_dims == 3 && s == 2) dim_size = bottom_blob.w;
            if (in_dims == 4 && s == 0) dim_size = bottom_blob.c;
            if (in_dims == 4 && s == 1) dim_size = bottom_blob.d;
            if (in_dims == 4 && s == 2) dim_size = bottom_blob.h;
            if (in_dims == 4 && s == 3) dim_size = bottom_blob.w;

            int dim_sizes_index = lhs_token[s] - 'i';
            dim_sizes[dim_sizes_index] = dim_size;
            dim_sizes_count = std::max(dim_sizes_count, dim_sizes_index + 1);
        }
    }

    dim_sizes.resize(dim_sizes_count);

    const int out_dims = (int)rhs_token.size();

    std::vector<int> indexes(dim_sizes_count);

    if (out_dims == 1)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create(dim_sizes[0], elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        for (int i = 0; i < top_blob.w; i++)
        {
            indexes[0] = i;

            float sum = sum_dim(dim_sizes, 1, bottom_blobs, lhs_tokens, indexes);

            top_blob[i] = sum;
        }
    }

    if (out_dims == 2)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create(dim_sizes[1], dim_sizes[0], elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        for (int i = 0; i < top_blob.h; i++)
        {
            indexes[0] = i;

            for (int j = 0; j < top_blob.w; j++)
            {
                indexes[1] = j;

                float sum = sum_dim(dim_sizes, 2, bottom_blobs, lhs_tokens, indexes);

                top_blob.row(i)[j] = sum;
            }
        }
    }

    if (out_dims == 3)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create(dim_sizes[2], dim_sizes[1], dim_sizes[0], elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        for (int i = 0; i < top_blob.c; i++)
        {
            indexes[0] = i;

            for (int j = 0; j < top_blob.h; j++)
            {
                indexes[1] = j;

                for (int k = 0; k < top_blob.w; k++)
                {
                    indexes[2] = k;

                    float sum = sum_dim(dim_sizes, 3, bottom_blobs, lhs_tokens, indexes);

                    top_blob.channel(i).row(j)[k] = sum;
                }
            }
        }
    }

    if (out_dims == 4)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create(dim_sizes[3], dim_sizes[2], dim_sizes[1], dim_sizes[0], elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        for (int i = 0; i < top_blob.c; i++)
        {
            indexes[0] = i;

            for (int j = 0; j < top_blob.d; j++)
            {
                indexes[1] = j;

                for (int k = 0; k < top_blob.h; k++)
                {
                    indexes[2] = k;

                    for (int l = 0; l < top_blob.w; l++)
                    {
                        indexes[3] = l;

                        float sum = sum_dim(dim_sizes, 4, bottom_blobs, lhs_tokens, indexes);

                        top_blob.channel(i).depth(j).row(k)[l] = sum;
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
