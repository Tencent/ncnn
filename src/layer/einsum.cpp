// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "einsum.h"

#include "layer_type.h"

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

// the generic path below visits every index tuple for every output element and
// resolves the operand tokens with string comparisons, which costs far more
// than the arithmetic itself. a plain two operand contraction is packed into
// two dense (rows, K) blocks with the contracted dimensions innermost and
// reduced with a contiguous dot product instead. the conditions are: exactly
// two fp32 operands, dims 1..4, token length equal to dims, no repeated index
// inside one token, every index that appears in only one operand must be in the
// output, indices shared by both operands and absent from the output are
// contracted, and 1..4 output dimensions. anything else falls back unchanged.

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

// return 0 when handled, 1 when the equation is not a plain two operand
// contraction, negative on error
static int forward_two_blob_contraction(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const std::vector<std::string>& lhs_tokens, const std::string& rhs_token, const Option& opt)
{
    if (lhs_tokens.size() != 2)
        return 1;

    const Mat& A = bottom_blobs[0];
    const Mat& B = bottom_blobs[1];

    if (A.elemsize != 4u || B.elemsize != 4u)
        return 1;
    if (A.elempack != 1 || B.elempack != 1)
        return 1;
    if (A.dims < 1 || A.dims > 4 || B.dims < 1 || B.dims > 4)
        return 1;
    if ((int)lhs_tokens[0].size() != A.dims || (int)lhs_tokens[1].size() != B.dims)
        return 1;
    if (rhs_token.empty() || rhs_token.size() > 4)
        return 1;

    const std::string& token_a = lhs_tokens[0];
    const std::string& token_b = lhs_tokens[1];

    int dim_sizes[16];
    int token_pos_a[16];
    int token_pos_b[16];
    bool in_a[16];
    bool in_b[16];
    bool in_out[16];

    for (int c = 0; c < 16; c++)
    {
        dim_sizes[c] = 1;
        token_pos_a[c] = -1;
        token_pos_b[c] = -1;
        in_a[c] = false;
        in_b[c] = false;
        in_out[c] = false;
    }

    for (int p = 0; p < (int)token_a.size(); p++)
    {
        const int c = token_a[p] - 'i';
        if (c < 0 || c >= 16)
            return 1;
        if (in_a[c])
            return 1; // a repeated index inside one operand is a diagonal, not handled here

        const int s = dim_size_at(A, p);
        if (s <= 0)
            return 1;

        in_a[c] = true;
        token_pos_a[c] = p;
        dim_sizes[c] = s;
    }

    for (int p = 0; p < (int)token_b.size(); p++)
    {
        const int c = token_b[p] - 'i';
        if (c < 0 || c >= 16)
            return 1;
        if (in_b[c])
            return 1;

        const int s = dim_size_at(B, p);
        if (s <= 0)
            return 1;
        if (in_a[c] && dim_sizes[c] != s)
            return 1; // the operands disagree on this dimension

        in_b[c] = true;
        token_pos_b[c] = p;
        dim_sizes[c] = s;
    }

    for (size_t p = 0; p < rhs_token.size(); p++)
    {
        const int c = rhs_token[p] - 'i';
        if (c < 0 || c >= 16)
            return 1;
        if (in_out[c])
            return 1; // a repeated output index is not a canonical equation
        if (!in_a[c] && !in_b[c])
            return 1; // an output index has to come from an operand

        in_out[c] = true;
    }

    int batch_chars[16];
    int aout_chars[16];
    int bout_chars[16];
    int kchars[16];
    int batch_count = 0;
    int aout_count = 0;
    int bout_count = 0;
    int kcount = 0;

    for (int c = 0; c < 16; c++)
    {
        if (!in_a[c] && !in_b[c] && !in_out[c])
            continue;

        if (in_a[c] && in_b[c])
        {
            if (in_out[c])
                batch_chars[batch_count++] = c;
            else
                kchars[kcount++] = c;

            continue;
        }

        if (!in_out[c])
            return 1; // a reduction over one operand only, the generic path handles it

        if (in_a[c])
            aout_chars[aout_count++] = c;
        else
            bout_chars[bout_count++] = c;
    }

    size_t M = 1;
    size_t N = 1;
    size_t K = 1;
    size_t batch = 1;
    for (int d = 0; d < aout_count; d++)
        M *= dim_sizes[aout_chars[d]];
    for (int d = 0; d < bout_count; d++)
        N *= dim_sizes[bout_chars[d]];
    for (int d = 0; d < kcount; d++)
        K *= dim_sizes[kchars[d]];
    for (int d = 0; d < batch_count; d++)
        batch *= dim_sizes[batch_chars[d]];

    if (M * K > (size_t)INT_MAX || N * K > (size_t)INT_MAX || M * N > (size_t)INT_MAX)
        return 1;

    // a slice with almost no reduction left is cheaper on the generic path,
    // which pays nothing for packing and per slice bookkeeping. this keeps
    // equations where every index is a batch index (an elementwise product with
    // a large batch) from regressing behind the interpreter
    if (M * N * K < 256)
        return 1;

    // the reduction is delegated to the library's own Gemm, the way
    // src/layer/x86/matmul_x86.cpp drives its own Gemm, which reaches the arch
    // optimized kernels that a generic layer translation unit cannot. a
    // selective build can compile Gemm out, so resolve it before reserving
    // anything: otherwise a build without Gemm would hand |A| + |B| floats to
    // the workspace pool, which caches them for the lifetime of the net
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

    gemm->load_model(ModelBinFromMatArray(0));
    gemm->create_pipeline(opt);

    // dims 2 with w = K, h = rows, which is the layout Gemm reads and the layout
    // the packing loops below write with a linear index. allocate before
    // touching the output, so that a failed allocation falls back instead of
    // failing a graph the generic path still computes
    Mat W((int)K, (int)M, 4u, opt.workspace_allocator);
    Mat X((int)K, (int)N, 4u, opt.workspace_allocator);
    if (W.empty() || X.empty())
    {
        gemm->destroy_pipeline(opt);
        delete gemm;
        return 1;
    }

    Mat& top_blob = top_blobs[0];
    {
        const int out_dims = (int)rhs_token.size();
        if (out_dims == 1)
            top_blob.create(dim_sizes[rhs_token[0] - 'i'], 4u, opt.blob_allocator);
        if (out_dims == 2)
            top_blob.create(dim_sizes[rhs_token[1] - 'i'], dim_sizes[rhs_token[0] - 'i'], 4u, opt.blob_allocator);
        if (out_dims == 3)
            top_blob.create(dim_sizes[rhs_token[2] - 'i'], dim_sizes[rhs_token[1] - 'i'], dim_sizes[rhs_token[0] - 'i'], 4u, opt.blob_allocator);
        if (out_dims == 4)
            top_blob.create(dim_sizes[rhs_token[3] - 'i'], dim_sizes[rhs_token[2] - 'i'], dim_sizes[rhs_token[1] - 'i'], dim_sizes[rhs_token[0] - 'i'], 4u, opt.blob_allocator);
    }
    if (top_blob.empty())
    {
        gemm->destroy_pipeline(opt);
        delete gemm;
        return -100;
    }

    // free dimensions first, then the contracted ones, so that the destination
    // index is m * K + k
    int list_a_char[4];
    size_t list_a_stride[4];
    int list_a_count = 0;
    for (int d = 0; d < aout_count; d++)
    {
        const int c = aout_chars[d];
        list_a_char[list_a_count] = c;
        list_a_stride[list_a_count] = stride_at(A, token_pos_a[c]);
        list_a_count++;
    }
    for (int d = 0; d < kcount; d++)
    {
        const int c = kchars[d];
        list_a_char[list_a_count] = c;
        list_a_stride[list_a_count] = stride_at(A, token_pos_a[c]);
        list_a_count++;
    }

    int list_b_char[4];
    size_t list_b_stride[4];
    int list_b_count = 0;
    for (int d = 0; d < bout_count; d++)
    {
        const int c = bout_chars[d];
        list_b_char[list_b_count] = c;
        list_b_stride[list_b_count] = stride_at(B, token_pos_b[c]);
        list_b_count++;
    }
    for (int d = 0; d < kcount; d++)
    {
        const int c = kchars[d];
        list_b_char[list_b_count] = c;
        list_b_stride[list_b_count] = stride_at(B, token_pos_b[c]);
        list_b_count++;
    }

    size_t stride_out[4];
    for (int p = 0; p < (int)rhs_token.size(); p++)
        stride_out[p] = stride_at(top_blob, p);

    int char_index[16];
    for (int c = 0; c < 16; c++)
        char_index[c] = 0;

    const float* aptr = A;
    const float* bptr = B;
    float* wptr = W;
    float* xptr = X;
    float* outptr = top_blob;

    int gemm_ret = 0;

    for (size_t bi = 0; bi < batch; bi++)
    {
        size_t base_a = 0;
        size_t base_b = 0;
        for (int d = 0; d < batch_count; d++)
        {
            const int c = batch_chars[d];
            if (token_pos_a[c] >= 0)
                base_a += (size_t)char_index[c] * stride_at(A, token_pos_a[c]);
            if (token_pos_b[c] >= 0)
                base_b += (size_t)char_index[c] * stride_at(B, token_pos_b[c]);
        }

        {
            for (int d = 0; d < list_a_count; d++)
                char_index[list_a_char[d]] = 0;

            const size_t total = M * K;
            size_t off = base_a;
            for (size_t i = 0; i < total; i++)
            {
                wptr[i] = aptr[off];
                if (i + 1 < total)
                    advance_index(list_a_char, list_a_stride, dim_sizes, list_a_count, char_index, off);
            }
        }

        {
            for (int d = 0; d < list_b_count; d++)
                char_index[list_b_char[d]] = 0;

            const size_t total = N * K;
            size_t off = base_b;
            for (size_t i = 0; i < total; i++)
            {
                xptr[i] = bptr[off];
                if (i + 1 < total)
                    advance_index(list_b_char, list_b_stride, dim_sizes, list_b_count, char_index, off);
            }
        }

        std::vector<Mat> gemm_bottoms(2);
        gemm_bottoms[0] = W;
        gemm_bottoms[1] = X;

        std::vector<Mat> gemm_tops(1);

        gemm_ret = gemm->forward(gemm_bottoms, gemm_tops, opt);
        if (gemm_ret != 0)
            break;

        const Mat& product = gemm_tops[0];
        if (product.dims != 2 || product.w != (int)N || product.h != (int)M || product.elemsize != 4u)
        {
            gemm_ret = -1;
            break;
        }

        const float* cptr = product;

        for (size_t m = 0; m < M; m++)
        {
            size_t mm = m;
            for (int d = aout_count - 1; d >= 0; d--)
            {
                const int c = aout_chars[d];
                char_index[c] = (int)(mm % dim_sizes[c]);
                mm /= dim_sizes[c];
            }

            for (size_t n = 0; n < N; n++)
            {
                size_t nn = n;
                for (int d = bout_count - 1; d >= 0; d--)
                {
                    const int c = bout_chars[d];
                    char_index[c] = (int)(nn % dim_sizes[c]);
                    nn /= dim_sizes[c];
                }

                size_t off = 0;
                for (size_t p = 0; p < rhs_token.size(); p++)
                    off += (size_t)char_index[rhs_token[p] - 'i'] * stride_out[p];

                outptr[off] = cptr[m * N + n];
            }
        }

        if (bi + 1 < batch)
            advance_char(batch_chars, dim_sizes, batch_count, char_index);
    }

    gemm->destroy_pipeline(opt);
    delete gemm;

    if (gemm_ret != 0)
        return 1;

    return 0;
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

        // not a plain pair contraction, keep going with the generic path below
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
