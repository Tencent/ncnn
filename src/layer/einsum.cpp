// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "einsum.h"

namespace ncnn {

Einsum::Einsum()
{
    one_blob_only = false;
    support_inplace = false;
}

int Einsum::load_param(const ParamDict& pd)
{
    Mat equation_mat = pd.get(0, Mat());

    {
        const int equation_mat_type = pd.type(0);
        if (equation_mat_type != 0 && equation_mat_type != 4 && equation_mat_type != 5)
            return -1;

        if ((equation_mat.dims != 0 || equation_mat.w != 0 || equation_mat.data) && (equation_mat.dims != 1 || equation_mat.w < 0 || equation_mat.elempack != 1 || equation_mat.elemsize != 4u || (equation_mat.w > 0 && !equation_mat.data)))
            return -1;
    }

    if (equation_mat.empty())
        return -1;

    // validate character values before narrowing to char
    const int* p = equation_mat;
    std::string equation;
    equation.resize(equation_mat.w);
    for (int i = 0; i < equation_mat.w; i++)
    {
        if ((p[i] < 'i' || p[i] > 'x') && p[i] != ',' && p[i] != '-' && p[i] != '>')
            return -1;
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
    bool seen[16] = {false};
    int arrow = -1;
    for (int i = 0; i < equation_mat.w; i++)
    {
        const char ch = equation[i];
        if (ch == ',' || ch == '-')
        {
            if (token.empty() || token.size() > 4)
                return -1;
            tokens.push_back(token);
            token.clear();
            if (ch == '-')
            {
                if (i + 1 >= equation_mat.w || equation[i + 1] != '>')
                    return -1;
                arrow = i;
                break;
            }
        }
        else
        {
            if (ch < 'i' || ch > 'x')
                return -1;
            token.push_back(ch);
            seen[ch - 'i'] = true;
        }
    }

    if (arrow < 0)
        return -1;

    const int output_dims = equation_mat.w - arrow - 2;
    if (output_dims < 1 || output_dims > 4)
        return -1;

    // the implementation emits dimensions in the canonical i,j,k,l order
    std::string output;
    output.resize(output_dims);
    for (int i = 0; i < output_dims; i++)
    {
        if (equation[arrow + 2 + i] != 'i' + i || !seen[i])
            return -1;
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
