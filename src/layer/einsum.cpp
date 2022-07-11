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

#include "einsum.h"
#include <string.h>

namespace ncnn {

Einsum::Einsum()
{
    one_blob_only = false;
    support_inplace = false;
}

int Einsum::load_param(const ParamDict& pd)
{
    Mat equation_mat = pd.get(0, Mat());

    const int equation_len = equation_mat.w;

    // restore to lexical equation string
    std::string equation;
    equation.resize(equation_len);
    char* equation_ptr = (char*)equation.c_str();
    {
        const int* p = equation_mat;
        for (int i = 0; i < equation_len; i++)
        {
            equation_ptr[i] = p[i];
        }
    }

    if (equation == "ii")
    {
        // trace
        rhs_token = "ii";

        return 0;
    }

    // split into tokens
    char* arrow = strstr(equation_ptr, "->");
    if (!arrow)
    {
        NCNN_LOGE("invalid equation %s", equation_ptr);
        return -1;
    }

    arrow[0] = '\0';
    arrow[1] = '\0';

    char* lhs = equation_ptr;
    char* rhs = arrow + 2;

    {
        char* t = strtok(lhs, ",");
        while (t)
        {
            lhs_tokens.push_back(std::string(t));
            t = strtok(NULL, ",");
        }
    }

    rhs_token = std::string(rhs);

    // check token always in ijkl
    {
        for (size_t i = 0; i < rhs_token.size(); i++)
        {
            if (rhs_token[i] < 'i' || rhs_token[i] > 'l')
            {
                NCNN_LOGE("invalid rhs_token %s", rhs_token.c_str());
                return -1;
            }
        }

        for (size_t i = 0; i < lhs_tokens.size(); i++)
        {
            const std::string& lhs_token = lhs_tokens[i];
            for (size_t j = 0; j < lhs_token.size(); j++)
            {
                if (lhs_token[j] < 'i' || lhs_token[j] > 'x')
                {
                    NCNN_LOGE("invalid lhs_token %s", lhs_token.c_str());
                    return -1;
                }
            }
        }
    }

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
