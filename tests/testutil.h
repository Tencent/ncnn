// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TESTUTIL_H
#define TESTUTIL_H

#include <math.h>
#include <stdio.h>

#include <algorithm>

#include "prng.h"

#include "mat.h"
#include "layer.h"

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND() prng_rand(&g_prng_rand_state)

static float RandomFloat(float a = -2, float b = 2)
{
    float random = ((float) RAND()) / (float) uint64_t(-1);//RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

static void Randomize(ncnn::Mat& m)
{
    for (size_t i=0; i<m.total(); i++)
    {
        m[i] = RandomFloat();
    }
}

static bool FloatNearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    // relative error
    float diff = fabs(a - b);

    return diff < epsilon * std::max(fabs(a), fabs(b));
}

static int CompareMat(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001)
{
#define CHECK_MEMBER(m) \
    if (a.m != b.m) \
    { \
        fprintf(stderr, #m" not match %d %d\n", (int)a.m, (int)b.m); \
        return -1; \
    }

    CHECK_MEMBER(dims)
    CHECK_MEMBER(w)
    CHECK_MEMBER(h)
    CHECK_MEMBER(c)
    CHECK_MEMBER(elemsize)
    CHECK_MEMBER(elempack)

#undef CHECK_MEMBER

    for (int q=0; q<a.c; q++)
    {
        const ncnn::Mat ma = a.channel(q);
        const ncnn::Mat mb = b.channel(q);
        for (int i=0; i<a.h; i++)
        {
            const float* pa = ma.row(i);
            const float* pb = mb.row(i);
            for (int j=0; j<a.w; j++)
            {
                if (!FloatNearlyEqual(pa[j], pb[j], epsilon))
                {
                    fprintf(stderr, "value not match at %d %d %d    %f %f\n", q, i, j, pa[j], pb[j]);
                    return -1;
                }
            }
        }
    }

    return 0;
}

template <typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const ncnn::ModelBin& mb, const ncnn::Option& opt, const std::vector<ncnn::Mat>& a)
{
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (op->one_blob_only && a.size() != 1)
    {
        fprintf(stderr, "layer with one_blob_only but consume multiple inputs\n");
        return -1;
    }

    op->load_param(pd);

    op->load_model(mb);

    op->create_pipeline(opt);

    std::vector<ncnn::Mat> b;
    ((T*)op)->T::forward(a, b, opt);

    std::vector<ncnn::Mat> c;
    op->forward(a, c, opt);

    op->destroy_pipeline(opt);

    delete op;

    if (b.size() != c.size())
    {
        fprintf(stderr, "output blob count not match %zu %zu\n", b.size(), c.size());
        return -1;
    }

    for (size_t i=0; i<b.size(); i++)
    {
        if (CompareMat(b[i], c[i]))
            return -1;
    }

    return 0;
}

template <typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const ncnn::ModelBin& mb, const ncnn::Option& opt, const ncnn::Mat& a)
{
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    op->load_param(pd);

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat b;
    ((T*)op)->T::forward(a, b, opt);

    ncnn::Mat c;
    op->forward(a, c, opt);

    op->destroy_pipeline(opt);

    delete op;

    return CompareMat(b, c);
}

template <typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const ncnn::ModelBin& mb, const ncnn::Option& opt, const std::vector<ncnn::Mat>& a)
{
    return test_layer<T>(ncnn::layer_to_index(layer_type), pd, mb, opt, a);
}

template <typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const ncnn::ModelBin& mb, const ncnn::Option& opt, const ncnn::Mat& a)
{
    return test_layer<T>(ncnn::layer_to_index(layer_type), pd, mb, opt, a);
}

#endif // TESTUTIL_H
