// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "testutil.h"

#include "layer/reshape.h"

static int test_reshape(const ncnn::Mat& a, int outw, int outh, int outc, bool use_packing_layout)
{
    ncnn::ParamDict pd;
    pd.set(0, outw);// w
    pd.set(1, outh);// h
    pd.set(2, outc);// c

    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = use_packing_layout;

    int ret = test_layer<ncnn::Reshape>("Reshape", pd, mb, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reshape failed a.dims=%d a=(%d %d %d) outw=%d outh=%d outc=%d use_packing_layout=%d\n", a.dims, a.w, a.h, a.c, outw, outh, outc, use_packing_layout);
    }

    return ret;
}

static int test_reshape_0()
{
    ncnn::Mat a = RandomMat(3, 7, 16);

    return 0
        || test_reshape(a, 7, 3, 16, false)
        || test_reshape(a, 3, 16, 7, false)
        || test_reshape(a, 16, 7, 3, false)
        || test_reshape(a, 2, 3, -1, false)
        || test_reshape(a, -1, 8, 2, false)
        || test_reshape(a, -1, 4, -233, false)
        || test_reshape(a, 8, -1, -233, false)
        || test_reshape(a, 16, 21, -233, false)
        || test_reshape(a, -1, -233, -233, false)

        || test_reshape(a, 7, 3, 16, true)
        || test_reshape(a, 3, 16, 7, true)
        || test_reshape(a, 16, 7, 3, true)
        || test_reshape(a, 2, 3, -1, true)
        || test_reshape(a, -1, 8, 2, true)
        || test_reshape(a, -1, 4, -233, true)
        || test_reshape(a, 8, -1, -233, true)
        || test_reshape(a, 16, 21, -233, true)
        || test_reshape(a, -1, -233, -233, true)
        ;
}

static int test_reshape_1()
{
    ncnn::Mat a = RandomMat(4, 14, 13);

    return 0
        || test_reshape(a, 14, 4, 13, false)
        || test_reshape(a, 4, 13, 14, false)
        || test_reshape(a, 13, 14, 4, false)
        || test_reshape(a, 2, 7, -1, false)
        || test_reshape(a, -1, 13, 2, false)
        || test_reshape(a, -1, 4, -233, false)
        || test_reshape(a, 8, -1, -233, false)
        || test_reshape(a, 8, 91, -233, false)
        || test_reshape(a, -1, -233, -233, false)

        || test_reshape(a, 14, 4, 13, true)
        || test_reshape(a, 4, 13, 14, true)
        || test_reshape(a, 13, 14, 4, true)
        || test_reshape(a, 2, 7, -1, true)
        || test_reshape(a, -1, 13, 2, true)
        || test_reshape(a, -1, 4, -233, true)
        || test_reshape(a, 8, -1, -233, true)
        || test_reshape(a, 8, 91, -233, true)
        || test_reshape(a, -1, -233, -233, true)
        ;
}

static int test_reshape_2()
{
    ncnn::Mat a = RandomMat(14, 16);

    return 0
        || test_reshape(a, 7, 2, 16, false)
        || test_reshape(a, 2, 16, 7, false)
        || test_reshape(a, 16, 7, 2, false)
        || test_reshape(a, 2, 4, -1, false)
        || test_reshape(a, -1, 8, 2, false)
        || test_reshape(a, 28, 8, -233, false)
        || test_reshape(a, -1, 7, -233, false)
        || test_reshape(a, 16, -1, -233, false)
        || test_reshape(a, -1, -233, -233, false)

        || test_reshape(a, 7, 2, 16, true)
        || test_reshape(a, 2, 16, 7, true)
        || test_reshape(a, 16, 7, 2, true)
        || test_reshape(a, 2, 4, -1, true)
        || test_reshape(a, -1, 8, 2, true)
        || test_reshape(a, 28, 8, -233, true)
        || test_reshape(a, -1, 7, -233, true)
        || test_reshape(a, 16, -1, -233, true)
        || test_reshape(a, -1, -233, -233, true)
        ;
}

static int test_reshape_3()
{
    ncnn::Mat a = RandomMat(12, 14);

    return 0
        || test_reshape(a, 7, 2, 12, false)
        || test_reshape(a, 2, 12, 7, false)
        || test_reshape(a, 12, 7, 2, false)
        || test_reshape(a, 2, 4, -1, false)
        || test_reshape(a, -1, 4, 2, false)
        || test_reshape(a, 21, 8, -233, false)
        || test_reshape(a, -1, 7, -233, false)
        || test_reshape(a, 3, -1, -233, false)
        || test_reshape(a, -1, -233, -233, false)

        || test_reshape(a, 7, 2, 12, true)
        || test_reshape(a, 2, 12, 7, true)
        || test_reshape(a, 12, 7, 2, true)
        || test_reshape(a, 2, 4, -1, true)
        || test_reshape(a, -1, 4, 2, true)
        || test_reshape(a, 21, 8, -233, true)
        || test_reshape(a, -1, 7, -233, true)
        || test_reshape(a, 3, -1, -233, true)
        || test_reshape(a, -1, -233, -233, true)
        ;
}

static int test_reshape_4()
{
    ncnn::Mat a = RandomMat(120);

    return 0
        || test_reshape(a, 3, 5, 8, false)
        || test_reshape(a, 3, 8, 5, false)
        || test_reshape(a, 8, 5, 3, false)
        || test_reshape(a, 2, 5, -1, false)
        || test_reshape(a, -1, 5, 2, false)
        || test_reshape(a, 4, 30, -233, false)
        || test_reshape(a, -1, 2, -233, false)
        || test_reshape(a, 24, -1, -233, false)
        || test_reshape(a, -1, -233, -233, false)

        || test_reshape(a, 3, 5, 8, true)
        || test_reshape(a, 3, 8, 5, true)
        || test_reshape(a, 8, 5, 3, true)
        || test_reshape(a, 2, 5, -1, true)
        || test_reshape(a, -1, 5, 2, true)
        || test_reshape(a, 4, 30, -233, true)
        || test_reshape(a, -1, 2, -233, true)
        || test_reshape(a, 24, -1, -233, true)
        || test_reshape(a, -1, -233, -233, true)
        ;
}

static int test_reshape_5()
{
    ncnn::Mat a = RandomMat(210);

    return 0
        || test_reshape(a, 3, 5, 14, false)
        || test_reshape(a, 3, 14, 5, false)
        || test_reshape(a, 14, 5, 3, false)
        || test_reshape(a, 2, 5, -1, false)
        || test_reshape(a, -1, 5, 2, false)
        || test_reshape(a, 6, 35, -233, false)
        || test_reshape(a, -1, 7, -233, false)
        || test_reshape(a, 21, -1, -233, false)
        || test_reshape(a, -1, -233, -233, false)

        || test_reshape(a, 3, 5, 14, true)
        || test_reshape(a, 3, 14, 5, true)
        || test_reshape(a, 14, 5, 3, true)
        || test_reshape(a, 2, 5, -1, true)
        || test_reshape(a, -1, 5, 2, true)
        || test_reshape(a, 6, 35, -233, true)
        || test_reshape(a, -1, 7, -233, true)
        || test_reshape(a, 21, -1, -233, true)
        || test_reshape(a, -1, -233, -233, true)
        ;
}

int main()
{
    SRAND(7767517);

    return 0
        || test_reshape_0()
        || test_reshape_1()
        || test_reshape_2()
        || test_reshape_3()
        || test_reshape_4()
        || test_reshape_5()
        ;
}
