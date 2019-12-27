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

#include "testutil.h"

#include "layer/relu.h"

static int test_relu_0()
{
    ncnn::Mat a = RandomMat(6, 7, 8);

    ncnn::ParamDict pd;
    pd.set(0, 0.f);//slope

    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;

    return test_layer<ncnn::ReLU>("ReLU", pd, mb, opt, a);
}

static int test_relu_1()
{
    ncnn::Mat a = RandomMat(6, 7, 8);

    ncnn::ParamDict pd;
    pd.set(0, 0.1f);//slope

    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;

    return test_layer<ncnn::ReLU>("ReLU", pd, mb, opt, a);
}

int main()
{
    SRAND(7767517);

    return test_relu_0() || test_relu_1();
}
