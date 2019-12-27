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

#include "layer/convolution.h"

static int test_convolution_0()
{
    ncnn::Mat a(6, 7, 15);
    Randomize(a);

    ncnn::ParamDict pd;
    pd.set(0, 15);// num_output
    pd.set(1, 1);// kernel_w
    pd.set(11, 1);// kernel_h
    pd.set(2, 1);// dilation_w
    pd.set(12, 1);// dilation_h
    pd.set(3, 1);// stride_w
    pd.set(13, 1);// stride_h
    pd.set(4, 0);// pad_w
    pd.set(14, 0);// pad_h
    pd.set(5, 1);// bias_term
    pd.set(6, 15*15*1*1);

    ncnn::Mat weights[2];
    weights[0] = ncnn::Mat(15*15*1*1);
    weights[1] = ncnn::Mat(15);

    Randomize(weights[0]);
    Randomize(weights[1]);

    ncnn::ModelBinFromMatArray mb(weights);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;

    return test_layer<ncnn::Convolution>("Convolution", pd, mb, opt, a);
}

int main()
{
    SRAND(7767517);

    return test_convolution_0();
}
