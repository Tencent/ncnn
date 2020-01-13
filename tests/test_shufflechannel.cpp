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

#include "layer/shufflechannel.h"

static int test_shufflechannel(int w, int h, int c, int group, bool use_packing_layout)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, group);// group

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

    int ret = test_layer<ncnn::ShuffleChannel>("ShuffleChannel", pd, mb, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_shufflechannel failed w=%d h=%d c=%d group=%d use_packing_layout=%d\n", w, h, c, group, use_packing_layout);
    }

    return ret;
}

static int test_shufflechannel_0()
{
    return 0
        || test_shufflechannel(3, 7, 1, 1, false)
        || test_shufflechannel(3, 7, 2, 2, false)
        || test_shufflechannel(3, 7, 3, 3, false)
        || test_shufflechannel(3, 7, 4, 2, false)
        || test_shufflechannel(3, 7, 12, 3, false)
        || test_shufflechannel(3, 7, 15, 3, false)
        || test_shufflechannel(3, 7, 16, 2, false)

        || test_shufflechannel(3, 7, 1, 1, true)
        || test_shufflechannel(3, 7, 2, 2, true)
        || test_shufflechannel(3, 7, 3, 3, true)
        || test_shufflechannel(3, 7, 4, 2, true)
        || test_shufflechannel(3, 7, 12, 3, true)
        || test_shufflechannel(3, 7, 15, 3, true)
        || test_shufflechannel(3, 7, 16, 2, true)
        ;
}

int main()
{
    SRAND(7767517);

    return test_shufflechannel_0();
}
