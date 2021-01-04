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

#include "layer/priorbox.h"
#include "testutil.h"

static int test_priorbox_caffe()
{
    ncnn::Mat min_sizes(1);
    min_sizes[0] = 105.f;

    ncnn::Mat max_sizes(1);
    max_sizes[0] = 150.f;

    ncnn::Mat aspect_ratios(2);
    aspect_ratios[0] = 2.f;
    aspect_ratios[1] = 3.f;

    ncnn::ParamDict pd;
    pd.set(0, min_sizes);
    pd.set(1, max_sizes);
    pd.set(2, aspect_ratios);
    pd.set(3, 0.1f);    // variances[0]
    pd.set(4, 0.1f);    // variances[1]
    pd.set(5, 0.2f);    // variances[2]
    pd.set(6, 0.2f);    // variances[3]
    pd.set(7, 1);       // flip
    pd.set(8, 0);       // clip
    pd.set(9, -233);    // image_width
    pd.set(10, -233);   // image_height
    pd.set(11, -233.f); // step_width
    pd.set(12, -233.f); // step_height
    pd.set(13, 0.f);    // offset
    pd.set(14, 0.f);    // step_mmdetection
    pd.set(15, 0.f);    // center_mmdetection

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = RandomMat(72, 72, 1);
    as[1] = RandomMat(512, 512, 1);

    int ret = test_layer<ncnn::PriorBox>("PriorBox", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_priorbox_caffe failed\n");
    }

    return ret;
}

static int test_priorbox_mxnet()
{
    ncnn::Mat min_sizes(2);
    min_sizes[0] = 0.15f;
    min_sizes[1] = 0.2121f;

    ncnn::Mat max_sizes(0);

    ncnn::Mat aspect_ratios(5);
    aspect_ratios[0] = 1.f;
    aspect_ratios[1] = 2.f;
    aspect_ratios[2] = 0.5f;
    aspect_ratios[3] = 3.f;
    aspect_ratios[4] = 0.333333;

    ncnn::ParamDict pd;
    pd.set(0, min_sizes);
    pd.set(1, max_sizes);
    pd.set(2, aspect_ratios);
    pd.set(3, 0.1f);    // variances[0]
    pd.set(4, 0.1f);    // variances[1]
    pd.set(5, 0.2f);    // variances[2]
    pd.set(6, 0.2f);    // variances[3]
    pd.set(7, 0);       // flip
    pd.set(8, 0);       // clip
    pd.set(9, -233);    // image_width
    pd.set(10, -233);   // image_height
    pd.set(11, -233.f); // step_width
    pd.set(12, -233.f); // step_height
    pd.set(13, 0.5f);   // offset
    pd.set(14, 0.f);    // step_mmdetection
    pd.set(15, 0.f);    // center_mmdetection

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(1);
    as[0] = RandomMat(72, 72, 1);

    int ret = test_layer<ncnn::PriorBox>("PriorBox", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_priorbox_mxnet failed\n");
    }

    return ret;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_priorbox_caffe()
           || test_priorbox_mxnet();
}
