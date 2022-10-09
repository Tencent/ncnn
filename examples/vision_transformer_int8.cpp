// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "net.h"

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>

static int classify_vit(const cv::Mat& bgr)
{
    ncnn::Net vit;

    vit.opt.use_vulkan_compute = false;

    // the ncnn model https://github.com/tpoisonooo/mmdeploy-onnx2ncnn-testdata/tree/main/vit-int8-20220811
    vit.load_param("vit8.param");
    vit.load_model("vit8.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 384, 384);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = vit.create_extractor();

    ex.input("input", in);

    ncnn::Mat multiHeadOut;

    ex.extract("output", multiHeadOut);

    float max_value = multiHeadOut[0];
    int max_index = 0;
    for (int j = 0; j < multiHeadOut.w; j++)
    {
        if (max_value < multiHeadOut[j])
        {
            max_value = multiHeadOut[j];
            max_index = j;
        }
    }
    fprintf(stdout, "softmax result: %d %f\n", max_index, max_value);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    classify_vit(m);
    return 0;
}
