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

#include "c_api.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn_net_t squeezenet = ncnn_net_create();

    ncnn_option_t opt = ncnn_option_create();
    ncnn_option_set_use_vulkan_compute(opt, 1);

    ncnn_net_set_option(squeezenet, opt);

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    ncnn_net_load_param(squeezenet, "squeezenet_v1.1.param");
    ncnn_net_load_model(squeezenet, "squeezenet_v1.1.bin");

    ncnn_mat_t in = ncnn_mat_from_pixels_resize(bgr.data, NCNN_MAT_PIXEL_BGR, bgr.cols, bgr.rows, bgr.cols * 3, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    ncnn_mat_substract_mean_normalize(in, mean_vals, 0);

    ncnn_extractor_t ex = ncnn_extractor_create(squeezenet);

    ncnn_extractor_input(ex, "data", in);

    ncnn_mat_t out;
    ncnn_extractor_extract(ex, "prob", &out);

    const int out_w = ncnn_mat_get_w(out);
    const float* out_data = (const float*)ncnn_mat_get_data(out);

    cls_scores.resize(out_w);
    for (int j = 0; j < out_w; j++)
    {
        cls_scores[j] = out_data[j];
    }

    ncnn_mat_destroy(in);
    ncnn_mat_destroy(out);

    ncnn_extractor_destroy(ex);

    ncnn_option_destroy(opt);

    ncnn_net_destroy(squeezenet);

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

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

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores);

    print_topk(cls_scores, 3);

    return 0;
}
