// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <vector>

struct CrowdPoint
{
    cv::Point pt;
    float prob;
};

static void shift(int w, int h, int stride, std::vector<float> anchor_points, std::vector<float>& shifted_anchor_points)
{
    std::vector<float> x_, y_;
    for (int i = 0; i < w; i++)
    {
        float x = (i + 0.5) * stride;
        x_.push_back(x);
    }
    for (int i = 0; i < h; i++)
    {
        float y = (i + 0.5) * stride;
        y_.push_back(y);
    }

    std::vector<float> shift_x((size_t)w * h, 0), shift_y((size_t)w * h, 0);
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            shift_x[i * w + j] = x_[j];
        }
    }
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            shift_y[i * w + j] = y_[i];
        }
    }

    std::vector<float> shifts((size_t)w * h * 2, 0);
    for (int i = 0; i < w * h; i++)
    {
        shifts[i * 2] = shift_x[i];
        shifts[i * 2 + 1] = shift_y[i];
    }

    shifted_anchor_points.resize((size_t)2 * w * h * anchor_points.size() / 2, 0);
    for (int i = 0; i < w * h; i++)
    {
        for (int j = 0; j < anchor_points.size() / 2; j++)
        {
            float x = anchor_points[j * 2] + shifts[i * 2];
            float y = anchor_points[j * 2 + 1] + shifts[i * 2 + 1];
            shifted_anchor_points[i * anchor_points.size() / 2 * 2 + j * 2] = x;
            shifted_anchor_points[i * anchor_points.size() / 2 * 2 + j * 2 + 1] = y;
        }
    }
}
static void generate_anchor_points(int stride, int row, int line, std::vector<float>& anchor_points)
{
    float row_step = (float)stride / row;
    float line_step = (float)stride / line;

    std::vector<float> x_, y_;
    for (int i = 1; i < line + 1; i++)
    {
        float x = (i - 0.5) * line_step - stride / 2;
        x_.push_back(x);
    }
    for (int i = 1; i < row + 1; i++)
    {
        float y = (i - 0.5) * row_step - stride / 2;
        y_.push_back(y);
    }
    std::vector<float> shift_x((size_t)row * line, 0), shift_y((size_t)row * line, 0);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < line; j++)
        {
            shift_x[i * line + j] = x_[j];
        }
    }
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < line; j++)
        {
            shift_y[i * line + j] = y_[i];
        }
    }
    anchor_points.resize((size_t)row * line * 2, 0);
    for (int i = 0; i < row * line; i++)
    {
        float x = shift_x[i];
        float y = shift_y[i];
        anchor_points[i * 2] = x;
        anchor_points[i * 2 + 1] = y;
    }
}
static void generate_anchor_points(int img_w, int img_h, std::vector<int> pyramid_levels, int row, int line, std::vector<float>& all_anchor_points)
{
    std::vector<std::pair<int, int> > image_shapes;
    std::vector<int> strides;
    for (int i = 0; i < pyramid_levels.size(); i++)
    {
        int new_h = std::floor((img_h + std::pow(2, pyramid_levels[i]) - 1) / std::pow(2, pyramid_levels[i]));
        int new_w = std::floor((img_w + std::pow(2, pyramid_levels[i]) - 1) / std::pow(2, pyramid_levels[i]));
        image_shapes.push_back(std::make_pair(new_w, new_h));
        strides.push_back(std::pow(2, pyramid_levels[i]));
    }

    all_anchor_points.clear();
    for (int i = 0; i < pyramid_levels.size(); i++)
    {
        std::vector<float> anchor_points;
        generate_anchor_points(std::pow(2, pyramid_levels[i]), row, line, anchor_points);
        std::vector<float> shifted_anchor_points;
        shift(image_shapes[i].first, image_shapes[i].second, strides[i], anchor_points, shifted_anchor_points);
        all_anchor_points.insert(all_anchor_points.end(), shifted_anchor_points.begin(), shifted_anchor_points.end());
    }
}

static int detect_crowd(const cv::Mat& bgr, std::vector<CrowdPoint>& crowd_points)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_vulkan_compute = false;
    opt.use_bf16_storage = false;

    ncnn::Net net;
    net.opt = opt;

    // model is converted from
    // https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
    // the ncnn model  https://pan.baidu.com/s/1O1CBgvY6yJkrK8Npxx3VMg pwd: ezhx
    if (net.load_param("p2pnet.param"))
        exit(-1);
    if (net.load_model("p2pnet.bin"))
        exit(-1);

    int width = bgr.cols;
    int height = bgr.rows;

    int new_width = width / 128 * 128;
    int new_height = height / 128 * 128;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, new_width, new_height);

    std::vector<int> pyramid_levels(1, 3);
    std::vector<float> all_anchor_points;
    generate_anchor_points(in.w, in.h, pyramid_levels, 2, 2, all_anchor_points);

    ncnn::Mat anchor_points = ncnn::Mat(2, all_anchor_points.size() / 2, all_anchor_points.data());

    ncnn::Extractor ex = net.create_extractor();
    const float mean_vals1[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals1[3] = {0.01712475f, 0.0175f, 0.01742919f};

    in.substract_mean_normalize(mean_vals1, norm_vals1);

    ex.input("input", in);
    ex.input("anchor", anchor_points);

    ncnn::Mat score, points;
    ex.extract("pred_scores", score);
    ex.extract("pred_points", points);

    for (int i = 0; i < points.h; i++)
    {
        float* score_data = score.row(i);
        float* points_data = points.row(i);
        CrowdPoint cp;
        int x = points_data[0] / new_width * width;
        int y = points_data[1] / new_height * height;
        cp.pt = cv::Point(x, y);
        cp.prob = score_data[1];
        crowd_points.push_back(cp);
    }

    return 0;
}

static void draw_result(const cv::Mat& bgr, const std::vector<CrowdPoint>& crowd_points)
{
    cv::Mat image = bgr.clone();
    const float threshold = 0.5f;
    for (int i = 0; i < crowd_points.size(); i++)
    {
        if (crowd_points[i].prob > threshold)
        {
            cv::circle(image, crowd_points[i].pt, 4, cv::Scalar(0, 0, 255), -1, 8, 0);
        }
    }
    cv::imshow("image", image);
    cv::waitKey();
}
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat bgr = cv::imread(imagepath, 1);
    if (bgr.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<CrowdPoint> crowd_points;
    detect_crowd(bgr, crowd_points);
    draw_result(bgr, crowd_points);

    return 0;
}
