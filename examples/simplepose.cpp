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

#include "net.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>

struct KeyPoint
{
    cv::Point2f p;
    float prob;
};

static int detect_posenet(const cv::Mat& bgr, std::vector<KeyPoint>& keypoints)
{
    ncnn::Net posenet;

    posenet.opt.use_vulkan_compute = true;

    // the simple baseline human pose estimation from gluon-cv
    // https://gluon-cv.mxnet.io/build/examples_pose/demo_simple_pose.html
    // mxnet model exported via
    //      pose_net.hybridize()
    //      pose_net.export('pose')
    // then mxnet2ncnn
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    posenet.load_param("pose.param");
    posenet.load_model("pose.bin");

    int w = bgr.cols;
    int h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 192, 256);

    // transforms.ToTensor(),
    // transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    // R' = (R / 255 - 0.485) / 0.229 = (R - 0.485 * 255) / 0.229 / 255
    // G' = (G / 255 - 0.456) / 0.224 = (G - 0.456 * 255) / 0.224 / 255
    // B' = (B / 255 - 0.406) / 0.225 = (B - 0.406 * 255) / 0.225 / 255
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = posenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("conv3_fwd", out);

    // resolve point from heatmap
    keypoints.clear();
    for (int p = 0; p < out.c; p++)
    {
        const ncnn::Mat m = out.channel(p);

        float max_prob = 0.f;
        int max_x = 0;
        int max_y = 0;
        for (int y = 0; y < out.h; y++)
        {
            const float* ptr = m.row(y);
            for (int x = 0; x < out.w; x++)
            {
                float prob = ptr[x];
                if (prob > max_prob)
                {
                    max_prob = prob;
                    max_x = x;
                    max_y = y;
                }
            }
        }

        KeyPoint keypoint;
        keypoint.p = cv::Point2f(max_x * w / (float)out.w, max_y * h / (float)out.h);
        keypoint.prob = max_prob;

        keypoints.push_back(keypoint);
    }

    return 0;
}

static void draw_pose(const cv::Mat& bgr, const std::vector<KeyPoint>& keypoints)
{
    cv::Mat image = bgr.clone();

    // draw bone
    static const int joint_pairs[16][2] = {
        {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
    };

    for (int i = 0; i < 16; i++)
    {
        const KeyPoint& p1 = keypoints[joint_pairs[i][0]];
        const KeyPoint& p2 = keypoints[joint_pairs[i][1]];

        if (p1.prob < 0.2f || p2.prob < 0.2f)
            continue;

        cv::line(image, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
    }

    // draw joint
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        const KeyPoint& keypoint = keypoints[i];

        fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);

        if (keypoint.prob < 0.2f)
            continue;

        cv::circle(image, keypoint.p, 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<KeyPoint> keypoints;
    detect_posenet(m, keypoints);

    draw_pose(m, keypoints);

    return 0;
}
