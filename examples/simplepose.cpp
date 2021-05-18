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
#include "benchmark.h"

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

ncnn::Net posenet;

static int init_posenet()
{
    /* --> Set the params you need for the ncnn inference <-- */

    posenet.opt.num_threads = 4; //You need to compile with libgomp for multi thread support

    posenet.opt.use_vulkan_compute = true; //You need to compile with libvulkan for gpu support

    posenet.opt.use_winograd_convolution = true;
    posenet.opt.use_sgemm_convolution = true;
    posenet.opt.use_fp16_packed = true;
    posenet.opt.use_fp16_storage = true;
    posenet.opt.use_fp16_arithmetic = true;
    posenet.opt.use_packing_layout = true;
    posenet.opt.use_shader_pack8 = false;
    posenet.opt.use_image_storage = false;

    /* --> End of setting params <-- */
    int ret = 0;

    // the simple baseline human pose estimation from gluon-cv
    // https://gluon-cv.mxnet.io/build/examples_pose/demo_simple_pose.html
    // mxnet model exported via
    //      pose_net.hybridize()
    //      pose_net.export('pose')
    // then mxnet2ncnn
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    const char* posenet_param = "pose.param";
    const char* posenet_model = "pose.bin";

    ret = posenet.load_param(posenet_param);
    if (ret != 0)
    {
        return ret;
    }

    ret = posenet.load_model(posenet_model);
    if (ret != 0)
    {
        return ret;
    }

    return 0;
}

static int detect_posenet(const cv::Mat& bgr, std::vector<KeyPoint>& keypoints)
{
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

        cv::line(bgr, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
    }

    // draw joint
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        const KeyPoint& keypoint = keypoints[i];

        fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);

        if (keypoint.prob < 0.2f)
            continue;

        cv::circle(bgr, keypoint.p, 3, cv::Scalar(0, 255, 0), -1);
    }
}

static int draw_fps(cv::Mat& bgr)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = bgr.cols - label_size.width;

    cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(bgr, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage:(1) %s image [imagepath]\n", argv[0]);
        fprintf(stderr, "      (2) %s video [videopath]\n", argv[0]);
        fprintf(stderr, "      (3) %s capture [id]\n", argv[0]);
        return -1;
    }

    int ret = init_posenet(); //We load model and param first!
    if (ret != 0)
    {
        fprintf(stderr, "Failed to load model or param, error %d", ret);
        return -1;
    }

    const char* type = argv[1];
    if (0 == strcmp(type, "image"))
    {
        const char* imagepath = argv[2];

        cv::Mat m = cv::imread(imagepath, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }
        std::vector<KeyPoint> keypoints;
        detect_posenet(m, keypoints);

        draw_pose(m, keypoints);
        cv::waitKey(0);
    }
    else if (0 == strcmp(type, "video"))
    {
        const char* videopath = argv[2];
        cv::Mat frame;
        cv::VideoCapture cap(videopath);
        if (!cap.isOpened())
        {
            fprintf(stderr, "cv::VideoCapture %s failed\n", videopath);
            return -1;
        }
        while (true)
        {
            cap >> frame;
            std::vector<KeyPoint> keypoints;
            detect_posenet(frame, keypoints);

            draw_pose(frame, keypoints);
            draw_fps(frame);
            cv::imshow("video", frame);
            if (cv::waitKey(10) == 27)
            {
                break;
            }
        }
    }
    else if (0 == strcmp(type, "capture"))
    {
        int id = atoi(argv[2]);
        cv::Mat frame;
        cv::VideoCapture cap(id);
        if (!cap.isOpened())
        {
            fprintf(stderr, "cv::VideoCapture %d failed\n", id);
            return -1;
        }
        while (true)
        {
            cap >> frame;
            std::vector<KeyPoint> keypoints;
            detect_posenet(frame, keypoints);

            draw_pose(frame, keypoints);
            draw_fps(frame);
            cv::imshow("capture", frame);
            if (cv::waitKey(10) == 27)
            {
                break;
            }
        }
    }
    return 0;
}
