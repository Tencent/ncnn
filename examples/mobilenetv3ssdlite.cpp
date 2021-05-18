// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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
#include "platform.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

template<class T>
const T& clamp(const T& v, const T& lo, const T& hi)
{
    assert(!(hi < lo));
    return v < lo ? lo : hi < v ? hi : v;
}

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

ncnn::Net mobilenetv3;

static int init_mobilenetv3()
{
    /* --> Set the params you need for the ncnn inference <-- */

    mobilenetv3.opt.num_threads = 4; //You need to compile with libgomp for multi thread support

    mobilenetv3.opt.use_vulkan_compute = true; //You need to compile with libvulkan for gpu support

    mobilenetv3.opt.use_winograd_convolution = true;
    mobilenetv3.opt.use_sgemm_convolution = true;
    mobilenetv3.opt.use_fp16_packed = true;
    mobilenetv3.opt.use_fp16_storage = true;
    mobilenetv3.opt.use_fp16_arithmetic = true;
    mobilenetv3.opt.use_packing_layout = true;
    mobilenetv3.opt.use_shader_pack8 = false;
    mobilenetv3.opt.use_image_storage = false;

    /* --> End of setting params <-- */
    int ret = 0;

    // converted ncnn model from https://github.com/ujsyehao/mobilenetv3-ssd
    const char* mobilenetv3_param = "mobilenetv3_ssdlite_voc.param";
    const char* mobilenetv3_model = "mobilenetv3_ssdlite_voc.bin";

    ret = mobilenetv3.load_param(mobilenetv3_param);
    if (ret != 0)
    {
        return ret;
    }

    ret = mobilenetv3.load_model(mobilenetv3_model);
    if (ret != 0)
    {
        return ret;
    }

    return 0;
}

static int detect_mobilenetv3(const cv::Mat& bgr, std::vector<Object>& objects)
{
    const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {1.0f, 1.0f, 1.0f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenetv3.create_extractor();

    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);

    //     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];

        // filter out cross-boundary
        float x1 = clamp(values[2] * target_size, 0.f, float(target_size - 1)) / target_size * img_w;
        float y1 = clamp(values[3] * target_size, 0.f, float(target_size - 1)) / target_size * img_h;
        float x2 = clamp(values[4] * target_size, 0.f, float(target_size - 1)) / target_size * img_w;
        float y2 = clamp(values[5] * target_size, 0.f, float(target_size - 1)) / target_size * img_h;

        object.rect.x = x1;
        object.rect.y = y1;
        object.rect.width = x2 - x1;
        object.rect.height = y2 - y1;

        objects.push_back(object);
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"};

    for (size_t i = 0; i < objects.size(); i++)
    {
        if (objects[i].prob > 0.6)
        {
            const Object& obj = objects[i];

            fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

            cv::rectangle(bgr, obj.rect, cv::Scalar(255, 0, 0));

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > bgr.cols)
                x = bgr.cols - label_size.width;

            cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(255, 255, 255), -1);

            cv::putText(bgr, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
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
    int ret = init_mobilenetv3(); //We load model and param first!
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
        std::vector<Object> objects;
        detect_mobilenetv3(m, objects);

        draw_objects(m, objects);
        cv::imshow("image", m);
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
            std::vector<Object> objects;
            detect_mobilenetv3(frame, objects);

            draw_objects(frame, objects);
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
            std::vector<Object> objects;
            detect_mobilenetv3(frame, objects);

            draw_objects(frame, objects);
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
