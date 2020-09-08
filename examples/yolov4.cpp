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

#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if CV_MAJOR_VERSION >= 3
#include <opencv2/videoio/videoio.hpp>
#endif

#include <vector>

#include <stdio.h>

#define NCNN_PROFILING
#define YOLOV4_TINY //Using yolov4_tiny, if undef, using original yolov4

#ifdef NCNN_PROFILING
#include "benchmark.h"
#endif

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int init_yolov4(ncnn::Net* yolov4, int* target_size)
{
    /* --> Set the params you need for the ncnn inference <-- */

    yolov4->opt.num_threads = 4; //You need to compile with libgomp for multi thread support

    yolov4->opt.use_vulkan_compute = true; //You need to compile with libvulkan for gpu support

    yolov4->opt.use_winograd_convolution = true;
    yolov4->opt.use_sgemm_convolution = true;
    yolov4->opt.use_fp16_packed = true;
    yolov4->opt.use_fp16_storage = true;
    yolov4->opt.use_fp16_arithmetic = true;
    yolov4->opt.use_packing_layout = true;
    yolov4->opt.use_shader_pack8 = false;
    yolov4->opt.use_image_storage = false;

    /* --> End of setting params <-- */
    int ret = 0;

    // original pretrained model from https://github.com/AlexeyAB/darknet
    // the ncnn model https://drive.google.com/drive/folders/1YzILvh0SKQPS_lrb33dmGNq7aVTKPWS0?usp=sharing
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
#ifdef YOLOV4_TINY
    const char* yolov4_param = "yolov4-tiny-opt.param";
    const char* yolov4_model = "yolov4-tiny-opt.bin";
    *target_size = 416;
#else
    const char* yolov4_param = "yolov4-opt.param";
    const char* yolov4_model = "yolov4-opt.bin";
    *target_size = 608;
#endif

    ret = yolov4->load_param(yolov4_param);
    if (ret != 0)
    {
        return ret;
    }

    ret = yolov4->load_model(yolov4_model);
    if (ret != 0)
    {
        return ret;
    }

    return 0;
}

static int detect_yolov4(const cv::Mat& bgr, std::vector<Object>& objects, int target_size, ncnn::Net* yolov4)
{
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov4->create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);

    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static int draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, int is_streaming)
{
    static const char* class_names[] = {"background", "person", "bicycle",
                                        "car", "motorbike", "aeroplane", "bus", "train", "truck",
                                        "boat", "traffic light", "fire hydrant", "stop sign",
                                        "parking meter", "bench", "bird", "cat", "dog", "horse",
                                        "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase",
                                        "frisbee", "skis", "snowboard", "sports ball", "kite",
                                        "baseball bat", "baseball glove", "skateboard", "surfboard",
                                        "tennis racket", "bottle", "wine glass", "cup", "fork",
                                        "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                                        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                                        "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
                                        "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                                        "cell phone", "microwave", "oven", "toaster", "sink",
                                        "refrigerator", "book", "clock", "vase", "scissors",
                                        "teddy bear", "hair drier", "toothbrush"
                                       };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);

    if (is_streaming)
    {
        cv::waitKey(1);
    }
    else
    {
        cv::waitKey(0);
    }

    return 0;
}

int main(int argc, char** argv)
{
    cv::Mat frame;
    std::vector<Object> objects;

    cv::VideoCapture cap;

    ncnn::Net yolov4;

    const char* devicepath;

    int target_size = 0;
    int is_streaming = 0;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [v4l inpude device or image]\n", argv[0]);
        return -1;
    }

    devicepath = argv[1];

#ifdef NCNN_PROFILING
    double t_load_start = ncnn::get_current_time();
#endif

    int ret = init_yolov4(&yolov4, &target_size); //We load model and param first!
    if (ret != 0)
    {
        fprintf(stderr, "Failed to load model or param, error %d", ret);
        return -1;
    }

#ifdef NCNN_PROFILING
    double t_load_end = ncnn::get_current_time();
    fprintf(stdout, "NCNN Init time %.02lfms\n", t_load_end - t_load_start);
#endif

    if (strstr(devicepath, "/dev/video") == NULL)
    {
        frame = cv::imread(argv[1], 1);
        if (frame.empty())
        {
            fprintf(stderr, "Failed to read image %s.\n", argv[1]);
            return -1;
        }
    }
    else
    {
        cap.open(devicepath);

        if (!cap.isOpened())
        {
            fprintf(stderr, "Failed to open %s", devicepath);
            return -1;
        }

        cap >> frame;

        if (frame.empty())
        {
            fprintf(stderr, "Failed to read from device %s.\n", devicepath);
            return -1;
        }

        is_streaming = 1;
    }

    while (1)
    {
        if (is_streaming)
        {
#ifdef NCNN_PROFILING
            double t_capture_start = ncnn::get_current_time();
#endif

            cap >> frame;

#ifdef NCNN_PROFILING
            double t_capture_end = ncnn::get_current_time();
            fprintf(stdout, "NCNN OpenCV capture time %.02lfms\n", t_capture_end - t_capture_start);
#endif
            if (frame.empty())
            {
                fprintf(stderr, "OpenCV Failed to Capture from device %s\n", devicepath);
                return -1;
            }
        }

#ifdef NCNN_PROFILING
        double t_detect_start = ncnn::get_current_time();
#endif

        detect_yolov4(frame, objects, target_size, &yolov4); //Create an extractor and run detection

#ifdef NCNN_PROFILING
        double t_detect_end = ncnn::get_current_time();
        fprintf(stdout, "NCNN detection time %.02lfms\n", t_detect_end - t_detect_start);
#endif

#ifdef NCNN_PROFILING
        double t_draw_start = ncnn::get_current_time();
#endif

        draw_objects(frame, objects, is_streaming); //Draw detection results on opencv image

#ifdef NCNN_PROFILING
        double t_draw_end = ncnn::get_current_time();
        fprintf(stdout, "NCNN OpenCV draw result time %.02lfms\n", t_draw_end - t_draw_start);
#endif

        if (!is_streaming)
        {   //If it is a still image, exit!
            return 0;
        }
    }

    return 0;
}
