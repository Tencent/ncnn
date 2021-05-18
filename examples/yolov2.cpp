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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

ncnn::Net yolov2;

static int init_yolov2()
{

    /* --> Set the params you need for the ncnn inference <-- */

    yolov2.opt.num_threads = 4; //You need to compile with libgomp for multi thread support

    yolov2.opt.use_vulkan_compute = true; //You need to compile with libvulkan for gpu support

    yolov2.opt.use_winograd_convolution = true;
    yolov2.opt.use_sgemm_convolution = true;
    yolov2.opt.use_fp16_packed = true;
    yolov2.opt.use_fp16_storage = true;
    yolov2.opt.use_fp16_arithmetic = true;
    yolov2.opt.use_packing_layout = true;
    yolov2.opt.use_shader_pack8 = false;
    yolov2.opt.use_image_storage = false;

    /* --> End of setting params <-- */
    int ret = 0;

    // original pretrained model from https://github.com/eric612/MobileNet-YOLO
    // https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov2/mobilenet_yolo_deploy.prototxt
    // https://github.com/eric612/MobileNet-YOLO/blob/master/models/yolov2/mobilenet_yolo_deploy_iter_80000.caffemodel
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    const char* yolov2_param = "mobilenet_yolo.param";
    const char* yolov2_model = "mobilenet_yolo.bin";


    ret = yolov2.load_param(yolov2_param);
    if (ret != 0)
    {
        return ret;
    }

    ret = yolov2.load_model(yolov2_model);
    if (ret != 0)
    {
        return ret;
    }

    return 0;
}

static int detect_yolov2(const cv::Mat& bgr, std::vector<Object>& objects)
{
    const int target_size = 416;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    // the Caffe-YOLOv2-Windows style
    // X' = X * scale - mean
    const float mean_vals[3] = {1.0f, 1.0f, 1.0f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    in.substract_mean_normalize(0, norm_vals);
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = yolov2.create_extractor();

    ex.input("data", in);

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
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

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
                                        "sheep", "sofa", "train", "tvmonitor"
                                       };

    for (size_t i = 0; i < objects.size(); i++)
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

    cv::imshow("image", bgr);
    cv::waitKey(0);
}

static int draw_fps(cv::Mat& bgr)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = { 0.f };

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

    int ret = init_yolov2(); //We load model and param first!
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
        detect_yolov2(m, objects);

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
            detect_yolov2(frame, objects);

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
            detect_yolov2(frame, objects);

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
