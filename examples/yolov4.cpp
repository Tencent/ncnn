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
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include <string.h>
#include <stdio.h>

#include <sys/ioctl.h>
#include <linux/fb.h>
#include <fcntl.h>

#define NCNN_PROFILING
#define ENABLE_LINUX_FB_SUPPORT

int framebuffer_width = 240;
int framebuffer_depth = 16;
cv::Size2f frame_size;

#ifdef ENABLE_LINUX_FB_SUPPORT
struct framebuffer_info { 
    uint32_t bits_per_pixel; uint32_t xres_virtual; 
};

struct framebuffer_info get_framebuffer_info(const char* framebuffer_device_path) {
    struct framebuffer_info info;
    struct fb_var_screeninfo screen_info;
    int fd = -1;
    fd = open(framebuffer_device_path, O_RDWR);
    if (fd >= 0) {
        if (!ioctl(fd, FBIOGET_VSCREENINFO, &screen_info)) {
            info.xres_virtual = screen_info.xres_virtual;
            info.bits_per_pixel = screen_info.bits_per_pixel;
        }
    }
    return info;
};
#endif

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int init_yolov4(char *filename, ncnn::Net *yolov4){

/* --> Set the params you need for the ncnn inference <-- */

    yolov4->opt.num_threads = 4; //You need to compile with libgomp for multi thread support
    
    yolov4->opt.use_winograd_convolution = true;
    yolov4->opt.use_sgemm_convolution = true;
    yolov4->opt.use_fp16_packed = true;
    yolov4->opt.use_fp16_storage = true;
    yolov4->opt.use_fp16_arithmetic = true;
    yolov4->opt.use_packing_layout = true;
    yolov4->opt.use_shader_pack8 = false;
    yolov4->opt.use_image_storage = false;

/* --> End of setting params <-- */

    char paramname[100];
    sprintf(paramname, "%s.param", filename);

    char modelname[100];
    sprintf(modelname, "%s.bin", filename);

    int ret = yolov4->load_param(paramname);  
    if(ret != 0){
        return ret;
    }

    ret = yolov4->load_model(modelname);
    if(ret != 0){
        return ret;
    }

    return 0;
}

static int detect_yolov4(const cv::Mat& bgr, std::vector<Object>& objects, int target_size, ncnn::Net *yolov4)
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

static int draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::ofstream& fbdevice, int is_using_gui, int is_still_picture)
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

    if(is_using_gui){
        cv::imshow("image", image);
        if(is_still_picture){
            cv::waitKey(0);
        }else{
            cv::waitKey(1);
        }
    }else{
#ifdef ENABLE_LINUX_FB_SUPPORT
        frame_size = image.size();

        cv::cvtColor(image, image, cv::COLOR_BGR2BGR565);
        
        for (int y = 0; y < frame_size.height ; y++){
            fbdevice.seekp(y*framebuffer_width*2);
            fbdevice.write(reinterpret_cast<char*>(image.ptr(y)),frame_size.width*2);
        }
#else
        fprintf(stderr, "Linux Framebuffer support not enabled.\n");
        return -1;
#endif
    }

    return 0;
    
}

int main(int argc, char** argv){

    cv::Mat frame;
    std::vector<Object> objects;

    cv::VideoCapture cap;
    
    ncnn::Net yolov4;

    std::ofstream ofs; //For writing to linux framebuffer

    const char* devicepath;

    int detection_resolution = 0;
    int is_using_gui = 1;
    int is_streaming = 0;

    if (argc < 5){
        fprintf(stderr, "Usage: %s [v4l inpude device or image file] {\"gui\", [output devices]} [ncnn filename] [detection input size]\n", argv[0]);
        fprintf(stderr, "For output to gui, use \"gui\" as output devices, and for linux framebuffer, use /dev/fb* as output devices.");
        return -1;
    }

    detection_resolution = std::stoi(argv[4]);
    if(detection_resolution <= 0){
        fprintf(stderr, "Invalid input resolution %s.\n", argv[4]);
        return -1;
    }

    if(strstr(argv[1], "/dev/video") == NULL){
        frame = cv::imread(argv[1], 1);
        if(frame.empty()){
            fprintf(stderr, "Failed to read image %s.\n", argv[1]);
            return -1;
        }
    }else{
        devicepath = argv[1];

        cap.open(devicepath);

        if (!cap.isOpened()){
            fprintf(stderr, "Failed to open %s", devicepath);
            return -1;
        }

        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        if(detection_resolution < 240){
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
        }
        
        cap >> frame;

        if(frame.empty()){
            fprintf(stderr, "Failed to read from device %s.\n", devicepath);
            return -1;
        }

        is_streaming = 1;
    }

#ifdef NCNN_PROFILING
    auto t_load_start = std::chrono::high_resolution_clock::now();
#endif

    int ret = init_yolov4(argv[3], &yolov4);
    if(ret != 0){
        fprintf(stderr, "Failed to load model or param %s, error %d", argv[3], ret);
        return -1;
    }

#ifdef NCNN_PROFILING
    auto t_load_end = std::chrono::high_resolution_clock::now();
    auto t_load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_load_start - t_load_end).count();
    fprintf(stdout, "NCNN Init time %dms\n", t_load_duration);
#endif

    if(strstr(argv[2], "gui") == NULL){
#ifdef ENABLE_LINUX_FB_SUPPORT
        if(strstr(argv[2], "/dev/fb") == NULL){
            fprintf(stderr, "The device %s you are pointing to may not be a valid linux fb sink.", argv[2]);
        }

        framebuffer_info fb_info = get_framebuffer_info(argv[2]);
        framebuffer_width = fb_info.xres_virtual;
        framebuffer_depth = fb_info.bits_per_pixel;

        ofs.open(argv[2]);
        if(ofs.bad()){
            fprintf(stderr, "Failed to open device %s.", argv[2]);
            return -1;
        }

        is_using_gui = 0;
#else
        fprintf(stderr, "Linux Framebuffer support not enabled.\n");
        return -1;
#endif
    }

    while(1){

        if(is_streaming){
#ifdef NCNN_PROFILING
            auto t_capture_start = std::chrono::high_resolution_clock::now();
#endif
            cap >> frame;
#ifdef NCNN_PROFILING
            auto t_capture_end = std::chrono::high_resolution_clock::now();
            auto t_capture_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_capture_start - t_capture_end).count();
            fprintf(stdout, "NCNN OpenCV capture time %dms\n", t_capture_duration);
#endif
            if (frame.empty()){
                fprintf(stderr, "OpenCV Failed to Capture from device %s\n", devicepath);
                return -1;
            }
        }

#ifdef NCNN_PROFILING
        auto t_detect_start = std::chrono::high_resolution_clock::now();
#endif
        detect_yolov4(frame, objects, detection_resolution, &yolov4);
#ifdef NCNN_PROFILING
        auto t_detect_end = std::chrono::high_resolution_clock::now();
        auto t_detect_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_detect_start - t_detect_end).count();
        fprintf(stdout, "NCNN detection time %dms\n", t_detect_duration);
#endif


#ifdef NCNN_PROFILING
        auto t_draw_start = std::chrono::high_resolution_clock::now();
#endif
        draw_objects(frame, objects, ofs, is_using_gui, !is_streaming);
#ifdef NCNN_PROFILING
        auto t_draw_end = std::chrono::high_resolution_clock::now();
        auto t_draw_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_draw_start - t_draw_end).count();
        fprintf(stdout, "NCNN OpenCV draw result time %dms\n", t_draw_duration);
#endif

        if(!is_streaming){
            return 0;
        }
    }

    return 0;
}
