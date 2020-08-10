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

static int detect_peleenet(const cv::Mat& bgr, std::vector<Object>& objects, ncnn::Mat& resized)
{
    ncnn::Net peleenet;

    peleenet.opt.use_vulkan_compute = true;

    // model is converted from https://github.com/eric612/MobileNet-YOLO
    // and can be downloaded from https://drive.google.com/open?id=1Wt6jKv13sBRMHgrGAJYlOlRF-o80pC0g
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    peleenet.load_param("pelee.param");
    peleenet.load_model("pelee.bin");

    const int target_size = 304;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {103.9f, 116.7f, 123.6f};
    const float norm_vals[3] = {0.017f, 0.017f, 0.017f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = peleenet.create_extractor();

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
    ncnn::Mat seg_out;
    ex.extract("sigmoid", seg_out);
    resize_bilinear(seg_out, resized, img_w, img_h);
    //resize_bicubic(seg_out,resized,img_w,img_h); // sharpness
    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, ncnn::Mat map)
{
    static const char* class_names[] = {"background",
                                        "person", "rider", "car", "bus",
                                        "truck", "bike", "motor",
                                        "traffic light", "traffic sign", "train"
                                       };

    cv::Mat image = bgr.clone();
    const int color[] = {128, 255, 128, 244, 35, 232};
    const int color_count = sizeof(color) / sizeof(int);

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
    int width = map.w;
    int height = map.h;
    int size = map.c;
    int img_index2 = 0;
    float threshold = 0.45;
    const float* ptr2 = map;
    for (int i = 0; i < height; i++)
    {
        unsigned char* ptr1 = image.ptr<unsigned char>(i);
        int img_index1 = 0;
        for (int j = 0; j < width; j++)
        {
            float maxima = threshold;
            int index = -1;
            for (int c = 0; c < size; c++)
            {
                //const float* ptr3 = map.channel(c);
                const float* ptr3 = ptr2 + c * width * height;
                if (ptr3[img_index2] > maxima)
                {
                    maxima = ptr3[img_index2];
                    index = c;
                }
            }
            if (index > -1)
            {
                int color_index = (index)*3;
                if (color_index < color_count)
                {
                    int b = color[color_index];
                    int g = color[color_index + 1];
                    int r = color[color_index + 2];
                    ptr1[img_index1] = b / 2 + ptr1[img_index1] / 2;
                    ptr1[img_index1 + 1] = g / 2 + ptr1[img_index1 + 1] / 2;
                    ptr1[img_index1 + 2] = r / 2 + ptr1[img_index1 + 2] / 2;
                }
            }
            img_index1 += 3;
            img_index2++;
        }
    }
    cv::imshow("image", image);
    cv::waitKey(0);
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

    std::vector<Object> objects;
    ncnn::Mat seg_out;
    detect_peleenet(m, objects, seg_out);

    draw_objects(m, objects, seg_out);

    return 0;
}
