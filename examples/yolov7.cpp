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

#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>

#define MAX_STRIDE 32

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);
                float box_confidence = sigmoid(featptr[4]);
                if (box_confidence >= prob_threshold)
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = box_confidence * sigmoid(class_score);
                    if (confidence >= prob_threshold)
                    {
                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

static int detect_yolov7(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov7;

    yolov7.opt.use_vulkan_compute = true;
    // yolov7.opt.use_bf16_storage = true;

    // original pretrained model from https://github.com/WongKinYiu/yolov7
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    yolov7.load_param("yolov7-tiny.param");
    yolov7.load_model("yolov7-tiny.bin");

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov7.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output", out);

        ncnn::Mat anchors(6);
        anchors[0] = 12.f;
        anchors[1] = 16.f;
        anchors[2] = 19.f;
        anchors[3] = 36.f;
        anchors[4] = 40.f;
        anchors[5] = 28.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat out;

        ex.extract("288", out);

        ncnn::Mat anchors(6);
        anchors[0] = 36.f;
        anchors[1] = 75.f;
        anchors[2] = 76.f;
        anchors[3] = 55.f;
        anchors[4] = 72.f;
        anchors[5] = 146.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat out;

        ex.extract("302", out);

        ncnn::Mat anchors(6);
        anchors[0] = 142.f;
        anchors[1] = 110.f;
        anchors[2] = 192.f;
        anchors[3] = 243.f;
        anchors[4] = 459.f;
        anchors[5] = 401.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    static const unsigned char colors[19][3] = {
        {54, 67, 244},
        {99, 30, 233},
        {176, 39, 156},
        {183, 58, 103},
        {181, 81, 63},
        {243, 150, 33},
        {244, 169, 3},
        {212, 188, 0},
        {136, 150, 0},
        {80, 175, 76},
        {74, 195, 139},
        {57, 220, 205},
        {59, 235, 255},
        {7, 193, 255},
        {0, 152, 255},
        {34, 87, 255},
        {72, 85, 121},
        {158, 158, 158},
        {139, 125, 96}
    };

    int color_index = 0;

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cc, 2);

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
                      cc, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
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
    detect_yolov7(m, objects);

    draw_objects(m, objects);

    return 0;
}
