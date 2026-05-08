// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yoloworld torchscript
//      yolo export model=yolov8s-world.pt format=torchscript
//      yolo export model=yolov8m-world.pt format=torchscript
//      yolo export model=yolov8l-world.pt format=torchscript
//      yolo export model=yolov8x-world.pt format=torchscript
//      yolo export model=yolov8s-worldv2.pt format=torchscript
//      yolo export model=yolov8m-worldv2.pt format=torchscript
//      yolo export model=yolov8l-worldv2.pt format=torchscript
//      yolo export model=yolov8x-worldv2.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolov8s-world.torchscript
//      pnnx yolov8m-world.torchscript
//      pnnx yolov8l-world.torchscript
//      pnnx yolov8x-world.torchscript
//      pnnx yolov8s-worldv2.torchscript
//      pnnx yolov8m-worldv2.torchscript
//      pnnx yolov8l-worldv2.torchscript
//      pnnx yolov8x-worldv2.torchscript

// the out blob would be a 2-dim tensor with w=8400 h=84
//
//        |    all boxes (8400)     |
//        +-------------------------+
//        | center-x   .            |
//  bbox  | center-y   .            |
//        |   w        .            |
//        |   h        .            |
//        +-------------------------+
//        | 0.1        .            |
//   per  | 0.0        .            |
//  class | 0.5        .            |
// scores |  .         .            |
//  (80)  |  .         .            |
//        +-------------------------+

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

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        // #pragma omp section
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

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

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

static void generate_proposals(const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_boxes = pred.w;
    const int num_class = pred.h - 4;

    const ncnn::Mat pred_bbox = pred.row_range(0, 4);
    const ncnn::Mat pred_score = pred.row_range(4, num_class);

    for (int i = 0; i < num_boxes; i++)
    {
        int label = 0;
        float score = -9999.f;
        for (int j = 0; j < num_class; j++)
        {
            const float prob = pred_score.row(j)[i];
            if (prob > score)
            {
                score = prob;
                label = j;
            }
        }

        if (score >= prob_threshold)
        {
            const float cx = pred_bbox.row(0)[i];
            const float cy = pred_bbox.row(1)[i];
            const float w = pred_bbox.row(2)[i];
            const float h = pred_bbox.row(3)[i];

            Object obj;
            obj.rect.x = cx - w / 2;
            obj.rect.y = cy - h / 2;
            obj.rect.width = w;
            obj.rect.height = h;
            obj.label = label;
            obj.prob = score;

            objects.push_back(obj);
        }
    }
}

static int detect_yoloworld(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yoloworld;

    yoloworld.opt.use_vulkan_compute = true;
    // yoloworld.opt.use_bf16_storage = true;

    // https://github.com/nihui/ncnn-assets/tree/master/models
    // yoloworld.load_param("yolov8s_world.ncnn.param");
    // yoloworld.load_model("yolov8s_world.ncnn.bin");
    // yoloworld.load_param("yolov8m_world.ncnn.param");
    // yoloworld.load_model("yolov8m_world.ncnn.bin");
    // yoloworld.load_param("yolov8l_world.ncnn.param");
    // yoloworld.load_model("yolov8l_world.ncnn.bin");
    // yoloworld.load_param("yolov8x_world.ncnn.param");
    // yoloworld.load_model("yolov8x_world.ncnn.bin");
    yoloworld.load_param("yolov8s_worldv2.ncnn.param");
    yoloworld.load_model("yolov8s_worldv2.ncnn.bin");
    // yoloworld.load_param("yolov8m_worldv2.ncnn.param");
    // yoloworld.load_model("yolov8m_worldv2.ncnn.bin");
    // yoloworld.load_param("yolov8l_worldv2.ncnn.param");
    // yoloworld.load_model("yolov8l_worldv2.ncnn.bin");
    // yoloworld.load_param("yolov8x_worldv2.ncnn.param");
    // yoloworld.load_model("yolov8x_worldv2.ncnn.bin");

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad
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

    // letterbox pad to target_size rectangle
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yoloworld.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<Object> proposals;
    generate_proposals(out, prob_threshold, proposals);

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

    static cv::Scalar colors[] = {
        cv::Scalar(244, 67, 54),
        cv::Scalar(233, 30, 99),
        cv::Scalar(156, 39, 176),
        cv::Scalar(103, 58, 183),
        cv::Scalar(63, 81, 181),
        cv::Scalar(33, 150, 243),
        cv::Scalar(3, 169, 244),
        cv::Scalar(0, 188, 212),
        cv::Scalar(0, 150, 136),
        cv::Scalar(76, 175, 80),
        cv::Scalar(139, 195, 74),
        cv::Scalar(205, 220, 57),
        cv::Scalar(255, 235, 59),
        cv::Scalar(255, 193, 7),
        cv::Scalar(255, 152, 0),
        cv::Scalar(255, 87, 34),
        cv::Scalar(121, 85, 72),
        cv::Scalar(158, 158, 158),
        cv::Scalar(96, 125, 139)
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[i % 19];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, color);

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
    detect_yoloworld(m, objects);

    draw_objects(m, objects);

    return 0;
}
