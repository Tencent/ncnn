// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_YOLOV3DETECTIONOUTPUT_H
#define LAYER_YOLOV3DETECTIONOUTPUT_H

#include "layer.h"

namespace ncnn {

class Yolov3DetectionOutput : public Layer
{
public:
    Yolov3DetectionOutput();
    ~Yolov3DetectionOutput();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int num_class;
    int num_box;
    float confidence_threshold;
    float nms_threshold;
    Mat biases;
    Mat mask;
    Mat anchors_scale;
    int mask_group_num;
    ncnn::Layer* softmax;

public:
    struct BBoxRect
    {
        float score;
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float area;
        int label;
    };

    void qsort_descent_inplace(std::vector<BBoxRect>& datas, int left, int right) const;
    void qsort_descent_inplace(std::vector<BBoxRect>& datas) const;
    void nms_sorted_bboxes(std::vector<BBoxRect>& bboxes, std::vector<size_t>& picked, float nms_threshold) const;
};

} // namespace ncnn

#endif // LAYER_YOLODETECTIONOUTPUT_H
