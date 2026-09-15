// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_yolov3detectionoutput(const std::vector<ncnn::Mat>& a, int num_class,
                                      int num_box, float confidence_threshold, float nms_threshold,
                                      ncnn::Mat& biases, ncnn::Mat& mask, ncnn::Mat& anchors_scale)
{
    ncnn::ParamDict pd;
    pd.set(0, num_class);
    pd.set(1, num_box);
    pd.set(2, confidence_threshold);
    pd.set(3, nms_threshold);
    pd.set(4, biases);
    pd.set(5, mask);
    pd.set(6, anchors_scale);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Yolov3DetectionOutput", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_yolov3detectionoutput failed a.dims=%d a=(%d %d %d) ", a[0].dims, a[0].w, a[0].h, a[0].c);
        fprintf(stderr, " num_class=%d num_box=%d", num_class, num_box);
        fprintf(stderr, " confidence_threshold=%f nms_threshold=%f\n", confidence_threshold, nms_threshold);
    }

    return ret;
}

static ncnn::Mat create_mat_from(const float* src, int length)
{
    ncnn::Mat ret(length);
    memcpy(ret.data, src, length * sizeof(float));
    return ret;
}

static ncnn::Mat MyRandomMat(int w, int h, int c)
{
    ncnn::Mat m(w, h, c);
    Randomize(m, -15.f, 1.5f);
    return m;
}

static int test_yolov3detectionoutput_v4()
{
    const float b[] = {12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401};
    const float m[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    const float s[] = {9.6, 17.6, 33.6};

    ncnn::Mat biases = create_mat_from(b, sizeof(b) / sizeof(b[0]));
    ncnn::Mat mask = create_mat_from(m, sizeof(m) / sizeof(m[0]));
    ncnn::Mat anchors_scale = create_mat_from(s, sizeof(s) / sizeof(s[0]));

    std::vector<ncnn::Mat> a(3);
    a[0] = MyRandomMat(76, 76, 255);
    a[1] = MyRandomMat(38, 38, 255);
    a[2] = MyRandomMat(19, 19, 255);

    return 0
           || test_yolov3detectionoutput(a, 80, 3, 0.55f, 0.45f, biases, mask, anchors_scale);
}

static int test_yolov3detectionoutput_v4tiny()
{
    const float b[] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
    const float m[] = {3, 4, 5, 1, 2, 3};
    const float s[] = {33.6, 16.8};

    ncnn::Mat biases = create_mat_from(b, sizeof(b) / sizeof(b[0]));
    ncnn::Mat mask = create_mat_from(m, sizeof(m) / sizeof(m[0]));
    ncnn::Mat anchors_scale = create_mat_from(s, sizeof(s) / sizeof(s[0]));

    std::vector<ncnn::Mat> a(2);
    a[0] = MyRandomMat(13, 13, 255);
    a[1] = MyRandomMat(26, 26, 255);

    return 0
           || test_yolov3detectionoutput(a, 80, 3, 0.4f, 0.45f, biases, mask, anchors_scale);
}

static int test_yolov3detectionoutput_v3()
{
    const float b[] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
    const float m[] = {6, 7, 8, 3, 4, 5, 0, 1, 2};
    const float s[] = {32, 16, 8};

    ncnn::Mat biases = create_mat_from(b, sizeof(b) / sizeof(b[0]));
    ncnn::Mat mask = create_mat_from(m, sizeof(m) / sizeof(m[0]));
    ncnn::Mat anchors_scale = create_mat_from(s, sizeof(s) / sizeof(s[0]));

    std::vector<ncnn::Mat> a(3);
    a[0] = MyRandomMat(19, 19, 255);
    a[1] = MyRandomMat(38, 38, 255);
    a[2] = MyRandomMat(76, 76, 255);

    return 0
           || test_yolov3detectionoutput(a, 80, 3, 0.6f, 0.45f, biases, mask, anchors_scale);
}

static int test_yolov3detectionoutput_v3tiny()
{
    const float b[] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
    const float m[] = {3, 4, 5, 1, 2, 3};
    const float s[] = {32, 16};

    ncnn::Mat biases = create_mat_from(b, sizeof(b) / sizeof(b[0]));
    ncnn::Mat mask = create_mat_from(m, sizeof(m) / sizeof(m[0]));
    ncnn::Mat anchors_scale = create_mat_from(s, sizeof(s) / sizeof(s[0]));

    std::vector<ncnn::Mat> a(2);
    a[0] = MyRandomMat(13, 13, 255);
    a[1] = MyRandomMat(26, 26, 255);

    return 0
           || test_yolov3detectionoutput(a, 80, 3, 0.3f, 0.45f, biases, mask, anchors_scale);
}

static int test_yolov3detectionoutput_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::Yolov3DetectionOutput);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if ((ret == 0) != valid)
    {
        fprintf(stderr, "Yolov3DetectionOutput load_param returned %d, expected %s\n", ret, valid ? "success" : "failure");
        return -1;
    }

    return 0;
}

static int test_yolov3detectionoutput_load_param()
{
    ncnn::ParamDict pd;
    pd.set(0, 20);
    pd.set(1, 1);
    ncnn::Mat biases(2);
    biases.fill(1.f);
    pd.set(4, biases);
    ncnn::Mat mask(1);
    mask[0] = 0.f;
    pd.set(5, mask);
    ncnn::Mat scales(1);
    scales[0] = 32.f;
    pd.set(6, scales);
    if (test_yolov3detectionoutput_load_param_case(pd, true) != 0)
        return -1;

    const float invalid[] = {-1.f, 0.5f, 1.f, (float)INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        mask[0] = invalid[i];
        if (test_yolov3detectionoutput_load_param_case(pd, false) != 0)
            return -1;
    }
    mask[0] = 0.f;
    pd.set(6, ncnn::Mat());
    if (test_yolov3detectionoutput_load_param_case(pd, false) != 0)
        return -1;

    pd.set(4, biases.range(0, 1));
    if (test_yolov3detectionoutput_load_param_case(pd, false) != 0)
        return -1;
    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_yolov3detectionoutput_v3tiny()
           || test_yolov3detectionoutput_v3()
           || test_yolov3detectionoutput_v4tiny()
           || test_yolov3detectionoutput_v4()
           || test_yolov3detectionoutput_load_param();
}
