#include "net.h"
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <vector>

static void draw_objects(const cv::Mat& bgr, const cv::Mat& fgr, const cv::Mat& pha)
{
    cv::Mat fgr8U;
    fgr.convertTo(fgr8U, CV_8UC3, 255.0, 0);
    cv::Mat pha8U;
    pha.convertTo(pha8U, CV_8UC1, 255.0, 0);

    cv::Mat comp;
    cv::resize(bgr, comp, pha.size(), 0, 0, 1);
    for (int i = 0; i < pha8U.rows; i++)
    {
        for (int j = 0; j < pha8U.cols; j++)
        {
            uchar data = pha8U.at<uchar>(i, j);
            float alpha = (float)data / 255;
            comp.at<cv::Vec3b>(i, j)[0] = fgr8U.at<cv::Vec3b>(i, j)[0] * alpha + (1 - alpha) * 155;
            comp.at<cv::Vec3b>(i, j)[1] = fgr8U.at<cv::Vec3b>(i, j)[1] * alpha + (1 - alpha) * 255;
            comp.at<cv::Vec3b>(i, j)[2] = fgr8U.at<cv::Vec3b>(i, j)[2] * alpha + (1 - alpha) * 120;
        }
    }

    cv::imshow("pha", pha8U);
    cv::imshow("fgr", fgr8U);
    cv::imshow("comp", comp);
    cv::waitKey(0);
}
static int detect_rvm(const cv::Mat& bgr, cv::Mat& pha, cv::Mat& fgr)
{
    const float downsample_ratio = 0.5f;
    const int target_width = 512;
    const int target_height = 512;

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    //original pretrained model from https://github.com/PeterL1n/RobustVideoMatting
    //ncnn model https://pan.baidu.com/s/11iEY2RGfzWFtce8ue7T3JQ password: d9t6
    net.load_param("rvm_512.param");
    net.load_model("rvm_512.bin");

    //if you use another input size,pleaze change input shape
    ncnn::Mat r1i = ncnn::Mat(128, 128, 16);
    ncnn::Mat r2i = ncnn::Mat(64, 64, 20);
    ncnn::Mat r3i = ncnn::Mat(32, 32, 40);
    ncnn::Mat r4i = ncnn::Mat(16, 16, 64);
    r1i.fill(0.0f);
    r2i.fill(0.0f);
    r3i.fill(0.0f);
    r4i.fill(0.0f);

    ncnn::Extractor ex = net.create_extractor();
    const float mean_vals1[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals1[3] = {0.01712475f, 0.0175f, 0.01742919f};
    const float mean_vals2[3] = {0, 0, 0};
    const float norm_vals2[3] = {1 / 255.0, 1 / 255.0, 1 / 255.0};
    ncnn::Mat ncnn_in2 = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_width, target_height);
    ncnn::Mat ncnn_in1 = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_width * downsample_ratio, target_height * downsample_ratio);

    ncnn_in1.substract_mean_normalize(mean_vals1, norm_vals1);
    ncnn_in2.substract_mean_normalize(mean_vals2, norm_vals2);

    ex.input("src1", ncnn_in1);
    ex.input("src2", ncnn_in2);
    ex.input("r1i", r1i);
    ex.input("r2i", r2i);
    ex.input("r3i", r3i);
    ex.input("r4i", r4i);

    //if use video matting,these output will be input of next infer
    ex.extract("r4o", r4i);
    ex.extract("r3o", r3i);
    ex.extract("r2o", r2i);
    ex.extract("r1o", r1i);

    ncnn::Mat pha_;
    ex.extract("pha", pha_);
    ncnn::Mat fgr_;
    ex.extract("fgr", fgr_);

    cv::Mat cv_pha = cv::Mat(pha_.h, pha_.w, CV_32FC1, (float*)pha_.data);
    cv::Mat cv_fgr = cv::Mat(fgr_.h, fgr_.w, CV_32FC3);
    float* fgr_data = (float*)fgr_.data;
    for (int i = 0; i < fgr_.h; i++)
    {
        for (int j = 0; j < fgr_.w; j++)
        {
            cv_fgr.at<cv::Vec3f>(i, j)[2] = fgr_data[0 * fgr_.h * fgr_.w + i * fgr_.w + j];
            cv_fgr.at<cv::Vec3f>(i, j)[1] = fgr_data[1 * fgr_.h * fgr_.w + i * fgr_.w + j];
            cv_fgr.at<cv::Vec3f>(i, j)[0] = fgr_data[2 * fgr_.h * fgr_.w + i * fgr_.w + j];
        }
    }

    cv_pha.copyTo(pha);
    cv_fgr.copyTo(fgr);

    return 0;
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

    cv::Mat fgr, pha;
    detect_rvm(m, pha, fgr);
    draw_objects(m, fgr, pha);

    return 0;
}
