// Copyright 2025 heabeounMKTO
// SPDX-License-Identifier: BSD-3-Clause
/* ncnn example using yolo-face and arcface to extract embeddings from a face
 *
 *
 *  the arcface model is converted from
 * https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface
 * 1. first simplify the arcface.onnx using onnxsim
 * 2. then convert it using ncnn's onnx exporter onnx2ncnn
 *  using pnnx to convert would cause -nan output!
 *
 *  the yolov8-face model is converted from
 *  https://github.com/derronqi/yolov8-face
 *
 *
 * you can find the models preconverted at
 * https://drive.google.com/drive/folders/1P0RDzj9V7FHEL8w_-yqls5RHeVpO-2PS?usp=sharing
 *
 * */

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <float.h>
#include "layer.h"
#include "net.h"
#include "mat.h"

#ifndef ARCFACE_EXAMPLE_YOLO_INFER_SIZE
#define ARCFACE_EXAMPLE_YOLO_INFER_SIZE 320
#endif

struct Bbox
{
    float x1, y1, x2, y2, confidence;
    int label;
    Bbox()
        : x1(0.0f), y1(0.0f), x2(0.0f), y2(0.0f), confidence(0.0f), label(0)
    {
    }
    Bbox(float x1,
         float y1,
         float x2,
         float y2,
         float confidence,
         int label = 0,
         std::string label_name = "")
        : x1(x1), y1(y1), x2(x2), y2(y2), confidence(confidence), label(label)
    {
    }
    Bbox apply_image_scale(const cv::Mat& original_image,
                           const float scale_factor,
                           const int pad_w,
                           const int pad_h)
    {
        int img_w = original_image.cols;
        int img_h = original_image.rows;

        x1 = (x1 - pad_w) / scale_factor;
        y1 = (y1 - pad_h) / scale_factor;
        x2 = (x2 - pad_w) / scale_factor;
        y2 = (y2 - pad_h) / scale_factor;

        // clamp
        x1 = std::max(0.0f, std::min(x1, (float)img_w));
        y1 = std::max(0.0f, std::min(y1, (float)img_h));
        x2 = std::max(0.0f, std::min(x2, (float)img_w));
        y2 = std::max(0.0f, std::min(y2, (float)img_h));
        return Bbox(x1, y1, x2, y2, confidence, label);
    }
    std::string get_label_name(const std::vector<std::string>& classes)
    {
        return classes[this->label];
    }

    /// what more do you need to know vro
    float area() const
    {
        float width = x2 - x1;
        float height = y2 - y1;
        return width * height;
    }
    cv::Mat crop_bbox(const cv::Mat& originalImage) const
    {
        // Calculate width and height
        int bbox_width = static_cast<int>(x2 - x1);
        int bbox_height = static_cast<int>(y2 - y1);

        // Ensure valid dimensions
        if (bbox_width <= 0 || bbox_height <= 0)
        {
            fprintf(stderr, "Invalid bounding box dimensions\n");
            return cv::Mat();
        }

        // Ensure coordinates are within image bounds
        int x1_int = static_cast<int>(x1);
        int y1_int = static_cast<int>(y1);
        int x2_int = static_cast<int>(x2);
        int y2_int = static_cast<int>(y2);

        // Clamp to image bounds
        x1_int = std::max(0, x1_int);
        y1_int = std::max(0, y1_int);
        x2_int = std::min(originalImage.cols, x2_int);
        y2_int = std::min(originalImage.rows, y2_int);

        // Create ROI and return cropped image
        cv::Rect roi(x1_int, y1_int, x2_int - x1_int, y2_int - y1_int);
        return originalImage(roi).clone();
    }
    cv::Rect_<float> get_rect() const
    {
        int x1_int = static_cast<int>(x1);
        int y1_int = static_cast<int>(y1);
        int width = static_cast<int>(x2 - x1);
        int height = static_cast<int>(y2 - y1);

        // Ensure valid dimensions
        if (width <= 0 || height <= 0)
        {
            return cv::Rect(0, 0, 0, 0); // Return invalid rect
        }

        return cv::Rect(x1_int, y1_int, width, height);
    }
};

static void print_bbox(Bbox& bbox)
{
    printf("Bbox(x1=%.2f, y1=%.2f, x2=%.2f, y2=%.2f, conf=%.4f, label=%d)\n",
           bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.confidence, bbox.label);
}

static void qsort_descent_inplace(std::vector<Bbox>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].confidence;

    while (i <= j)
    {
        while (faceobjects[i].confidence > p)
            i++;

        while (faceobjects[j].confidence < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Bbox>& faceobjects)
{
    if (faceobjects.empty()) return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

float calculate_iou(const Bbox& box1, const Bbox& box2)
{
    float x1 = std::max(box1.x1, box2.x1);
    float y1 = std::max(box1.y1, box2.y1);
    float x2 = std::min(box1.x2, box2.x2);
    float y2 = std::min(box1.y2, box2.y2);

    if (x2 <= x1 || y2 <= y1)
    {
        return 0.0f; // no intersect
    }

    float intersection_area = (x2 - x1) * (y2 - y1);
    float box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    float union_area = box1_area + box2_area - intersection_area;
    return intersection_area / union_area;
}

static std::vector<int>
non_maximum_supression(const std::vector<Bbox>& bbox, float iou_thresh, bool class_agnostic = false)
{
    std::vector<int> picked;
    const int n = bbox.size();
    if (n == 0) return picked;

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = bbox[i].area();
    }

    for (int i = 0; i < n; i++)
    {
        const Bbox& a = bbox[i];
        bool keep = true;

        for (int j : picked)
        {
            const Bbox& b = bbox[j];

            // Enhanced class comparison logic using labels
            if (!class_agnostic)
            {
                if (a.label != b.label)
                {
                    continue; // Different classes, don't suppress
                }
            }

            float iou = calculate_iou(a, b);
            if (iou > iou_thresh)
            {
                keep = false;
                break;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }

    return picked;
}

static std::vector<float> scale_wh(float w0, float h0, float w1, float h1)
{
    float r = std::min(w1 / w0, h1 / h0);
    std::vector<float> _scale_factor(3);
    _scale_factor[0] = r;
    _scale_factor[1] = (float)std::round(w0 * r);
    _scale_factor[2] = (float)std::round(h0 * r);
    return _scale_factor;
}

struct ImagePreProcessResults
{
    ncnn::Mat result;
    float img_scale, pad_w, pad_h;

    ImagePreProcessResults(ncnn::Mat result, float img_scale, float pad_w, float pad_h)
        : result(result), img_scale(img_scale), pad_w(pad_w), pad_h(pad_h)
    {
    }
};

struct DetectionResult
{
    std::vector<Bbox> bboxes;
    std::vector<std::vector<float> > keypoints;
};

static ImagePreProcessResults preprocess_yolo_kpts(cv::Mat& input_image, int infer_size) noexcept
{
    float mean_vals[] = {0.f, 0.f, 0.f};

    float norm_vals[] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    int img_w = input_image.cols;
    int img_h = input_image.rows;
    float scale_factor, new_w, new_h;
    std::vector<float> _scale_factor = scale_wh(img_w, img_h, (float)infer_size, (float)infer_size);
    scale_factor = _scale_factor[0];
    new_w = _scale_factor[1];
    new_h = _scale_factor[2];
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(input_image.data,
                   ncnn::Mat::PIXEL_BGR2RGB, img_w,
                   img_h, new_w, new_h);

    // padding calculation
    int pad_w = (infer_size - new_w) / 2;
    int pad_h = (infer_size - new_h) / 2;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, pad_h, infer_size - new_h - pad_h, pad_w,
                           infer_size - new_w - pad_w, ncnn::BORDER_CONSTANT, 114.f);
    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    return ImagePreProcessResults(in_pad, scale_factor, pad_w, pad_h);
}

/// parses extra keypoints data for face mmodel
/// the format is this:
/// [x, y, w, h, conf, class_scores..., kp1_conf, kp1_x, kp1_y, kp2_conf, kp2_x, kp2_y,  ...]
static DetectionResult parse_yolo_keypoints_results(ncnn::Mat& result,
        cv::Mat& original_image,
        ImagePreProcessResults& preproc_img,
        float confidence_threshold,
        float iou_threshold,
        std::vector<std::string> class_names)
{
    cv::Mat output((int)result.w, (int)result.h, CV_32FC1);
    for (int i = 0; i < output.cols; i++)
    {
        for (int j = 0; j < output.rows; j++)
        {
            output.ptr<float>(j)[i] = result.row(i)[j];
        }
    }
    std::vector<Bbox> detections;
    std::vector<std::vector<float> > all_keypoints;

    int num_classes = class_names.size();
    int kp_stride = 3;
    int num_keypoints = 5;

    for (int i = 0; i < output.rows; i++)
    {
        const float* row_ptr = output.ptr<float>(i);
        const float* bboxes_ptr = row_ptr;
        const float* classes_ptr = row_ptr + 4;
        const float* max_s_ptr = std::max_element(classes_ptr, classes_ptr + num_classes);

        float score = *max_s_ptr;
        int class_id = max_s_ptr - classes_ptr;

        if (score >= confidence_threshold)
        {
            float x = bboxes_ptr[0];
            float y = bboxes_ptr[1];
            float w = bboxes_ptr[2];
            float h = bboxes_ptr[3];
            float x1 = x - w / 2.0f;
            float y1 = y - h / 2.0f;
            float x2 = x + w / 2.0f;
            float y2 = y + h / 2.0f;

            if (x2 > x1 && y2 > y1)
            {
                Bbox bbox = Bbox(x1, y1, x2, y2, score, class_id)
                            .apply_image_scale(original_image, preproc_img.img_scale,
                                               preproc_img.pad_w, preproc_img.pad_h);
                // Parse exactly 5 keypoints for this face model
                std::vector<float> face_keypoints;
                face_keypoints.reserve(15);
                const float* kp_ptr = row_ptr + 4 + num_classes;
                float scale = 1.0f / preproc_img.img_scale;

                for (int k = 0; k < num_keypoints; k++)
                {
                    float kp_x = kp_ptr[k * kp_stride];
                    float kp_y = kp_ptr[k * kp_stride + 1];
                    float kp_conf_raw = kp_ptr[k * kp_stride + 2];

                    // Apply sigmoid to convert logit to probability
                    float kp_conf = 1.0f / (1.0f + expf(-kp_conf_raw));

                    // Scale keypoints to original
                    kp_x = (kp_x - preproc_img.pad_w) * scale;
                    kp_y = (kp_y - preproc_img.pad_h) * scale;

                    face_keypoints.push_back(kp_x);
                    face_keypoints.push_back(kp_y);
                    face_keypoints.push_back(kp_conf);
                }

                detections.push_back(bbox);
                all_keypoints.push_back(face_keypoints);
            }
        }
    }

    // nms
    qsort_descent_inplace(detections);
    std::vector<int> picked = non_maximum_supression(detections, iou_threshold, false);
    DetectionResult res;
    for (size_t i = 0; i < picked.size(); i++)
    {
        int idx = picked[i];
        res.bboxes.push_back(detections[idx]);
        res.keypoints.push_back(all_keypoints[idx]);
    }

    return res;
}

static inline float get_similarity(std::vector<float> f1, std::vector<float> f2)
{
    float sim = 0.0;
    for (size_t i = 0; i < f1.size(); i++)
    {
        sim += f1[i] * f2[i];
    }
    return sim;
}

// these are converted from here
// https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
static int estimate_norm(float* transform_matrix, const float* lmk, int image_size = 112)
{
    float ARCFACE_DST[] {
        38.2946f, 51.6963f, // left eye
        73.5318f, 51.5014f, // right eye
        56.0252f, 71.7366f, // nose
        41.5493f, 92.3655f, // left mouth
        70.7299f, 92.2041f  // right mouth
    };
    if (image_size % 112 != 0 && image_size % 128 != 0)
    {
        return -1;
    }

    float ratio, diff_x;
    if (image_size % 112 == 0)
    {
        ratio = static_cast<float>(image_size) / 112.0f;
        diff_x = 0.0f;
    }
    else
    {
        ratio = static_cast<float>(image_size) / 128.0f;
        diff_x = 8.0f * ratio;
    }

    float src_points[10];
    for (int i = 0; i < 5; i++)
    {
        src_points[i * 2] = lmk[i * 3];
        src_points[i * 2 + 1] = lmk[i * 3 + 1];
    }

    float dst_points[10];
    for (int i = 0; i < 5; i++)
    {
        dst_points[i * 2] = ARCFACE_DST[i * 2] * ratio + diff_x;
        dst_points[i * 2 + 1] = ARCFACE_DST[i * 2 + 1] * ratio;
    }

    ncnn::get_affine_transform(dst_points, src_points, 5, transform_matrix);

    return 0;
}

static int norm_crop(cv::Mat& output, const cv::Mat& input, const float* lmk, int image_size = 112)
{
    float transform_matrix[6];
    int status = estimate_norm(transform_matrix, lmk, image_size);

    if (status != 0)
    {
        return status;
    }
    output = cv::Mat(image_size, image_size, CV_8UC3);
    ncnn::warpaffine_bilinear_c3(input.data, input.cols, input.rows,
                                 output.data, image_size, image_size,
                                 transform_matrix);
    return 0;
}

void normalize_arcface(std::vector<float>& feature)
{
    if (feature.empty())
        return;
    float sum = 0;
    for (auto it = feature.begin(); it != feature.end(); it++)
        sum += (float)*it * (float)*it;
    sum = sqrt(sum);
    if (sum == 0.0f)
        return;
    for (auto it = feature.begin(); it != feature.end(); it++)
        *it /= sum;
}

static int get_face(const cv::Mat& rgb, DetectionResult& result)
{
    int status = 0;
    ncnn::Net yoloface;
    yoloface.opt.use_vulkan_compute = true;
    status = yoloface.load_param("yolov8-face.param");

    if (status != 0)
    {
        fprintf(stderr, "couldn't load params");
        return status;
    }

    status = yoloface.load_model("yolov8-face.bin");

    if (status != 0)
    {
        fprintf(stderr, "couldn't load model");
        return status;
    }

    cv::Mat input_image = rgb.clone();
    ImagePreProcessResults preproc_img = preprocess_yolo_kpts(input_image, ARCFACE_EXAMPLE_YOLO_INFER_SIZE);
    ncnn::Extractor ex = yoloface.create_extractor();
    ex.input("in0", preproc_img.result);
    ncnn::Mat out;
    ex.extract("out0", out);
    std::vector<std::string> class_names = {"face"};
    result = parse_yolo_keypoints_results(out, input_image, preproc_img, 0.5, 0.4, class_names);
    if (result.bboxes.size() < 1)
    {
        fprintf(stderr, "no faces are found!");
        return -1;
    }
    return 0;
}

static int get_embedding(const cv::Mat& rgb, std::vector<float>& result)
{
    ncnn::Net arcface;
    arcface.opt.use_vulkan_compute = true;
    int status = arcface.load_param("arcfaceresnet.param");
    if (status != 0)
    {
        fprintf(stderr, "couldn't load arcface params");
        return status;
    }
    status = arcface.load_model("arcfaceresnet.bin");
    if (status != 0)
    {
        fprintf(stderr, "couldn't load arcface model");
        return status;
    }

    if (rgb.empty() || rgb.type() != CV_8UC3)
    {
        fprintf(stderr, "invalid input image!");
        return -1;
    }
    /*
    * the arcface model provided in the link has builtin normalization layers,
    * no need to run substract_mean_normalize
    *
    *  reference from .param
    BinaryOp         _minusscalar0            2 1 data scalar_op2 _minusscalar0 0=1
    BinaryOp         _mulscalar0              2 1 _minusscalar0 scalar_op3 _mulscalar0 0=2
    * */
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
                       rgb.data,
                       ncnn::Mat::PIXEL_BGR2RGB,
                       rgb.cols,
                       rgb.rows,
                       112,
                       112);
    ncnn::Extractor ex = arcface.create_extractor();
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("fc1", out);
    const float* ptr = (const float*)out.data;
    for (int i = 0; i < 512; i++)
    {
        result[i] = ptr[i];
    }
    normalize_arcface(result);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <face1_path> <face2_path>\n", argv[0]);
        return -1;
    }

    const char* face1_path = argv[1];
    const char* face2_path = argv[2];

    int status = 0;
    cv::Mat face_img1 = cv::imread(face1_path);
    cv::Mat face_img2 = cv::imread(face2_path);

    if (face_img1.empty())
    {
        fprintf(stderr, "Failed to load image: %s\n", face1_path);
        return -1;
    }
    if (face_img2.empty())
    {
        fprintf(stderr, "Failed to load image: %s\n", face2_path);
        return -1;
    }

    cv::Mat input_embed1, input_embed2;
    DetectionResult res1, res2;
    std::vector<float> embedding1(512), embedding2(512);

    status = get_face(face_img1, res1);
    if (status != 0)
    {
        fprintf(stderr, "get face failed for %s!\n", face1_path);
        return -1;
    }
    fprintf(stdout, "found faces in face1: %d\n", (int)res1.bboxes.size());
    for (size_t i = 0; i < res1.bboxes.size(); i++)
    {
        print_bbox(res1.bboxes[i]);
    }

    status = get_face(face_img2, res2);
    if (status != 0)
    {
        fprintf(stderr, "get face failed for %s!\n", face2_path);
        return -1;
    }
    fprintf(stdout, "found faces in face2: %d\n", (int)res2.bboxes.size());
    for (size_t i = 0; i < res2.bboxes.size(); i++)
    {
        print_bbox(res2.bboxes[i]);
    }

    status = norm_crop(input_embed1, face_img1, res1.keypoints[0].data());
    status = get_embedding(input_embed1, embedding1);
    if (status != 0)
    {
        fprintf(stderr, "get embedding failed for %s!\n", face1_path);
        return -1;
    }

    status = norm_crop(input_embed2, face_img2, res2.keypoints[0].data());
    if (status != 0)
    {
        fprintf(stderr, "norm_crop failed for face2!\n");
        return -1;
    }
    status = get_embedding(input_embed2, embedding2);
    if (status != 0)
    {
        fprintf(stderr, "get embedding failed for face2!\n");
        return -1;
    }
    if (status != 0)
    {
        fprintf(stderr, "get embedding failed for %s!\n", face2_path);
        return -1;
    }

    float similarity = get_similarity(embedding1, embedding2);
    fprintf(stdout, "Similarity: %f\n", similarity);
}
