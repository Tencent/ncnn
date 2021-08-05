// Tencent is pleased to support the open source community by making ncnn available.
//
// author:BUG1989 (https://github.com/BUG1989/) Long-term support.
// author:JansonZhu (https://github.com/JansonZhu) Implemented the function of entropy calibration.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#elif defined(USE_LOCAL_IMREADWRITE)
#include "imreadwrite.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <string>
#include <vector>

// ncnn public header
#include "benchmark.h"
#include "cpu.h"
#include "net.h"

// ncnn private header
#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/innerproduct.h"

class QuantBlobStat
{
public:
    QuantBlobStat()
    {
        threshold = 0.f;
        absmax = 0.f;
        total = 0;
    }

public:
    float threshold;
    float absmax;

    // ACIQ
    int total;

    // KL
    std::vector<uint64_t> histogram;
    std::vector<float> histogram_normed;
};

class QuantNet : public ncnn::Net
{
public:
    QuantNet();

    std::vector<ncnn::Blob>& blobs;
    std::vector<ncnn::Layer*>& layers;

public:
    std::vector<std::vector<std::string> > listspaths;
    std::vector<std::vector<float> > means;
    std::vector<std::vector<float> > norms;
    std::vector<std::vector<int> > shapes;
    std::vector<int> type_to_pixels;
    int quantize_num_threads;

public:
    int init();
    void print_quant_info() const;
    int save_table(const char* tablepath);
    int quantize_KL();
    int quantize_ACIQ();
    int quantize_EQ();

public:
    std::vector<int> input_blobs;
    std::vector<int> conv_layers;
    std::vector<int> conv_bottom_blobs;
    std::vector<int> conv_top_blobs;

    // result
    std::vector<QuantBlobStat> quant_blob_stats;
    std::vector<ncnn::Mat> weight_scales;
    std::vector<ncnn::Mat> bottom_blob_scales;
};

QuantNet::QuantNet()
    : blobs(mutable_blobs()), layers(mutable_layers())
{
    quantize_num_threads = ncnn::get_cpu_count();
}

int QuantNet::init()
{
    // find all input layers
    for (int i = 0; i < (int)layers.size(); i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "Input")
        {
            input_blobs.push_back(layer->tops[0]);
        }
    }

    // find all conv layers
    for (int i = 0; i < (int)layers.size(); i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise" || layer->type == "InnerProduct")
        {
            conv_layers.push_back(i);
            conv_bottom_blobs.push_back(layer->bottoms[0]);
            conv_top_blobs.push_back(layer->tops[0]);
        }
    }

    const int conv_layer_count = (int)conv_layers.size();
    const int conv_bottom_blob_count = (int)conv_bottom_blobs.size();

    quant_blob_stats.resize(conv_bottom_blob_count);
    weight_scales.resize(conv_layer_count);
    bottom_blob_scales.resize(conv_bottom_blob_count);

    return 0;
}

int QuantNet::save_table(const char* tablepath)
{
    FILE* fp = fopen(tablepath, "wb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", tablepath);
        return -1;
    }

    const int conv_layer_count = (int)conv_layers.size();
    const int conv_bottom_blob_count = (int)conv_bottom_blobs.size();

    for (int i = 0; i < conv_layer_count; i++)
    {
        const ncnn::Mat& weight_scale = weight_scales[i];

        fprintf(fp, "%s_param_0 ", layers[conv_layers[i]]->name.c_str());
        for (int j = 0; j < weight_scale.w; j++)
        {
            fprintf(fp, "%f ", weight_scale[j]);
        }
        fprintf(fp, "\n");
    }

    for (int i = 0; i < conv_bottom_blob_count; i++)
    {
        const ncnn::Mat& bottom_blob_scale = bottom_blob_scales[i];

        fprintf(fp, "%s ", layers[conv_layers[i]]->name.c_str());
        for (int j = 0; j < bottom_blob_scale.w; j++)
        {
            fprintf(fp, "%f ", bottom_blob_scale[j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    fprintf(stderr, "ncnn int8 calibration table create success, best wish for your int8 inference has a low accuracy loss...\\(^0^)/...233...\n");

    return 0;
}

void QuantNet::print_quant_info() const
{
    for (int i = 0; i < (int)conv_bottom_blobs.size(); i++)
    {
        const QuantBlobStat& stat = quant_blob_stats[i];

        float scale = 127 / stat.threshold;

        fprintf(stderr, "%-40s : max = %-15f  threshold = %-15f  scale = %-15f\n", layers[conv_layers[i]]->name.c_str(), stat.absmax, stat.threshold, scale);
    }
}

/**
 * Read and resize image
 * shape is input as [w,h,...]
 * if w and h both are given, image will be resized to exactly size.
 * if w and h both are zero or negative, image will not be resized.
 * if only h is zero or negative, image's width will scaled resize to w, keeping aspect ratio.
 * if only w is zero or negative, image's height will scaled resize to h
 * @return ncnn::Mat
 */

inline ncnn::Mat read_and_resize_image(const std::vector<int>& shape, const std::string& imagepath, int pixel_convert_type)
{
    int target_w = shape[0];
    int target_h = shape[1];
    cv::Mat bgr = cv::imread(imagepath, 1);
    if (target_h <= 0 && target_w <= 0)
    {
        return ncnn::Mat::from_pixels(bgr.data, pixel_convert_type, bgr.cols, bgr.rows);
    }
    if (target_h <= 0 || target_w <= 0)
    {
        float scale = 1.0;
        if (target_h <= 0)
        {
            scale = 1.0 * bgr.cols / target_w;
            target_h = int(1.0 * bgr.rows / scale);
        }
        if (target_w <= 0)
        {
            scale = 1.0 * bgr.rows / target_h;
            target_w = int(1.0 * bgr.cols / scale);
        }
    }
    return ncnn::Mat::from_pixels_resize(bgr.data, pixel_convert_type, bgr.cols, bgr.rows, target_w, target_h);
}

static float compute_kl_divergence(const std::vector<float>& a, const std::vector<float>& b)
{
    const size_t length = a.size();

    float result = 0;
    for (size_t i = 0; i < length; i++)
    {
        result += a[i] * log(a[i] / b[i]);
    }

    return result;
}

int QuantNet::quantize_KL()
{
    const int input_blob_count = (int)input_blobs.size();
    const int conv_layer_count = (int)conv_layers.size();
    const int conv_bottom_blob_count = (int)conv_bottom_blobs.size();
    const int image_count = (int)listspaths[0].size();

    const int num_histogram_bins = 2048;

    std::vector<ncnn::UnlockedPoolAllocator> blob_allocators(quantize_num_threads);
    std::vector<ncnn::UnlockedPoolAllocator> workspace_allocators(quantize_num_threads);

    // initialize conv weight scales
    #pragma omp parallel for num_threads(quantize_num_threads)
    for (int i = 0; i < conv_layer_count; i++)
    {
        const ncnn::Layer* layer = layers[conv_layers[i]];

        if (layer->type == "Convolution")
        {
            const ncnn::Convolution* convolution = (const ncnn::Convolution*)layer;

            const int num_output = convolution->num_output;
            const int kernel_w = convolution->kernel_w;
            const int kernel_h = convolution->kernel_h;
            const int dilation_w = convolution->dilation_w;
            const int dilation_h = convolution->dilation_h;
            const int stride_w = convolution->stride_w;
            const int stride_h = convolution->stride_h;

            const int weight_data_size_output = convolution->weight_data_size / num_output;

            // int8 winograd F43 needs weight data to use 6bit quantization
            // TODO proper condition for winograd 3x3 int8
            bool quant_6bit = false;
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                quant_6bit = true;

            weight_scales[i].create(num_output);

            for (int n = 0; n < num_output; n++)
            {
                const ncnn::Mat weight_data_n = convolution->weight_data.range(weight_data_size_output * n, weight_data_size_output);

                float absmax = 0.f;
                for (int k = 0; k < weight_data_size_output; k++)
                {
                    absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
                }

                if (quant_6bit)
                {
                    weight_scales[i][n] = 31 / absmax;
                }
                else
                {
                    weight_scales[i][n] = 127 / absmax;
                }
            }
        }

        if (layer->type == "ConvolutionDepthWise")
        {
            const ncnn::ConvolutionDepthWise* convolutiondepthwise = (const ncnn::ConvolutionDepthWise*)layer;

            const int group = convolutiondepthwise->group;
            const int weight_data_size_output = convolutiondepthwise->weight_data_size / group;

            std::vector<float> scales;

            weight_scales[i].create(group);

            for (int n = 0; n < group; n++)
            {
                const ncnn::Mat weight_data_n = convolutiondepthwise->weight_data.range(weight_data_size_output * n, weight_data_size_output);

                float absmax = 0.f;
                for (int k = 0; k < weight_data_size_output; k++)
                {
                    absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
                }

                weight_scales[i][n] = 127 / absmax;
            }
        }

        if (layer->type == "InnerProduct")
        {
            const ncnn::InnerProduct* innerproduct = (const ncnn::InnerProduct*)layer;

            const int num_output = innerproduct->num_output;
            const int weight_data_size_output = innerproduct->weight_data_size / num_output;

            weight_scales[i].create(num_output);

            for (int n = 0; n < num_output; n++)
            {
                const ncnn::Mat weight_data_n = innerproduct->weight_data.range(weight_data_size_output * n, weight_data_size_output);

                float absmax = 0.f;
                for (int k = 0; k < weight_data_size_output; k++)
                {
                    absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
                }

                weight_scales[i][n] = 127 / absmax;
            }
        }
    }

    // count the absmax
    #pragma omp parallel for num_threads(quantize_num_threads) schedule(static, 1)
    for (int i = 0; i < image_count; i++)
    {
        if (i % 100 == 0)
        {
            fprintf(stderr, "count the absmax %.2f%% [ %d / %d ]\n", i * 100.f / image_count, i, image_count);
        }

        ncnn::Extractor ex = create_extractor();

        const int thread_num = ncnn::get_omp_thread_num();
        ex.set_blob_allocator(&blob_allocators[thread_num]);
        ex.set_workspace_allocator(&workspace_allocators[thread_num]);

        for (int j = 0; j < input_blob_count; j++)
        {
            const int type_to_pixel = type_to_pixels[j];
            const std::vector<float>& mean_vals = means[j];
            const std::vector<float>& norm_vals = norms[j];

            int pixel_convert_type = ncnn::Mat::PIXEL_BGR;
            if (type_to_pixel != pixel_convert_type)
            {
                pixel_convert_type = pixel_convert_type | (type_to_pixel << ncnn::Mat::PIXEL_CONVERT_SHIFT);
            }

            ncnn::Mat in = read_and_resize_image(shapes[j], listspaths[j][i], pixel_convert_type);

            in.substract_mean_normalize(mean_vals.data(), norm_vals.data());

            ex.input(input_blobs[j], in);
        }

        for (int j = 0; j < conv_bottom_blob_count; j++)
        {
            ncnn::Mat out;
            ex.extract(conv_bottom_blobs[j], out);

            // count absmax
            {
                float absmax = 0.f;

                const int outc = out.c;
                const int outsize = out.w * out.h;
                for (int p = 0; p < outc; p++)
                {
                    const float* ptr = out.channel(p);
                    for (int k = 0; k < outsize; k++)
                    {
                        absmax = std::max(absmax, (float)fabs(ptr[k]));
                    }
                }

                #pragma omp critical
                {
                    QuantBlobStat& stat = quant_blob_stats[j];
                    stat.absmax = std::max(stat.absmax, absmax);
                }
            }
        }
    }

    // initialize histogram
    #pragma omp parallel for num_threads(quantize_num_threads)
    for (int i = 0; i < conv_bottom_blob_count; i++)
    {
        QuantBlobStat& stat = quant_blob_stats[i];

        stat.histogram.resize(num_histogram_bins, 0);
        stat.histogram_normed.resize(num_histogram_bins, 0);
    }

    // build histogram
    #pragma omp parallel for num_threads(quantize_num_threads) schedule(static, 1)
    for (int i = 0; i < image_count; i++)
    {
        if (i % 100 == 0)
        {
            fprintf(stderr, "build histogram %.2f%% [ %d / %d ]\n", i * 100.f / image_count, i, image_count);
        }

        ncnn::Extractor ex = create_extractor();

        const int thread_num = ncnn::get_omp_thread_num();
        ex.set_blob_allocator(&blob_allocators[thread_num]);
        ex.set_workspace_allocator(&workspace_allocators[thread_num]);

        for (int j = 0; j < input_blob_count; j++)
        {
            const int type_to_pixel = type_to_pixels[j];
            const std::vector<float>& mean_vals = means[j];
            const std::vector<float>& norm_vals = norms[j];

            int pixel_convert_type = ncnn::Mat::PIXEL_BGR;
            if (type_to_pixel != pixel_convert_type)
            {
                pixel_convert_type = pixel_convert_type | (type_to_pixel << ncnn::Mat::PIXEL_CONVERT_SHIFT);
            }

            ncnn::Mat in = read_and_resize_image(shapes[j], listspaths[j][i], pixel_convert_type);

            in.substract_mean_normalize(mean_vals.data(), norm_vals.data());

            ex.input(input_blobs[j], in);
        }

        for (int j = 0; j < conv_bottom_blob_count; j++)
        {
            ncnn::Mat out;
            ex.extract(conv_bottom_blobs[j], out);

            // count histogram bin
            {
                const float absmax = quant_blob_stats[j].absmax;

                std::vector<uint64_t> histogram(num_histogram_bins, 0);

                const int outc = out.c;
                const int outsize = out.w * out.h;
                for (int p = 0; p < outc; p++)
                {
                    const float* ptr = out.channel(p);
                    for (int k = 0; k < outsize; k++)
                    {
                        if (ptr[k] == 0.f)
                            continue;

                        const int index = std::min((int)(fabs(ptr[k]) / absmax * num_histogram_bins), (num_histogram_bins - 1));

                        histogram[index] += 1;
                    }
                }

                #pragma omp critical
                {
                    QuantBlobStat& stat = quant_blob_stats[j];

                    for (int k = 0; k < num_histogram_bins; k++)
                    {
                        stat.histogram[k] += histogram[k];
                    }
                }
            }
        }
    }

    // using kld to find the best threshold value
    #pragma omp parallel for num_threads(quantize_num_threads)
    for (int i = 0; i < conv_bottom_blob_count; i++)
    {
        QuantBlobStat& stat = quant_blob_stats[i];

        // normalize histogram bin
        {
            uint64_t sum = 0;
            for (int j = 0; j < num_histogram_bins; j++)
            {
                sum += stat.histogram[j];
            }

            for (int j = 0; j < num_histogram_bins; j++)
            {
                stat.histogram_normed[j] = (float)(stat.histogram[j] / (double)sum);
            }
        }

        const int target_bin = 128;

        int target_threshold = target_bin;
        float min_kl_divergence = FLT_MAX;

        for (int threshold = target_bin; threshold < num_histogram_bins; threshold++)
        {
            const float kl_eps = 0.0001f;

            std::vector<float> clip_distribution(threshold, kl_eps);
            {
                for (int j = 0; j < threshold; j++)
                {
                    clip_distribution[j] += stat.histogram_normed[j];
                }
                for (int j = threshold; j < num_histogram_bins; j++)
                {
                    clip_distribution[threshold - 1] += stat.histogram_normed[j];
                }
            }

            const float num_per_bin = (float)threshold / target_bin;

            std::vector<float> quantize_distribution(target_bin, 0.f);
            {
                {
                    const float end = num_per_bin;

                    const int right_lower = (int)floor(end);
                    const float right_scale = end - right_lower;

                    if (right_scale > 0)
                    {
                        quantize_distribution[0] += right_scale * stat.histogram_normed[right_lower];
                    }

                    for (int k = 0; k < right_lower; k++)
                    {
                        quantize_distribution[0] += stat.histogram_normed[k];
                    }

                    quantize_distribution[0] /= right_lower + right_scale;
                }
                for (int j = 1; j < target_bin - 1; j++)
                {
                    const float start = j * num_per_bin;
                    const float end = (j + 1) * num_per_bin;

                    const int left_upper = (int)ceil(start);
                    const float left_scale = left_upper - start;

                    const int right_lower = (int)floor(end);
                    const float right_scale = end - right_lower;

                    if (left_scale > 0)
                    {
                        quantize_distribution[j] += left_scale * stat.histogram_normed[left_upper - 1];
                    }

                    if (right_scale > 0)
                    {
                        quantize_distribution[j] += right_scale * stat.histogram_normed[right_lower];
                    }

                    for (int k = left_upper; k < right_lower; k++)
                    {
                        quantize_distribution[j] += stat.histogram_normed[k];
                    }

                    quantize_distribution[j] /= right_lower - left_upper + left_scale + right_scale;
                }
                {
                    const float start = threshold - num_per_bin;

                    const int left_upper = (int)ceil(start);
                    const float left_scale = left_upper - start;

                    if (left_scale > 0)
                    {
                        quantize_distribution[target_bin - 1] += left_scale * stat.histogram_normed[left_upper - 1];
                    }

                    for (int k = left_upper; k < threshold; k++)
                    {
                        quantize_distribution[target_bin - 1] += stat.histogram_normed[k];
                    }

                    quantize_distribution[target_bin - 1] /= threshold - left_upper + left_scale;
                }
            }

            std::vector<float> expand_distribution(threshold, kl_eps);
            {
                {
                    const float end = num_per_bin;

                    const int right_lower = (int)floor(end);
                    const float right_scale = end - right_lower;

                    if (right_scale > 0)
                    {
                        expand_distribution[right_lower] += right_scale * quantize_distribution[0];
                    }

                    for (int k = 0; k < right_lower; k++)
                    {
                        expand_distribution[k] += quantize_distribution[0];
                    }
                }
                for (int j = 1; j < target_bin - 1; j++)
                {
                    const float start = j * num_per_bin;
                    const float end = (j + 1) * num_per_bin;

                    const int left_upper = (int)ceil(start);
                    const float left_scale = left_upper - start;

                    const int right_lower = (int)floor(end);
                    const float right_scale = end - right_lower;

                    if (left_scale > 0)
                    {
                        expand_distribution[left_upper - 1] += left_scale * quantize_distribution[j];
                    }

                    if (right_scale > 0)
                    {
                        expand_distribution[right_lower] += right_scale * quantize_distribution[j];
                    }

                    for (int k = left_upper; k < right_lower; k++)
                    {
                        expand_distribution[k] += quantize_distribution[j];
                    }
                }
                {
                    const float start = threshold - num_per_bin;

                    const int left_upper = (int)ceil(start);
                    const float left_scale = left_upper - start;

                    if (left_scale > 0)
                    {
                        expand_distribution[left_upper - 1] += left_scale * quantize_distribution[target_bin - 1];
                    }

                    for (int k = left_upper; k < threshold; k++)
                    {
                        expand_distribution[k] += quantize_distribution[target_bin - 1];
                    }
                }
            }

            // kl
            const float kl_divergence = compute_kl_divergence(clip_distribution, expand_distribution);

            // the best num of bin
            if (kl_divergence < min_kl_divergence)
            {
                min_kl_divergence = kl_divergence;
                target_threshold = threshold;
            }
        }

        stat.threshold = (target_threshold + 0.5f) * stat.absmax / num_histogram_bins;
        float scale = 127 / stat.threshold;

        bottom_blob_scales[i].create(1);
        bottom_blob_scales[i][0] = scale;
    }

    return 0;
}

static float compute_aciq_gaussian_clip(float absmax, int N, int num_bits = 8)
{
    const float alpha_gaussian[8] = {0, 1.71063519, 2.15159277, 2.55913646, 2.93620062, 3.28691474, 3.6151146, 3.92403714};

    const double gaussian_const = (0.5 * 0.35) * (1 + sqrt(3.14159265358979323846 * log(4)));

    double std = (absmax * 2 * gaussian_const) / sqrt(2 * log(N));

    return (float)(alpha_gaussian[num_bits - 1] * std);
}

int QuantNet::quantize_ACIQ()
{
    const int input_blob_count = (int)input_blobs.size();
    const int conv_layer_count = (int)conv_layers.size();
    const int conv_bottom_blob_count = (int)conv_bottom_blobs.size();
    const int image_count = (int)listspaths[0].size();

    std::vector<ncnn::UnlockedPoolAllocator> blob_allocators(quantize_num_threads);
    std::vector<ncnn::UnlockedPoolAllocator> workspace_allocators(quantize_num_threads);

    // initialize conv weight scales
    #pragma omp parallel for num_threads(quantize_num_threads)
    for (int i = 0; i < conv_layer_count; i++)
    {
        const ncnn::Layer* layer = layers[conv_layers[i]];

        if (layer->type == "Convolution")
        {
            const ncnn::Convolution* convolution = (const ncnn::Convolution*)layer;

            const int num_output = convolution->num_output;
            const int kernel_w = convolution->kernel_w;
            const int kernel_h = convolution->kernel_h;
            const int dilation_w = convolution->dilation_w;
            const int dilation_h = convolution->dilation_h;
            const int stride_w = convolution->stride_w;
            const int stride_h = convolution->stride_h;

            const int weight_data_size_output = convolution->weight_data_size / num_output;

            // int8 winograd F43 needs weight data to use 6bit quantization
            // TODO proper condition for winograd 3x3 int8
            bool quant_6bit = false;
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                quant_6bit = true;

            weight_scales[i].create(num_output);

            for (int n = 0; n < num_output; n++)
            {
                const ncnn::Mat weight_data_n = convolution->weight_data.range(weight_data_size_output * n, weight_data_size_output);

                float absmax = 0.f;
                for (int k = 0; k < weight_data_size_output; k++)
                {
                    absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
                }

                if (quant_6bit)
                {
                    const float threshold = compute_aciq_gaussian_clip(absmax, weight_data_size_output, 6);
                    weight_scales[i][n] = 31 / threshold;
                }
                else
                {
                    const float threshold = compute_aciq_gaussian_clip(absmax, weight_data_size_output);
                    weight_scales[i][n] = 127 / threshold;
                }
            }
        }

        if (layer->type == "ConvolutionDepthWise")
        {
            const ncnn::ConvolutionDepthWise* convolutiondepthwise = (const ncnn::ConvolutionDepthWise*)layer;

            const int group = convolutiondepthwise->group;
            const int weight_data_size_output = convolutiondepthwise->weight_data_size / group;

            std::vector<float> scales;

            weight_scales[i].create(group);

            for (int n = 0; n < group; n++)
            {
                const ncnn::Mat weight_data_n = convolutiondepthwise->weight_data.range(weight_data_size_output * n, weight_data_size_output);

                float absmax = 0.f;
                for (int k = 0; k < weight_data_size_output; k++)
                {
                    absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
                }

                const float threshold = compute_aciq_gaussian_clip(absmax, weight_data_size_output);
                weight_scales[i][n] = 127 / threshold;
            }
        }

        if (layer->type == "InnerProduct")
        {
            const ncnn::InnerProduct* innerproduct = (const ncnn::InnerProduct*)layer;

            const int num_output = innerproduct->num_output;
            const int weight_data_size_output = innerproduct->weight_data_size / num_output;

            weight_scales[i].create(num_output);

            for (int n = 0; n < num_output; n++)
            {
                const ncnn::Mat weight_data_n = innerproduct->weight_data.range(weight_data_size_output * n, weight_data_size_output);

                float absmax = 0.f;
                for (int k = 0; k < weight_data_size_output; k++)
                {
                    absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
                }

                const float threshold = compute_aciq_gaussian_clip(absmax, weight_data_size_output);
                weight_scales[i][n] = 127 / threshold;
            }
        }
    }

    // count the absmax
    #pragma omp parallel for num_threads(quantize_num_threads) schedule(static, 1)
    for (int i = 0; i < image_count; i++)
    {
        if (i % 100 == 0)
        {
            fprintf(stderr, "count the absmax %.2f%% [ %d / %d ]\n", i * 100.f / image_count, i, image_count);
        }

        ncnn::Extractor ex = create_extractor();

        const int thread_num = ncnn::get_omp_thread_num();
        ex.set_blob_allocator(&blob_allocators[thread_num]);
        ex.set_workspace_allocator(&workspace_allocators[thread_num]);

        for (int j = 0; j < input_blob_count; j++)
        {
            const int type_to_pixel = type_to_pixels[j];
            const std::vector<float>& mean_vals = means[j];
            const std::vector<float>& norm_vals = norms[j];

            int pixel_convert_type = ncnn::Mat::PIXEL_BGR;
            if (type_to_pixel != pixel_convert_type)
            {
                pixel_convert_type = pixel_convert_type | (type_to_pixel << ncnn::Mat::PIXEL_CONVERT_SHIFT);
            }

            ncnn::Mat in = read_and_resize_image(shapes[j], listspaths[j][i], pixel_convert_type);

            in.substract_mean_normalize(mean_vals.data(), norm_vals.data());

            ex.input(input_blobs[j], in);
        }

        for (int j = 0; j < conv_bottom_blob_count; j++)
        {
            ncnn::Mat out;
            ex.extract(conv_bottom_blobs[j], out);

            // count absmax
            {
                float absmax = 0.f;

                const int outc = out.c;
                const int outsize = out.w * out.h;
                for (int p = 0; p < outc; p++)
                {
                    const float* ptr = out.channel(p);
                    for (int k = 0; k < outsize; k++)
                    {
                        absmax = std::max(absmax, (float)fabs(ptr[k]));
                    }
                }

                #pragma omp critical
                {
                    QuantBlobStat& stat = quant_blob_stats[j];
                    stat.absmax = std::max(stat.absmax, absmax);
                    stat.total = outc * outsize;
                }
            }
        }
    }

    // alpha gaussian
    #pragma omp parallel for num_threads(quantize_num_threads)
    for (int i = 0; i < conv_bottom_blob_count; i++)
    {
        QuantBlobStat& stat = quant_blob_stats[i];

        stat.threshold = compute_aciq_gaussian_clip(stat.absmax, stat.total);
        float scale = 127 / stat.threshold;

        bottom_blob_scales[i].create(1);
        bottom_blob_scales[i][0] = scale;
    }

    return 0;
}

static float cosine_similarity(const ncnn::Mat& a, const ncnn::Mat& b)
{
    const int chanenls = a.c;
    const int size = a.w * a.h;

    float sa = 0;
    float sb = 0;
    float sum = 0;

    for (int p = 0; p < chanenls; p++)
    {
        const float* pa = a.channel(p);
        const float* pb = b.channel(p);

        for (int i = 0; i < size; i++)
        {
            sa += pa[i] * pa[i];
            sb += pb[i] * pb[i];
            sum += pa[i] * pb[i];
        }
    }

    float sim = (float)sum / sqrt(sa) / sqrt(sb);

    return sim;
}

static int get_layer_param(const ncnn::Layer* layer, ncnn::ParamDict& pd)
{
    if (layer->type == "Convolution")
    {
        ncnn::Convolution* convolution = (ncnn::Convolution*)layer;

        pd.set(0, convolution->num_output);
        pd.set(1, convolution->kernel_w);
        pd.set(11, convolution->kernel_h);
        pd.set(2, convolution->dilation_w);
        pd.set(12, convolution->dilation_h);
        pd.set(3, convolution->stride_w);
        pd.set(13, convolution->stride_h);
        pd.set(4, convolution->pad_left);
        pd.set(15, convolution->pad_right);
        pd.set(14, convolution->pad_top);
        pd.set(16, convolution->pad_bottom);
        pd.set(18, convolution->pad_value);
        pd.set(5, convolution->bias_term);
        pd.set(6, convolution->weight_data_size);
        pd.set(8, convolution->int8_scale_term);
        pd.set(9, convolution->activation_type);
        pd.set(10, convolution->activation_params);
    }
    else if (layer->type == "ConvolutionDepthWise")
    {
        ncnn::ConvolutionDepthWise* convolutiondepthwise = (ncnn::ConvolutionDepthWise*)layer;

        pd.set(0, convolutiondepthwise->num_output);
        pd.set(1, convolutiondepthwise->kernel_w);
        pd.set(11, convolutiondepthwise->kernel_h);
        pd.set(2, convolutiondepthwise->dilation_w);
        pd.set(12, convolutiondepthwise->dilation_h);
        pd.set(3, convolutiondepthwise->stride_w);
        pd.set(13, convolutiondepthwise->stride_h);
        pd.set(4, convolutiondepthwise->pad_left);
        pd.set(15, convolutiondepthwise->pad_right);
        pd.set(14, convolutiondepthwise->pad_top);
        pd.set(16, convolutiondepthwise->pad_bottom);
        pd.set(18, convolutiondepthwise->pad_value);
        pd.set(5, convolutiondepthwise->bias_term);
        pd.set(6, convolutiondepthwise->weight_data_size);
        pd.set(7, convolutiondepthwise->group);
        pd.set(8, convolutiondepthwise->int8_scale_term);
        pd.set(9, convolutiondepthwise->activation_type);
        pd.set(10, convolutiondepthwise->activation_params);
    }
    else if (layer->type == "InnerProduct")
    {
        ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)layer;

        pd.set(0, innerproduct->num_output);
        pd.set(1, innerproduct->bias_term);
        pd.set(2, innerproduct->weight_data_size);
        pd.set(8, innerproduct->int8_scale_term);
        pd.set(9, innerproduct->activation_type);
        pd.set(10, innerproduct->activation_params);
    }
    else
    {
        fprintf(stderr, "unexpected layer type %s in get_layer_param\n", layer->type.c_str());
        return -1;
    }

    return 0;
}

static int get_layer_weights(const ncnn::Layer* layer, std::vector<ncnn::Mat>& weights)
{
    if (layer->type == "Convolution")
    {
        ncnn::Convolution* convolution = (ncnn::Convolution*)layer;
        weights.push_back(convolution->weight_data);
        if (convolution->bias_term)
            weights.push_back(convolution->bias_data);
    }
    else if (layer->type == "ConvolutionDepthWise")
    {
        ncnn::ConvolutionDepthWise* convolutiondepthwise = (ncnn::ConvolutionDepthWise*)layer;
        weights.push_back(convolutiondepthwise->weight_data);
        if (convolutiondepthwise->bias_term)
            weights.push_back(convolutiondepthwise->bias_data);
    }
    else if (layer->type == "InnerProduct")
    {
        ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)layer;
        weights.push_back(innerproduct->weight_data);
        if (innerproduct->bias_term)
            weights.push_back(innerproduct->bias_data);
    }
    else
    {
        fprintf(stderr, "unexpected layer type %s in get_layer_weights\n", layer->type.c_str());
        return -1;
    }

    return 0;
}

int QuantNet::quantize_EQ()
{
    // find the initial scale via KL
    quantize_KL();

    print_quant_info();

    const int input_blob_count = (int)input_blobs.size();
    const int conv_layer_count = (int)conv_layers.size();
    const int conv_bottom_blob_count = (int)conv_bottom_blobs.size();

    std::vector<ncnn::UnlockedPoolAllocator> blob_allocators(quantize_num_threads);
    std::vector<ncnn::UnlockedPoolAllocator> workspace_allocators(quantize_num_threads);

    // max 50 images for EQ
    const int image_count = std::min((int)listspaths[0].size(), 50);

    const float scale_range_lower = 0.5f;
    const float scale_range_upper = 2.0f;
    const int search_steps = 100;

    for (int i = 0; i < conv_layer_count; i++)
    {
        ncnn::Mat& weight_scale = weight_scales[i];
        ncnn::Mat& bottom_blob_scale = bottom_blob_scales[i];

        const ncnn::Layer* layer = layers[conv_layers[i]];

        // search weight scale
        for (int j = 0; j < weight_scale.w; j++)
        {
            const float scale = weight_scale[j];
            const float scale_lower = scale * scale_range_lower;
            const float scale_upper = scale * scale_range_upper;
            const float scale_step = (scale_upper - scale_lower) / search_steps;

            std::vector<double> avgsims(search_steps, 0.0);

            #pragma omp parallel for num_threads(quantize_num_threads) schedule(static, 1)
            for (int ii = 0; ii < image_count; ii++)
            {
                if (ii % 100 == 0)
                {
                    fprintf(stderr, "search weight scale %.2f%% [ %d / %d ] for %d / %d of %d / %d\n", ii * 100.f / image_count, ii, image_count, j, weight_scale.w, i, conv_layer_count);
                }

                ncnn::Extractor ex = create_extractor();

                const int thread_num = ncnn::get_omp_thread_num();
                ex.set_blob_allocator(&blob_allocators[thread_num]);
                ex.set_workspace_allocator(&workspace_allocators[thread_num]);

                for (int jj = 0; jj < input_blob_count; jj++)
                {
                    const int type_to_pixel = type_to_pixels[jj];
                    const std::vector<float>& mean_vals = means[jj];
                    const std::vector<float>& norm_vals = norms[jj];

                    int pixel_convert_type = ncnn::Mat::PIXEL_BGR;
                    if (type_to_pixel != pixel_convert_type)
                    {
                        pixel_convert_type = pixel_convert_type | (type_to_pixel << ncnn::Mat::PIXEL_CONVERT_SHIFT);
                    }

                    ncnn::Mat in = read_and_resize_image(shapes[jj], listspaths[jj][ii], pixel_convert_type);

                    in.substract_mean_normalize(mean_vals.data(), norm_vals.data());

                    ex.input(input_blobs[jj], in);
                }

                ncnn::Mat in;
                ex.extract(conv_bottom_blobs[i], in);

                ncnn::Mat out;
                ex.extract(conv_top_blobs[i], out);

                ncnn::Layer* layer_int8 = ncnn::create_layer(layer->typeindex);

                ncnn::ParamDict pd;
                get_layer_param(layer, pd);
                pd.set(8, 1); //int8_scale_term
                layer_int8->load_param(pd);

                std::vector<float> sims(search_steps);
                for (int k = 0; k < search_steps; k++)
                {
                    ncnn::Mat new_weight_scale = weight_scale.clone();
                    new_weight_scale[j] = scale_lower + k * scale_step;

                    std::vector<ncnn::Mat> weights;
                    get_layer_weights(layer, weights);
                    weights.push_back(new_weight_scale);
                    weights.push_back(bottom_blob_scale);
                    layer_int8->load_model(ncnn::ModelBinFromMatArray(weights.data()));

                    ncnn::Option opt_int8;
                    opt_int8.use_packing_layout = false;

                    layer_int8->create_pipeline(opt_int8);

                    ncnn::Mat out_int8;
                    layer_int8->forward(in, out_int8, opt_int8);

                    layer_int8->destroy_pipeline(opt_int8);

                    sims[k] = cosine_similarity(out, out_int8);
                }

                delete layer_int8;

                #pragma omp critical
                {
                    for (int k = 0; k < search_steps; k++)
                    {
                        avgsims[k] += sims[k];
                    }
                }
            }

            double max_avgsim = 0.0;
            float new_scale = scale;

            // find the scale with min cosine distance
            for (int k = 0; k < search_steps; k++)
            {
                if (max_avgsim < avgsims[k])
                {
                    max_avgsim = avgsims[k];
                    new_scale = scale_lower + k * scale_step;
                }
            }

            fprintf(stderr, "%s w %d  = %f -> %f\n", layer->name.c_str(), j, scale, new_scale);
            weight_scale[j] = new_scale;
        }

        // search bottom blob scale
        for (int j = 0; j < bottom_blob_scale.w; j++)
        {
            const float scale = bottom_blob_scale[j];
            const float scale_lower = scale * scale_range_lower;
            const float scale_upper = scale * scale_range_upper;
            const float scale_step = (scale_upper - scale_lower) / search_steps;

            std::vector<double> avgsims(search_steps, 0.0);

            #pragma omp parallel for num_threads(quantize_num_threads) schedule(static, 1)
            for (int ii = 0; ii < image_count; ii++)
            {
                if (ii % 100 == 0)
                {
                    fprintf(stderr, "search bottom blob scale %.2f%% [ %d / %d ] for %d / %d of %d / %d\n", ii * 100.f / image_count, ii, image_count, j, bottom_blob_scale.w, i, conv_layer_count);
                }

                ncnn::Extractor ex = create_extractor();

                const int thread_num = ncnn::get_omp_thread_num();
                ex.set_blob_allocator(&blob_allocators[thread_num]);
                ex.set_workspace_allocator(&workspace_allocators[thread_num]);

                for (int jj = 0; jj < input_blob_count; jj++)
                {
                    const int type_to_pixel = type_to_pixels[jj];
                    const std::vector<float>& mean_vals = means[jj];
                    const std::vector<float>& norm_vals = norms[jj];

                    int pixel_convert_type = ncnn::Mat::PIXEL_BGR;
                    if (type_to_pixel != pixel_convert_type)
                    {
                        pixel_convert_type = pixel_convert_type | (type_to_pixel << ncnn::Mat::PIXEL_CONVERT_SHIFT);
                    }

                    ncnn::Mat in = read_and_resize_image(shapes[jj], listspaths[jj][ii], pixel_convert_type);

                    in.substract_mean_normalize(mean_vals.data(), norm_vals.data());

                    ex.input(input_blobs[jj], in);
                }

                ncnn::Mat in;
                ex.extract(conv_bottom_blobs[i], in);

                ncnn::Mat out;
                ex.extract(conv_top_blobs[i], out);

                ncnn::Layer* layer_int8 = ncnn::create_layer(layer->typeindex);

                ncnn::ParamDict pd;
                get_layer_param(layer, pd);
                pd.set(8, 1); //int8_scale_term
                layer_int8->load_param(pd);

                std::vector<float> sims(search_steps);
                for (int k = 0; k < search_steps; k++)
                {
                    ncnn::Mat new_bottom_blob_scale = bottom_blob_scale.clone();
                    new_bottom_blob_scale[j] = scale_lower + k * scale_step;

                    std::vector<ncnn::Mat> weights;
                    get_layer_weights(layer, weights);
                    weights.push_back(weight_scale);
                    weights.push_back(new_bottom_blob_scale);
                    layer_int8->load_model(ncnn::ModelBinFromMatArray(weights.data()));

                    ncnn::Option opt_int8;
                    opt_int8.use_packing_layout = false;

                    layer_int8->create_pipeline(opt_int8);

                    ncnn::Mat out_int8;
                    layer_int8->forward(in, out_int8, opt_int8);

                    layer_int8->destroy_pipeline(opt_int8);

                    sims[k] = cosine_similarity(out, out_int8);
                }

                delete layer_int8;

                #pragma omp critical
                {
                    for (int k = 0; k < search_steps; k++)
                    {
                        avgsims[k] += sims[k];
                    }
                }
            }

            double max_avgsim = 0.0;
            float new_scale = scale;

            // find the scale with min cosine distance
            for (int k = 0; k < search_steps; k++)
            {
                if (max_avgsim < avgsims[k])
                {
                    max_avgsim = avgsims[k];
                    new_scale = scale_lower + k * scale_step;
                }
            }

            fprintf(stderr, "%s b %d  = %f -> %f\n", layer->name.c_str(), j, scale, new_scale);
            bottom_blob_scale[j] = new_scale;
        }

        // update quant info
        QuantBlobStat& stat = quant_blob_stats[i];
        stat.threshold = 127 / bottom_blob_scale[0];
    }

    return 0;
}

static std::vector<std::vector<std::string> > parse_comma_path_list(char* s)
{
    std::vector<std::vector<std::string> > aps;

    char* pch = strtok(s, ",");
    while (pch != NULL)
    {
        FILE* fp = fopen(pch, "rb");
        if (!fp)
        {
            fprintf(stderr, "fopen %s failed\n", pch);
            break;
        }

        std::vector<std::string> paths;

        // one filepath per line
        char line[1024];
        while (!feof(fp))
        {
            char* ss = fgets(line, 1024, fp);
            if (!ss)
                break;

            char filepath[256];
            int nscan = sscanf(line, "%255s", filepath);
            if (nscan != 1)
                continue;

            paths.push_back(std::string(filepath));
        }

        fclose(fp);

        aps.push_back(paths);

        pch = strtok(NULL, ",");
    }

    return aps;
}

static float vstr_to_float(const char vstr[20])
{
    double v = 0.0;

    const char* p = vstr;

    // sign
    bool sign = *p != '-';
    if (*p == '+' || *p == '-')
    {
        p++;
    }

    // digits before decimal point or exponent
    uint64_t v1 = 0;
    while (isdigit(*p))
    {
        v1 = v1 * 10 + (*p - '0');
        p++;
    }

    v = (double)v1;

    // digits after decimal point
    if (*p == '.')
    {
        p++;

        uint64_t pow10 = 1;
        uint64_t v2 = 0;

        while (isdigit(*p))
        {
            v2 = v2 * 10 + (*p - '0');
            pow10 *= 10;
            p++;
        }

        v += v2 / (double)pow10;
    }

    // exponent
    if (*p == 'e' || *p == 'E')
    {
        p++;

        // sign of exponent
        bool fact = *p != '-';
        if (*p == '+' || *p == '-')
        {
            p++;
        }

        // digits of exponent
        uint64_t expon = 0;
        while (isdigit(*p))
        {
            expon = expon * 10 + (*p - '0');
            p++;
        }

        double scale = 1.0;
        while (expon >= 8)
        {
            scale *= 1e8;
            expon -= 8;
        }
        while (expon > 0)
        {
            scale *= 10.0;
            expon -= 1;
        }

        v = fact ? v * scale : v / scale;
    }

    //     fprintf(stderr, "v = %f\n", v);
    return sign ? (float)v : (float)-v;
}

static std::vector<std::vector<float> > parse_comma_float_array_list(char* s)
{
    std::vector<std::vector<float> > aaf;

    char* pch = strtok(s, "[]");
    while (pch != NULL)
    {
        // parse a,b,c
        char vstr[20];
        int nconsumed = 0;
        int nscan = sscanf(pch, "%19[^,]%n", vstr, &nconsumed);
        if (nscan == 1)
        {
            // ok we get array
            pch += nconsumed;

            std::vector<float> af;
            float v = vstr_to_float(vstr);
            af.push_back(v);

            nscan = sscanf(pch, ",%19[^,]%n", vstr, &nconsumed);
            while (nscan == 1)
            {
                pch += nconsumed;

                float v = vstr_to_float(vstr);
                af.push_back(v);

                nscan = sscanf(pch, ",%19[^,]%n", vstr, &nconsumed);
            }

            // array end
            aaf.push_back(af);
        }

        pch = strtok(NULL, "[]");
    }

    return aaf;
}

static std::vector<std::vector<int> > parse_comma_int_array_list(char* s)
{
    std::vector<std::vector<int> > aai;

    char* pch = strtok(s, "[]");
    while (pch != NULL)
    {
        // parse a,b,c
        int v;
        int nconsumed = 0;
        int nscan = sscanf(pch, "%d%n", &v, &nconsumed);
        if (nscan == 1)
        {
            // ok we get array
            pch += nconsumed;

            std::vector<int> ai;
            ai.push_back(v);

            nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            while (nscan == 1)
            {
                pch += nconsumed;

                ai.push_back(v);

                nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            }

            // array end
            aai.push_back(ai);
        }

        pch = strtok(NULL, "[]");
    }

    return aai;
}

static std::vector<int> parse_comma_pixel_type_list(char* s)
{
    std::vector<int> aps;

    char* pch = strtok(s, ",");
    while (pch != NULL)
    {
        // RAW/RGB/BGR/GRAY/RGBA/BGRA
        if (strcmp(pch, "RAW") == 0)
            aps.push_back(-233);
        if (strcmp(pch, "RGB") == 0)
            aps.push_back(ncnn::Mat::PIXEL_RGB);
        if (strcmp(pch, "BGR") == 0)
            aps.push_back(ncnn::Mat::PIXEL_BGR);
        if (strcmp(pch, "GRAY") == 0)
            aps.push_back(ncnn::Mat::PIXEL_GRAY);
        if (strcmp(pch, "RGBA") == 0)
            aps.push_back(ncnn::Mat::PIXEL_RGBA);
        if (strcmp(pch, "BGRA") == 0)
            aps.push_back(ncnn::Mat::PIXEL_BGRA);

        pch = strtok(NULL, ",");
    }

    return aps;
}

static void print_float_array_list(const std::vector<std::vector<float> >& list)
{
    for (size_t i = 0; i < list.size(); i++)
    {
        const std::vector<float>& array = list[i];
        fprintf(stderr, "[");
        for (size_t j = 0; j < array.size(); j++)
        {
            fprintf(stderr, "%f", array[j]);
            if (j != array.size() - 1)
                fprintf(stderr, ",");
        }
        fprintf(stderr, "]");
        if (i != list.size() - 1)
            fprintf(stderr, ",");
    }
}

static void print_int_array_list(const std::vector<std::vector<int> >& list)
{
    for (size_t i = 0; i < list.size(); i++)
    {
        const std::vector<int>& array = list[i];
        fprintf(stderr, "[");
        for (size_t j = 0; j < array.size(); j++)
        {
            fprintf(stderr, "%d", array[j]);
            if (j != array.size() - 1)
                fprintf(stderr, ",");
        }
        fprintf(stderr, "]");
        if (i != list.size() - 1)
            fprintf(stderr, ",");
    }
}

static void print_pixel_type_list(const std::vector<int>& list)
{
    for (size_t i = 0; i < list.size(); i++)
    {
        const int type = list[i];
        if (type == -233)
            fprintf(stderr, "RAW");
        if (type == ncnn::Mat::PIXEL_RGB)
            fprintf(stderr, "RGB");
        if (type == ncnn::Mat::PIXEL_BGR)
            fprintf(stderr, "BGR");
        if (type == ncnn::Mat::PIXEL_GRAY)
            fprintf(stderr, "GRAY");
        if (type == ncnn::Mat::PIXEL_RGBA)
            fprintf(stderr, "RGBA");
        if (type == ncnn::Mat::PIXEL_BGRA)
            fprintf(stderr, "BGRA");
        if (i != list.size() - 1)
            fprintf(stderr, ",");
    }
}

static void show_usage()
{
    fprintf(stderr, "Usage: ncnn2table [ncnnparam] [ncnnbin] [list,...] [ncnntable] [(key=value)...]\n");
    fprintf(stderr, "  mean=[104.0,117.0,123.0],...\n");
    fprintf(stderr, "  norm=[1.0,1.0,1.0],...\n");
    fprintf(stderr, "  shape=[224,224,3],...[w,h,c] or [w,h] **[0,0] will not resize\n");
    fprintf(stderr, "  pixel=RAW/RGB/BGR/GRAY/RGBA/BGRA,...\n");
    fprintf(stderr, "  thread=8\n");
    fprintf(stderr, "  method=kl/aciq/eq\n");
    fprintf(stderr, "Sample usage: ncnn2table squeezenet.param squeezenet.bin imagelist.txt squeezenet.table mean=[104.0,117.0,123.0] norm=[1.0,1.0,1.0] shape=[227,227,3] pixel=BGR method=kl\n");
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        show_usage();
        return -1;
    }

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            show_usage();
            return -1;
        }
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    char* lists = argv[3];
    const char* outtable = argv[4];

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;

    QuantNet net;
    net.opt = opt;
    net.load_param(inparam);
    net.load_model(inbin);

    net.init();

    // load lists
    net.listspaths = parse_comma_path_list(lists);

    std::string method = "kl";

    for (int i = 5; i < argc; i++)
    {
        // key=value
        char* kv = argv[i];

        char* eqs = strchr(kv, '=');
        if (eqs == NULL)
        {
            fprintf(stderr, "unrecognized arg %s\n", kv);
            continue;
        }

        // split k v
        eqs[0] = '\0';
        const char* key = kv;
        char* value = eqs + 1;

        // load mean norm shape
        if (memcmp(key, "mean", 4) == 0)
            net.means = parse_comma_float_array_list(value);
        if (memcmp(key, "norm", 4) == 0)
            net.norms = parse_comma_float_array_list(value);
        if (memcmp(key, "shape", 5) == 0)
            net.shapes = parse_comma_int_array_list(value);
        if (memcmp(key, "pixel", 5) == 0)
            net.type_to_pixels = parse_comma_pixel_type_list(value);
        if (memcmp(key, "thread", 6) == 0)
            net.quantize_num_threads = atoi(value);
        if (memcmp(key, "method", 6) == 0)
            method = std::string(value);
    }

    // sanity check
    const size_t input_blob_count = net.input_blobs.size();
    if (net.listspaths.size() != input_blob_count)
    {
        fprintf(stderr, "expect %d lists, but got %d\n", (int)input_blob_count, (int)net.listspaths.size());
        return -1;
    }
    if (net.means.size() != input_blob_count)
    {
        fprintf(stderr, "expect %d means, but got %d\n", (int)input_blob_count, (int)net.means.size());
        return -1;
    }
    if (net.norms.size() != input_blob_count)
    {
        fprintf(stderr, "expect %d norms, but got %d\n", (int)input_blob_count, (int)net.norms.size());
        return -1;
    }
    if (net.shapes.size() != input_blob_count)
    {
        fprintf(stderr, "expect %d shapes, but got %d\n", (int)input_blob_count, (int)net.shapes.size());
        return -1;
    }
    if (net.type_to_pixels.size() != input_blob_count)
    {
        fprintf(stderr, "expect %d pixels, but got %d\n", (int)input_blob_count, (int)net.type_to_pixels.size());
        return -1;
    }
    if (net.quantize_num_threads < 0)
    {
        fprintf(stderr, "malformed thread %d\n", net.quantize_num_threads);
        return -1;
    }

    // print quantnet config
    {
        fprintf(stderr, "mean = ");
        print_float_array_list(net.means);
        fprintf(stderr, "\n");
        fprintf(stderr, "norm = ");
        print_float_array_list(net.norms);
        fprintf(stderr, "\n");
        fprintf(stderr, "shape = ");
        print_int_array_list(net.shapes);
        fprintf(stderr, "\n");
        fprintf(stderr, "pixel = ");
        print_pixel_type_list(net.type_to_pixels);
        fprintf(stderr, "\n");
        fprintf(stderr, "thread = %d\n", net.quantize_num_threads);
        fprintf(stderr, "method = %s\n", method.c_str());
        fprintf(stderr, "---------------------------------------\n");
    }

    if (method == "kl")
    {
        net.quantize_KL();
    }
    else if (method == "aciq")
    {
        net.quantize_ACIQ();
    }
    else if (method == "eq")
    {
        net.quantize_EQ();
    }
    else
    {
        fprintf(stderr, "not implemented yet !\n");
        fprintf(stderr, "unknown method %s, expect kl / aciq / eq\n", method.c_str());
        return -1;
    }

    net.print_quant_info();

    net.save_table(outtable);

    return 0;
}
