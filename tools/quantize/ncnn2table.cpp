// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// author:BUG1989 (https://github.com/BUG1989/) Long-term support.
// author:JansonZhu (https://github.com/JansonZhu) Implemented the function of entropy calibration.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <vector>

// ncnn public header
#include "benchmark.h"
#include "cpu.h"
#include "net.h"

// ncnn private header
#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/innerproduct.h"

static ncnn::Option g_default_option;
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

// Get the file names from direct path
int parse_images_dir(const std::string& base_path, std::vector<std::string>& file_path)
{
    file_path.clear();

    const cv::String base_path_str(base_path);
    std::vector<cv::String> image_list;

    cv::glob(base_path_str, image_list, true);

    for (size_t i = 0; i < image_list.size(); i++)
    {
        const cv::String& image_path = image_list[i];
        file_path.push_back(image_path);
    }

    return 0;
}

class QuantNet : public ncnn::Net
{
public:
    int get_conv_names();
    int get_conv_bottom_blob_names();
    int get_conv_weight_blob_scales();
    int get_input_names();

public:
    std::vector<std::string> conv_names;
    std::map<std::string, std::string> conv_bottom_blob_names;
    std::map<std::string, std::vector<float> > weight_scales;
    std::vector<std::string> input_names;
};

int QuantNet::get_input_names()
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        const ncnn::Layer* layer = layers[i];

        if (layer->type == "Input")
        {
            for (size_t j = 0; j < layer->tops.size(); j++)
            {
                int blob_index = layer->tops[j];
                std::string name = blobs[blob_index].name;
                input_names.push_back(name);
            }
        }
    }

    return 0;
}

int QuantNet::get_conv_names()
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        const ncnn::Layer* layer = layers[i];

        if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise" || layer->type == "InnerProduct")
        {
            std::string name = layer->name;
            conv_names.push_back(name);
        }
    }

    return 0;
}

int QuantNet::get_conv_bottom_blob_names()
{
    // find conv bottom name or index
    for (size_t i = 0; i < layers.size(); i++)
    {
        const ncnn::Layer* layer = layers[i];

        if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise" || layer->type == "InnerProduct")
        {
            const std::string& name = layer->name;
            const std::string& bottom_blob_name = blobs[layer->bottoms[0]].name;
            conv_bottom_blob_names[name] = bottom_blob_name;
        }
    }

    return 0;
}

int QuantNet::get_conv_weight_blob_scales()
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        const ncnn::Layer* layer = layers[i];

        if (layer->type == "Convolution")
        {
            const ncnn::Convolution* convolution = static_cast<const ncnn::Convolution*>(layer);

            std::string name = layer->name;
            const int weight_data_size_output = convolution->weight_data_size / convolution->num_output;
            std::vector<float> scales;

            // int8 winograd F43 needs weight data to use 6bit quantization
            bool quant_6bit = false;
            int kernel_w = convolution->kernel_w;
            int kernel_h = convolution->kernel_h;
            int dilation_w = convolution->dilation_w;
            int dilation_h = convolution->dilation_h;
            int stride_w = convolution->stride_w;
            int stride_h = convolution->stride_h;

            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                quant_6bit = true;

            for (int n = 0; n < convolution->num_output; n++)
            {
                const ncnn::Mat weight_data_n = convolution->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                const float* data_n = weight_data_n;
                float max_value = std::numeric_limits<float>::min();

                for (int k = 0; k < weight_data_size_output; k++)
                {
                    max_value = std::max(max_value, std::fabs(data_n[k]));
                }

                if (quant_6bit)
                {
                    scales.push_back(31 / max_value);
                }
                else
                {
                    scales.push_back(127 / max_value);
                }
            }

            weight_scales[name] = scales;
        }

        if (layer->type == "ConvolutionDepthWise")
        {
            const ncnn::ConvolutionDepthWise* convolutiondepthwise = static_cast<const ncnn::ConvolutionDepthWise*>(layer);

            std::string name = layer->name;
            const int weight_data_size_output = convolutiondepthwise->weight_data_size / convolutiondepthwise->group;
            std::vector<float> scales;

            for (int n = 0; n < convolutiondepthwise->group; n++)
            {
                const ncnn::Mat weight_data_n = convolutiondepthwise->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                const float* data_n = weight_data_n;
                float max_value = std::numeric_limits<float>::min();

                for (int k = 0; k < weight_data_size_output; k++)
                {
                    max_value = std::max(max_value, std::fabs(data_n[k]));
                }

                scales.push_back(127 / max_value);
            }

            weight_scales[name] = scales;
        }

        if (layer->type == "InnerProduct")
        {
            const ncnn::InnerProduct* innerproduct = static_cast<const ncnn::InnerProduct*>(layer);

            std::string name = layer->name;
            const int weight_data_size_output = innerproduct->weight_data_size / innerproduct->num_output;
            std::vector<float> scales;

            for (int n = 0; n < innerproduct->num_output; n++)
            {
                const ncnn::Mat weight_data_n = innerproduct->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                const float* data_n = weight_data_n;
                float max_value = std::numeric_limits<float>::min();

                for (int k = 0; k < weight_data_size_output; k++)
                    max_value = std::max(max_value, std::fabs(data_n[k]));

                scales.push_back(127 / max_value);
            }

            weight_scales[name] = scales;
        }
    }

    return 0;
}

class QuantizeData
{
public:
    QuantizeData(const std::string& layer_name, const int& num);

    int initial_blob_max(ncnn::Mat data);
    int initial_histogram_interval();
    int initial_histogram_value();

    int normalize_histogram();
    int update_histogram(ncnn::Mat data);

    float compute_kl_divergence(const std::vector<float>& dist_a, const std::vector<float>& dist_b) const;
    int threshold_distribution(const std::vector<float>& distribution, const int target_bin = 128) const;
    float get_data_blob_scale();

public:
    std::string name;

    float max_value;
    int num_bins;
    float histogram_interval;
    std::vector<float> histogram;

    float threshold;
    int threshold_bin;
    float scale;
};

QuantizeData::QuantizeData(const std::string& layer_name, const int& num)
{
    name = layer_name;
    max_value = 0.f;
    num_bins = num;
    histogram_interval = 0.f;
    histogram.resize(num_bins);
    initial_histogram_value();

    threshold = 0.f;
    threshold_bin = 0;
    scale = 1.0f;
}

int QuantizeData::initial_blob_max(ncnn::Mat data)
{
    const int channel_num = data.c;
    const int size = data.w * data.h;

    for (int q = 0; q < channel_num; q++)
    {
        const float* data_n = data.channel(q);
        for (int i = 0; i < size; i++)
        {
            max_value = std::max(max_value, std::fabs(data_n[i]));
        }
    }

    return 0;
}

int QuantizeData::initial_histogram_interval()
{
    histogram_interval = max_value / static_cast<float>(num_bins);

    return 0;
}

int QuantizeData::initial_histogram_value()
{
    for (size_t i = 0; i < histogram.size(); i++)
    {
        histogram[i] = 0.00001f;
    }

    return 0;
}

int QuantizeData::normalize_histogram()
{
    const size_t length = histogram.size();
    float sum = 0;

    for (size_t i = 0; i < length; i++)
        sum += histogram[i];

    for (size_t i = 0; i < length; i++)
        histogram[i] /= sum;

    return 0;
}

int QuantizeData::update_histogram(ncnn::Mat data)
{
    const int channel_num = data.c;
    const int size = data.w * data.h;

    for (int q = 0; q < channel_num; q++)
    {
        const float* data_n = data.channel(q);
        for (int i = 0; i < size; i++)
        {
            if (data_n[i] == 0)
                continue;

            const int index = std::min(static_cast<int>(std::abs(data_n[i]) / histogram_interval), 2047);

            histogram[index]++;
        }
    }

    return 0;
}

float QuantizeData::compute_kl_divergence(const std::vector<float>& dist_a, const std::vector<float>& dist_b) const
{
    const size_t length = dist_a.size();
    assert(dist_b.size() == length);
    float result = 0;

    for (size_t i = 0; i < length; i++)
    {
        if (dist_a[i] != 0)
        {
            if (dist_b[i] == 0)
            {
                result += 1;
            }
            else
            {
                result += dist_a[i] * log(dist_a[i] / dist_b[i]);
            }
        }
    }

    return result;
}

int QuantizeData::threshold_distribution(const std::vector<float>& distribution, const int target_bin) const
{
    int target_threshold = target_bin;
    float min_kl_divergence = 1000;
    const int length = static_cast<int>(distribution.size());

    std::vector<float> quantize_distribution(target_bin);

    float threshold_sum = 0;
    for (int threshold = target_bin; threshold < length; threshold++)
    {
        threshold_sum += distribution[threshold];
    }

    for (int threshold = target_bin; threshold < length; threshold++)
    {
        std::vector<float> t_distribution(distribution.begin(), distribution.begin() + threshold);

        t_distribution[threshold - 1] += threshold_sum;
        threshold_sum -= distribution[threshold];

        // get P
        fill(quantize_distribution.begin(), quantize_distribution.end(), 0.0f);

        const float num_per_bin = static_cast<float>(threshold) / static_cast<float>(target_bin);

        for (int i = 0; i < target_bin; i++)
        {
            const float start = static_cast<float>(i) * num_per_bin;
            const float end = start + num_per_bin;

            const int left_upper = static_cast<int>(ceil(start));
            if (static_cast<float>(left_upper) > start)
            {
                const float left_scale = static_cast<float>(left_upper) - start;
                quantize_distribution[i] += left_scale * distribution[left_upper - 1];
            }

            const int right_lower = static_cast<int>(floor(end));

            if (static_cast<float>(right_lower) < end)
            {
                const float right_scale = end - static_cast<float>(right_lower);
                quantize_distribution[i] += right_scale * distribution[right_lower];
            }

            for (int j = left_upper; j < right_lower; j++)
            {
                quantize_distribution[i] += distribution[j];
            }
        }

        // get Q
        std::vector<float> expand_distribution(threshold, 0);

        for (int i = 0; i < target_bin; i++)
        {
            const float start = static_cast<float>(i) * num_per_bin;
            const float end = start + num_per_bin;

            float count = 0;

            const int left_upper = static_cast<int>(ceil(start));
            float left_scale = 0;
            if (static_cast<float>(left_upper) > start)
            {
                left_scale = static_cast<float>(left_upper) - start;
                if (distribution[left_upper - 1] != 0)
                {
                    count += left_scale;
                }
            }

            const int right_lower = static_cast<int>(floor(end));
            float right_scale = 0;
            if (static_cast<float>(right_lower) < end)
            {
                right_scale = end - static_cast<float>(right_lower);
                if (distribution[right_lower] != 0)
                {
                    count += right_scale;
                }
            }

            for (int j = left_upper; j < right_lower; j++)
            {
                if (distribution[j] != 0)
                {
                    count++;
                }
            }

            const float expand_value = quantize_distribution[i] / count;

            if (static_cast<float>(left_upper) > start)
            {
                if (distribution[left_upper - 1] != 0)
                {
                    expand_distribution[left_upper - 1] += expand_value * left_scale;
                }
            }
            if (static_cast<float>(right_lower) < end)
            {
                if (distribution[right_lower] != 0)
                {
                    expand_distribution[right_lower] += expand_value * right_scale;
                }
            }
            for (int j = left_upper; j < right_lower; j++)
            {
                if (distribution[j] != 0)
                {
                    expand_distribution[j] += expand_value;
                }
            }
        }

        // kl
        const float kl_divergence = compute_kl_divergence(t_distribution, expand_distribution);

        // the best num of bin
        if (kl_divergence < min_kl_divergence)
        {
            min_kl_divergence = kl_divergence;
            target_threshold = threshold;
        }
    }

    return target_threshold;
}

float QuantizeData::get_data_blob_scale()
{
    normalize_histogram();
    threshold_bin = threshold_distribution(histogram);
    threshold = (static_cast<float>(threshold_bin) + 0.5f) * histogram_interval;
    scale = 127 / threshold;
    return scale;
}

struct PreParam
{
    float mean[3];
    float norm[3];
    int width;
    int height;
    bool swapRB;
};

static int post_training_quantize(const std::vector<std::string>& image_list, const std::string& param_path, const std::string& bin_path, const std::string& table_path, struct PreParam& per_param)
{
    size_t size = image_list.size();

    QuantNet net;
    net.opt = g_default_option;

    net.load_param(param_path.c_str());
    net.load_model(bin_path.c_str());

    float mean_vals[3];
    float norm_vals[3];

    int width = per_param.width;
    int height = per_param.height;
    bool swapRB = per_param.swapRB;

    mean_vals[0] = per_param.mean[0];
    mean_vals[1] = per_param.mean[1];
    mean_vals[2] = per_param.mean[2];

    norm_vals[0] = per_param.norm[0];
    norm_vals[1] = per_param.norm[1];
    norm_vals[2] = per_param.norm[2];

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    net.get_input_names();
    net.get_conv_names();
    net.get_conv_bottom_blob_names();
    net.get_conv_weight_blob_scales();

    if (net.input_names.empty())
    {
        fprintf(stderr, "not found [Input] Layer, Check your ncnn.param \n");
        return -1;
    }

    FILE* fp = fopen(table_path.c_str(), "w");

    // save quantization scale of weight
    printf("====> Quantize the parameters.\n");
    for (size_t i = 0; i < net.conv_names.size(); i++)
    {
        std::string layer_name = net.conv_names[i];
        std::string blob_name = net.conv_bottom_blob_names[layer_name];
        std::vector<float> weight_scale_n = net.weight_scales[layer_name];

        fprintf(fp, "%s_param_0 ", layer_name.c_str());
        for (size_t j = 0; j < weight_scale_n.size(); j++)
        {
            fprintf(fp, "%f ", weight_scale_n[j]);
        }
        fprintf(fp, "\n");
    }

    // initial quantization data
    std::vector<QuantizeData> quantize_datas;

    for (size_t i = 0; i < net.conv_names.size(); i++)
    {
        std::string layer_name = net.conv_names[i];

        QuantizeData quantize_data(layer_name, 2048);
        quantize_datas.push_back(quantize_data);
    }

    // step 1 count the max value
    printf("====> Quantize the activation.\n");
    printf("    ====> step 1 : find the max value.\n");

    for (size_t i = 0; i < image_list.size(); i++)
    {
        std::string img_name = image_list[i];

        if ((i + 1) % 100 == 0)
        {
            fprintf(stderr, "          %d/%d\n", static_cast<int>(i + 1), static_cast<int>(size));
        }

#if OpenCV_VERSION_MAJOR > 2
        cv::Mat bgr = cv::imread(img_name, cv::IMREAD_COLOR);
#else
        cv::Mat bgr = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);
#endif
        if (bgr.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
            return -1;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, swapRB ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, width, height);
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.input(net.input_names[0].c_str(), in);

        for (size_t j = 0; j < net.conv_names.size(); j++)
        {
            std::string layer_name = net.conv_names[j];
            std::string blob_name = net.conv_bottom_blob_names[layer_name];

            ncnn::Mat out;
            ex.extract(blob_name.c_str(), out);

            for (size_t k = 0; k < quantize_datas.size(); k++)
            {
                if (quantize_datas[k].name == layer_name)
                {
                    quantize_datas[k].initial_blob_max(out);
                    break;
                }
            }
        }
    }

    // step 2 histogram_interval
    printf("    ====> step 2 : generate the histogram_interval.\n");
    for (size_t i = 0; i < net.conv_names.size(); i++)
    {
        std::string layer_name = net.conv_names[i];

        for (size_t k = 0; k < quantize_datas.size(); k++)
        {
            if (quantize_datas[k].name == layer_name)
            {
                quantize_datas[k].initial_histogram_interval();

                fprintf(stderr, "%-20s : max = %-15f interval = %-10f\n", quantize_datas[k].name.c_str(), quantize_datas[k].max_value, quantize_datas[k].histogram_interval);
                break;
            }
        }
    }

    // step 3 histogram
    printf("    ====> step 3 : generate the histogram.\n");
    for (size_t i = 0; i < image_list.size(); i++)
    {
        std::string img_name = image_list[i];

        if ((i + 1) % 100 == 0)
            fprintf(stderr, "          %d/%d\n", (int)(i + 1), (int)size);
#if OpenCV_VERSION_MAJOR > 2
        cv::Mat bgr = cv::imread(img_name, cv::IMREAD_COLOR);
#else
        cv::Mat bgr = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);
#endif
        if (bgr.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
            return -1;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, swapRB ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, width, height);
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = net.create_extractor();
        ex.input(net.input_names[0].c_str(), in);

        for (size_t j = 0; j < net.conv_names.size(); j++)
        {
            std::string layer_name = net.conv_names[j];
            std::string blob_name = net.conv_bottom_blob_names[layer_name];

            ncnn::Mat out;
            ex.extract(blob_name.c_str(), out);

            for (size_t k = 0; k < quantize_datas.size(); k++)
            {
                if (quantize_datas[k].name == layer_name)
                {
                    quantize_datas[k].update_histogram(out);
                    break;
                }
            }
        }
    }

    // step4 kld
    printf("    ====> step 4 : using kld to find the best threshold value.\n");
    for (size_t i = 0; i < net.conv_names.size(); i++)
    {
        std::string layer_name = net.conv_names[i];
        std::string blob_name = net.conv_bottom_blob_names[layer_name];
        fprintf(stderr, "%-20s ", layer_name.c_str());

        for (size_t k = 0; k < quantize_datas.size(); k++)
        {
            if (quantize_datas[k].name == layer_name)
            {
                quantize_datas[k].get_data_blob_scale();
                fprintf(stderr, "bin : %-8d threshold : %-15f interval : %-10f scale : %-10f\n",
                        quantize_datas[k].threshold_bin,
                        quantize_datas[k].threshold,
                        quantize_datas[k].histogram_interval,
                        quantize_datas[k].scale);

                fprintf(fp, "%s %f\n", layer_name.c_str(), quantize_datas[k].scale);

                break;
            }
        }
    }

    fclose(fp);
    printf("====> Save the calibration table done.\n");

    return 0;
}

// usage
void showUsage()
{
    std::cout << "example: ./ncnn2table --param=squeezenet-fp32.param --bin=squeezenet-fp32.bin --images=images/ --output=squeezenet.table --mean=104.0,117.0,123.0 --norm=1.0,1.0,1.0 --size=224,224 --swapRB --thread=2" << std::endl;
}

static int find_all_value_in_string(const std::string& values_string, std::vector<float>& value)
{
    std::vector<int> masks_pos;

    for (size_t i = 0; i < values_string.size(); i++)
    {
        if (',' == values_string[i])
        {
            masks_pos.push_back(static_cast<int>(i));
        }
    }

    // check
    if (masks_pos.empty())
    {
        fprintf(stderr, "ERROR: Cannot find any ',' in string, please check.\n");
        return -1;
    }

    if (2 != masks_pos.size())
    {
        fprintf(stderr, "ERROR: Char ',' in fist of string, please check.\n");
        return -1;
    }

    if (masks_pos.front() == 0)
    {
        fprintf(stderr, "ERROR: Char ',' in fist of string, please check.\n");
        return -1;
    }

    if (masks_pos.back() == 0)
    {
        fprintf(stderr, "ERROR: Char ',' in last of string, please check.\n");
        return -1;
    }

    for (size_t i = 0; i < masks_pos.size(); i++)
    {
        if (i > 0)
        {
            if (!(masks_pos[i] - masks_pos[i - 1] > 1))
            {
                fprintf(stderr, "ERROR: Neighbouring char ',' was found.\n");
                return -1;
            }
        }
    }

    const cv::String ch0_val_str = values_string.substr(0, masks_pos[0]);
    const cv::String ch1_val_str = values_string.substr(masks_pos[0] + 1, masks_pos[1] - masks_pos[0] - 1);
    const cv::String ch2_val_str = values_string.substr(masks_pos[1] + 1, values_string.size() - masks_pos[1] - 1);

    value.push_back(static_cast<float>(std::atof(std::string(ch0_val_str).c_str())));
    value.push_back(static_cast<float>(std::atof(std::string(ch1_val_str).c_str())));
    value.push_back(static_cast<float>(std::atof(std::string(ch2_val_str).c_str())));

    return 0;
}

#if CV_MAJOR_VERSION < 3
class NcnnQuantCommandLineParser : public cv::CommandLineParser
{
public:
    NcnnQuantCommandLineParser(int argc, const char* const argv[], const char* key_map)
        : cv::CommandLineParser(argc, argv, key_map)
    {
    }
    bool has(const std::string& keys)
    {
        return cv::CommandLineParser::has(keys);
    }
    void printMessage()
    {
        cv::CommandLineParser::printParams();
    }
};
#endif

int main(int argc, char** argv)
{
    std::cout << "--- ncnn post training quantization tool --- " << __TIME__ << " " << __DATE__ << std::endl;

    const char* key_map = "{help h usage ? |   | print this message }"
                          "{param p        |   | path to ncnn.param file }"
                          "{bin b          |   | path to ncnn.bin file }"
                          "{images i       |   | path to calibration images folder }"
                          "{output o       |   | path to output calibration table file }"
                          "{mean m         |   | value of mean (mean value, default is 104.0,117.0,123.0) }"
                          "{norm n         |   | value of normalize (scale value, default is 1.0,1.0,1.0) }"
                          "{size s         |   | the size of input image(using the resize the original image,default is w=224,h=224) }"
                          "{swapRB c       |   | flag which indicates that swap first and last channels in 3-channel image is necessary }"
                          "{thread t       | 4 | count of processing threads }";

#if CV_MAJOR_VERSION < 3
    NcnnQuantCommandLineParser parser(argc, argv, key_map);
#else
    cv::CommandLineParser parser(argc, argv, key_map);
#endif

    if (parser.has("help"))
    {
        parser.printMessage();
        showUsage();
        return 0;
    }

    if (!parser.has("param") || !parser.has("bin") || !parser.has("images") || !parser.has("output") || !parser.has("mean") || !parser.has("norm"))
    {
        std::cout << "Inputs is does not include all needed param, pleas check..." << std::endl;
        parser.printMessage();
        showUsage();
        return 0;
    }

    const std::string image_folder_path = parser.get<cv::String>("images");
    const std::string ncnn_param_file_path = parser.get<cv::String>("param");
    const std::string ncnn_bin_file_path = parser.get<cv::String>("bin");
    const std::string saved_table_file_path = parser.get<cv::String>("output");

    // check the input param
    if (image_folder_path.empty() || ncnn_param_file_path.empty() || ncnn_bin_file_path.empty() || saved_table_file_path.empty())
    {
        fprintf(stderr, "One or more path may be empty, please check and try again.\n");
        return 0;
    }

    const int num_threads = parser.get<int>("thread");

    struct PreParam pre_param;
    pre_param.mean[0] = 104.f;
    pre_param.mean[1] = 117.f;
    pre_param.mean[2] = 103.f;
    pre_param.norm[0] = 1.f;
    pre_param.norm[1] = 1.f;
    pre_param.norm[2] = 1.f;
    pre_param.width = 224;
    pre_param.height = 224;
    pre_param.swapRB = false;

    if (parser.has("mean"))
    {
        const std::string mean_str = parser.get<std::string>("mean");

        std::vector<float> mean_values;
        const int ret = find_all_value_in_string(mean_str, mean_values);
        if (0 != ret && 3 != mean_values.size())
        {
            fprintf(stderr, "ERROR: Searching mean value from --mean was failed.\n");

            return -1;
        }

        pre_param.mean[0] = mean_values[0];
        pre_param.mean[1] = mean_values[1];
        pre_param.mean[2] = mean_values[2];
    }

    if (parser.has("norm"))
    {
        const std::string norm_str = parser.get<std::string>("norm");

        std::vector<float> norm_values;
        const int ret = find_all_value_in_string(norm_str, norm_values);
        if (0 != ret && 3 != norm_values.size())
        {
            fprintf(stderr, "ERROR: Searching mean value from --mean was failed, please check --mean param.\n");

            return -1;
        }

        pre_param.norm[0] = norm_values[0];
        pre_param.norm[1] = norm_values[1];
        pre_param.norm[2] = norm_values[2];
    }

    if (parser.has("size"))
    {
        cv::String size_str = parser.get<std::string>("size");

        size_t sep_pos = size_str.find_first_of(',');

        if (cv::String::npos != sep_pos && sep_pos < size_str.size())
        {
            cv::String width_value_str;
            cv::String height_value_str;

            width_value_str = size_str.substr(0, sep_pos);
            height_value_str = size_str.substr(sep_pos + 1, size_str.size() - sep_pos - 1);

            pre_param.width = static_cast<int>(std::atoi(std::string(width_value_str).c_str()));
            pre_param.height = static_cast<int>(std::atoi(std::string(height_value_str).c_str()));
        }
        else
        {
            fprintf(stderr, "ERROR: Searching size value from --size was failed, please check --size param.\n");

            return -1;
        }
    }

    if (parser.has("swapRB"))
    {
        pre_param.swapRB = true;
    }

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

    // default option
    g_default_option.lightmode = true;
    g_default_option.num_threads = num_threads;
    g_default_option.blob_allocator = &g_blob_pool_allocator;
    g_default_option.workspace_allocator = &g_workspace_pool_allocator;

    g_default_option.use_winograd_convolution = true;
    g_default_option.use_sgemm_convolution = true;
    g_default_option.use_int8_inference = true;
    g_default_option.use_fp16_packed = true;
    g_default_option.use_fp16_storage = true;
    g_default_option.use_fp16_arithmetic = true;
    g_default_option.use_int8_storage = true;
    g_default_option.use_int8_arithmetic = true;

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    std::vector<std::string> image_file_path_list;

    // parse the image file.
    parse_images_dir(image_folder_path, image_file_path_list);

    // get the calibration table file, and save it.
    const int ret = post_training_quantize(image_file_path_list, ncnn_param_file_path, ncnn_bin_file_path, saved_table_file_path, pre_param);
    if (!ret)
    {
        fprintf(stderr, "\nNCNN Int8 Calibration table create success, best wish for your INT8 inference has a low accuracy loss...\\(^0^)/...233...\n");
    }

    return 0;
}
