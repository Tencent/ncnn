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

#include <algorithm>
#include <assert.h>
#include <cctype>
#include <deque>
#include <fstream>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

#define OUTPUT_LAYER_MAP 0 //enable this to generate darknet style layer output

void file_error(const char* s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(EXIT_FAILURE);
}

void fread_or_error(void* buffer, size_t size, size_t count, FILE* fp, const char* s)
{
    if (count != fread(buffer, size, count, fp))
    {
        fprintf(stderr, "Couldn't read from file: %s\n", s);
        fclose(fp);
        assert(0);
        exit(EXIT_FAILURE);
    }
}

void error(const char* s)
{
    perror(s);
    assert(0);
    exit(EXIT_FAILURE);
}

typedef struct Section
{
    std::string name;
    int line_number = -1;
    int original_layer_count;

    std::unordered_map<std::string, std::string> options;
    int w = 416, h = 416, c = 3, inputs = 256;
    int out_w, out_h, out_c;
    int batch_normalize = 0, filters = 1, size = 1, groups = 1, stride = 1, padding = -1, pad = 0, dilation = 1;
    std::string activation;
    int from, reverse;
    std::vector<int> layers, mask, anchors;
    int group_id = -1;
    int classes = 0, num = 0;
    float ignore_thresh = 0.45f, scale_x_y = 1.f;

    std::vector<float> weights, bias, scales, rolling_mean, rolling_variance;

    std::string layer_type, layer_name;
    std::vector<std::string> input_blobs, output_blobs;
    std::vector<std::string> real_output_blobs;
    std::vector<std::string> param;
} Section;

static inline std::string& trim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(),
    s.end());
    return s;
}

typedef enum FIELD_TYPE
{
    INT,
    FLOAT,
    IARRAY,
    FARRAY,
    STRING,
    UNSUPPORTED
} FIELD_TYPE;

typedef struct Section_Field
{
    const char* name;
    FIELD_TYPE type;
    size_t offset;
} Section_Field;

#define FIELD_OFFSET(c) ((size_t) & (((Section*)0)->c))

int yolo_layer_count = 0;

std::vector<std::string> split(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

template<typename... Args>
std::string format(const char* fmt, Args... args)
{
    size_t size = snprintf(nullptr, 0, fmt, args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt, args...);
    return buf;
}

void update_field(Section* section, std::string key, std::string value)
{
    static const Section_Field fields[] = {
        //net
        {"width", INT, FIELD_OFFSET(w)},
        {"height", INT, FIELD_OFFSET(h)},
        {"channels", INT, FIELD_OFFSET(c)},
        {"inputs", INT, FIELD_OFFSET(inputs)},
        //convolutional, upsample, maxpool
        {"batch_normalize", INT, FIELD_OFFSET(batch_normalize)},
        {"filters", INT, FIELD_OFFSET(filters)},
        {"size", INT, FIELD_OFFSET(size)},
        {"groups", INT, FIELD_OFFSET(groups)},
        {"stride", INT, FIELD_OFFSET(stride)},
        {"padding", INT, FIELD_OFFSET(padding)},
        {"pad", INT, FIELD_OFFSET(pad)},
        {"dilation", INT, FIELD_OFFSET(dilation)},
        {"activation", STRING, FIELD_OFFSET(activation)},
        //shortcut
        {"from", INT, FIELD_OFFSET(from)},
        {"reverse", INT, FIELD_OFFSET(reverse)},
        //route
        {"layers", IARRAY, FIELD_OFFSET(layers)},
        {"group_id", INT, FIELD_OFFSET(group_id)},
        //yolo
        {"mask", IARRAY, FIELD_OFFSET(mask)},
        {"anchors", IARRAY, FIELD_OFFSET(anchors)},
        {"classes", INT, FIELD_OFFSET(classes)},
        {"num", INT, FIELD_OFFSET(num)},
        {"ignore_thresh", FLOAT, FIELD_OFFSET(ignore_thresh)},
        {"scale_x_y", FLOAT, FIELD_OFFSET(scale_x_y)},
    };

    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++)
    {
        auto f = fields[i];
        if (key != f.name)
            continue;
        char* addr = ((char*)section) + f.offset;
        switch (f.type)
        {
        case INT:
            *(int*)(addr) = std::stoi(value);
            return;

        case FLOAT:
            *(float*)(addr) = std::stof(value);
            return;

        case IARRAY:
            for (auto v : split(value, ','))
                reinterpret_cast<std::vector<int>*>(addr)->push_back(std::stoi(v));
            return;

        case FARRAY:
            for (auto v : split(value, ','))
                reinterpret_cast<std::vector<float>*>(addr)->push_back(std::stof(v));
            return;

        case STRING:
            *reinterpret_cast<std::string*>(addr) = value;
            return;

        case UNSUPPORTED:
            printf("unsupported option: %s\n", key.c_str());
            exit(EXIT_FAILURE);
        }
    }
}

void load_cfg(const char* filename, std::deque<Section*>& dnet)
{
    std::string line;
    std::ifstream icfg(filename, std::ifstream::in);
    if (!icfg.good())
    {
        fprintf(stderr, "Couldn't cfg open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    Section* section = NULL;
    size_t pos;
    int section_count = 0, line_count = 0;
    while (!icfg.eof())
    {
        line_count++;
        std::getline(icfg, line);
        trim(line);
        if (line.length() == 0 || line.at(0) == '#')
            continue;
        if (line.at(0) == '[' && line.at(line.length() - 1) == ']')
        {
            line = line.substr(1, line.length() - 2);
            section = new Section;
            section->name = line;
            section->line_number = line_count;
            section->original_layer_count = section_count++;
            dnet.push_back(section);
        }
        else if ((pos = line.find_first_of('=')) != std::string::npos)
        {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1, line.length() - 1);
            section->options[trim(key)] = trim(value);
            update_field(section, key, value);
        }
    }

    icfg.close();
}

Section* get_original_section(std::deque<Section*>& dnet, int count, int offset)
{
    if (offset >= 0)
        count = offset + 1;
    else
        count += offset;
    for (auto s : dnet)
        if (s->original_layer_count == count)
            return s;
    return dnet[0];
}

template<typename T>
std::string array_to_float_string(std::vector<T> vec)
{
    std::string ret;
    for (size_t i = 0; i < vec.size(); i++)
        ret.append(format(",%f", (float)vec[i]));
    return ret;
}

Section* get_section_by_output_blob(std::deque<Section*>& dnet, std::string blob)
{
    for (auto s : dnet)
        for (auto b : s->output_blobs)
            if (b == blob)
                return s;
    return NULL;
}

std::vector<Section*> get_sections_by_input_blob(std::deque<Section*>& dnet, std::string blob)
{
    std::vector<Section*> ret;
    for (auto s : dnet)
        for (auto b : s->input_blobs)
            if (b == blob)
                ret.push_back(s);
    return ret;
}

void addActivationLayer(Section* s, std::deque<Section*>::iterator& it, std::deque<Section*>& dnet)
{
    Section* act = new Section;

    if (s->activation == "relu")
    {
        act->layer_type = "ReLU";
        act->param.push_back("0=0");
    }
    else if (s->activation == "leaky")
    {
        act->layer_type = "ReLU";
        act->param.push_back("0=0.1");
    }
    else if (s->activation == "mish")
        act->layer_type = "Mish";
    else if (s->activation == "logistic")
        act->layer_type = "Sigmoid";
    else if (s->activation == "swish")
        act->layer_type = "Swish";

    if (s->batch_normalize)
        act->layer_name = s->layer_name + "_bn";
    else
        act->layer_name = s->layer_name;
    act->h = s->out_h;
    act->w = s->out_w;
    act->c = s->out_c;
    act->out_h = s->out_h;
    act->out_w = s->out_w;
    act->out_c = s->out_c;
    act->layer_name += "_" + s->activation;
    act->input_blobs = s->real_output_blobs;
    act->output_blobs.push_back(act->layer_name);

    s->real_output_blobs = act->real_output_blobs = act->output_blobs;
    it = dnet.insert(it + 1, act);
}

void parse_cfg(std::deque<Section*>& dnet, int merge_output)
{
    int input_w = 416, input_h = 416;
    int yolo_count = 0;
    std::vector<Section*> yolo_layers;

#if OUTPUT_LAYER_MAP
    printf("   layer   filters  size/strd(dil)      input                output\n");
#endif
    for (auto it = dnet.begin(); it != dnet.end(); it++)
    {
        auto s = *it;
        if (s->line_number < 0)
            continue;

        auto p = get_original_section(dnet, s->original_layer_count, -1);

#if OUTPUT_LAYER_MAP
        if (s->original_layer_count > 0)
            printf("%4d ", s->original_layer_count - 1);
#endif

        s->layer_name = format("%d_%d", s->original_layer_count - 1, s->line_number);
        s->input_blobs = p->real_output_blobs;
        s->output_blobs.push_back(s->layer_name);
        s->real_output_blobs = s->output_blobs;

        if (s->name == "net")
        {
            s->out_h = s->h;
            s->out_w = s->w;
            s->out_c = s->c;
            input_h = s->h;
            input_w = s->w;

            s->layer_type = "Input";
            s->layer_name = "data";
            s->input_blobs.clear();
            s->output_blobs.clear();
            s->output_blobs.push_back("data");
            s->real_output_blobs = s->output_blobs;
            s->param.push_back(format("0=%d", s->w));
            s->param.push_back(format("1=%d", s->h));
            s->param.push_back(format("2=%d", s->c));
        }
        else if (s->name == "convolutional")
        {
            if (s->padding == -1)
                s->padding = 0;
            s->h = p->out_h;
            s->w = p->out_w;
            s->c = p->out_c;
            s->out_h = s->h / s->stride;
            s->out_w = s->w / s->stride;
            s->out_c = s->filters;

#if OUTPUT_LAYER_MAP
            if (s->groups == 1)
                printf("conv %5d      %2d x%2d/%2d   ", s->filters, s->size, s->size, s->stride);
            else
                printf("conv %5d/%4d %2d x%2d/%2d   ", s->filters, s->groups, s->size, s->size, s->stride);
            printf("%4d x%4d x%4d -> %4d x%4d x%4d\n", s->h, s->w, s->c, s->out_h, s->out_w, s->out_c);
#endif

            if (s->groups == 1)
                s->layer_type = "Convolution";
            else
                s->layer_type = "ConvolutionDepthWise";
            s->param.push_back(format("0=%d", s->filters));                        //num_output
            s->param.push_back(format("1=%d", s->size));                           //kernel_w
            s->param.push_back(format("2=%d", s->dilation));                       //dilation_w
            s->param.push_back(format("3=%d", s->stride));                         //stride_w
            s->param.push_back(format("4=%d", s->pad ? s->size / 2 : s->padding)); //pad_left

            if (s->batch_normalize)
            {
                s->param.push_back("5=0"); //bias_term

                Section* bn = new Section;
                bn->layer_type = "BatchNorm";
                bn->layer_name = s->layer_name + "_bn";
                bn->h = s->out_h;
                bn->w = s->out_w;
                bn->c = s->out_c;
                bn->out_h = s->out_h;
                bn->out_w = s->out_w;
                bn->out_c = s->out_c;
                bn->input_blobs = s->real_output_blobs;
                bn->output_blobs.push_back(bn->layer_name);
                bn->param.push_back(format("0=%d", s->filters)); //channels
                bn->param.push_back("1=.00001");                 //eps

                s->real_output_blobs = bn->real_output_blobs = bn->output_blobs;
                it = dnet.insert(it + 1, bn);
            }
            else
            {
                s->param.push_back("5=1"); //bias_term
            }
            s->param.push_back(format("6=%d", s->c * s->size * s->size * s->filters / s->groups)); //weight_data_size

            if (s->groups > 1)
                s->param.push_back(format("7=%d", s->groups)); //stride_w

            if (s->activation.size() > 0)
            {
                if (s->activation == "relu" || s->activation == "leaky" || s->activation == "mish" || s->activation == "logistic" || s->activation == "swish")
                {
                    addActivationLayer(s, it, dnet);
                }
                else if (s->activation != "linear")
                    error(format("Unsupported convolutional activation type: %s", s->activation.c_str()).c_str());
            }
        }
        else if (s->name == "shortcut")
        {
            auto q = get_original_section(dnet, s->original_layer_count, s->from);
            if (p->out_h != q->out_h || p->out_w != q->out_w)
                error("shortcut dim not match");

            s->h = p->out_h;
            s->w = p->out_w;
            s->c = p->out_c;
            s->out_h = s->h;
            s->out_w = s->w;
            s->out_c = p->out_c;

#if OUTPUT_LAYER_MAP
            printf("Shortcut Layer: %d, ", q->original_layer_count - 1);
            printf("outputs: %4d x%4d x%4d\n", s->out_h, s->out_w, s->out_c);
            if (p->out_c != q->out_c)
                printf("(%4d x%4d x%4d) + (%4d x%4d x%4d)\n", p->out_h, p->out_w, p->out_c,
                       q->out_h, q->out_w, q->out_c);
#endif

            if (s->activation.size() > 0)
            {
                if (s->activation == "relu" || s->activation == "leaky" || s->activation == "mish" || s->activation == "logistic" || s->activation == "swish")
                {
                    addActivationLayer(s, it, dnet);
                }
                else if (s->activation != "linear")
                    error(format("Unsupported convolutional activation type: %s", s->activation.c_str()).c_str());
            }

            s->layer_type = "Eltwise";
            s->input_blobs.clear();
            s->input_blobs.push_back(p->real_output_blobs[0]);
            s->input_blobs.push_back(q->real_output_blobs[0]);

            s->param.push_back("0=1"); //op_type=Operation_SUM
        }
        else if (s->name == "maxpool")
        {
            if (s->padding == -1)
                s->padding = s->stride * int((s->size - 1) / 2);
            s->h = p->out_h;
            s->w = p->out_w;
            s->c = p->out_c;
            s->out_h = (s->h + s->padding - s->size) / s->stride + 1;
            s->out_w = (s->w + s->padding - s->size) / s->stride + 1;
            s->out_c = s->c;

#if OUTPUT_LAYER_MAP
            printf("max             %2d x%2d/%2d   ", s->size, s->size, s->stride);
            printf("%4d x%4d x%4d -> %4d x%4d x%4d\n", s->h, s->w, s->c, s->out_h, s->out_w, s->out_c);
#endif

            s->layer_type = "Pooling";
            s->param.push_back("0=0");                       //pooling_type=PoolMethod_MAX
            s->param.push_back(format("1=%d", s->size));     //kernel_w
            s->param.push_back(format("2=%d", s->stride));   //stride_w
            s->param.push_back("5=1");                       //pad_mode=SAME_UPPER
            s->param.push_back(format("3=%d", s->padding));  //pad_left
            s->param.push_back(format("13=%d", s->padding)); //pad_top
            s->param.push_back(format("14=%d", s->padding)); //pad_right
            s->param.push_back(format("15=%d", s->padding)); //pad_bottom
        }
        else if (s->name == "avgpool")
        {
            if (s->padding == -1)
                s->padding = s->size - 1;
            s->h = p->out_h;
            s->w = p->out_w;
            s->c = p->out_c;
            s->out_h = 1;
            s->out_w = s->out_h;
            s->out_c = s->c;

#if OUTPUT_LAYER_MAP
            printf("avg                         %4d x%4d x%4d ->   %4d\n", s->h, s->w, s->c, s->out_c);
#endif

            s->layer_type = "Pooling";
            s->param.push_back("0=1"); //pooling_type=PoolMethod_AVE
            s->param.push_back("4=1"); //global_pooling

            Section* r = new Section;
            r->layer_type = "Reshape";
            r->layer_name = s->layer_name + "_reshape";
            r->h = s->out_h;
            r->w = s->out_w;
            r->c = s->out_c;
            r->out_h = 1;
            r->out_w = 1;
            r->out_c = r->h * r->w * r->c;
            r->input_blobs.push_back(s->output_blobs[0]);
            r->output_blobs.push_back(r->layer_name);
            r->param.push_back("0=1");                    //w
            r->param.push_back("1=1");                    //h
            r->param.push_back(format("2=%d", r->out_c)); //c

            s->real_output_blobs.clear();
            s->real_output_blobs.push_back(r->layer_name);

            it = dnet.insert(it + 1, r);
        }
        else if (s->name == "scale_channels")
        {
            auto q = get_original_section(dnet, s->original_layer_count, s->from);
            if (p->out_c != q->out_c)
                error("scale channels not match");

            s->h = q->out_h;
            s->w = q->out_w;
            s->c = q->out_c;
            s->out_h = s->h;
            s->out_w = s->w;
            s->out_c = q->out_c;

#if OUTPUT_LAYER_MAP
            printf("scale Layer: %d\n", q->original_layer_count - 1);
#endif

            if (s->activation.size() > 0 && s->activation != "linear")
                error(format("Unsupported scale_channels activation type: %s", s->activation.c_str()).c_str());

            s->layer_type = "BinaryOp";
            s->input_blobs.clear();
            s->input_blobs.push_back(q->real_output_blobs[0]);
            s->input_blobs.push_back(p->real_output_blobs[0]);
            s->param.push_back("0=2"); //op_type=Operation_MUL
        }
        else if (s->name == "route")
        {
#if OUTPUT_LAYER_MAP
            printf("route  ");
#endif
            s->out_c = 0;
            s->input_blobs.clear();
            for (int l : s->layers)
            {
                auto q = get_original_section(dnet, s->original_layer_count, l);
#if OUTPUT_LAYER_MAP
                printf("%d ", q->original_layer_count - 1);
#endif
                s->out_h = q->out_h;
                s->out_w = q->out_w;
                s->out_c += q->out_c;

                for (auto blob : q->real_output_blobs)
                    s->input_blobs.push_back(blob);
            }
            if (s->input_blobs.size() == 1)
            {
                if (s->groups <= 1 || s->group_id == -1)
                    s->layer_type = "Noop";
                else
                {
                    s->out_c /= s->groups;
#if OUTPUT_LAYER_MAP
                    printf("%31d/%d -> %4d x%4d x%4d", 1, s->groups, s->out_w, s->out_h, s->out_c);
#endif

                    s->layer_type = "Crop";
                    s->param.push_back(format("2=%d", s->out_c * s->group_id));
                    s->param.push_back(format("3=%d", s->out_w));
                    s->param.push_back(format("4=%d", s->out_h));
                    s->param.push_back(format("5=%d", s->out_c));
                }
            }
            else
            {
                s->layer_type = "Concat";
            }
#if OUTPUT_LAYER_MAP
            printf("\n");
#endif
        }
        else if (s->name == "upsample")
        {
            s->h = p->out_h;
            s->w = p->out_w;
            s->c = p->out_c;
            s->out_h = s->h * s->stride;
            s->out_w = s->w * s->stride;
            s->out_c = s->c;

#if OUTPUT_LAYER_MAP
            printf("upsample               %2dx  ", s->stride);
            printf("%4d x%4d x%4d -> %4d x%4d x%4d\n", s->h, s->w, s->c, s->out_h, s->out_w, s->out_c);
#endif
            s->layer_type = "Interp";
            s->param.push_back("0=1");   //resize_type=nearest
            s->param.push_back("1=2.f"); //height_scale
            s->param.push_back("2=2.f"); //width_scale
        }
        else if (s->name == "yolo")
        {
#if OUTPUT_LAYER_MAP
            printf("yolo%d\n", yolo_count);
#endif

            if (s->ignore_thresh > 0.25)
            {
                fprintf(stderr, "WARNING: The ignore_thresh=%f of yolo%d is too high. "
                        "An alternative value 0.25 is written instead.\n",
                        s->ignore_thresh, yolo_count);
                s->ignore_thresh = 0.25;
            }

            s->layer_type = "Yolov3DetectionOutput";
            s->layer_name = format("yolo%d", yolo_count++);
            s->output_blobs[0] = s->layer_name;
            s->h = p->out_h;
            s->w = p->out_w;
            s->c = p->out_c;
            s->out_h = s->h;
            s->out_w = s->w;
            s->out_c = s->c * (int)s->mask.size();
            s->param.push_back(format("0=%d", s->classes));                                                             //num_class
            s->param.push_back(format("1=%d", s->mask.size()));                                                         //num_box
            s->param.push_back(format("2=%f", s->ignore_thresh));                                                       //confidence_threshold
            s->param.push_back(format("-23304=%d%s", s->anchors.size(), array_to_float_string(s->anchors).c_str()));    //biases
            s->param.push_back(format("-23305=%d%s", s->mask.size(), array_to_float_string(s->mask).c_str()));          //mask
            s->param.push_back(format("-23306=2,%f,%f", input_w * s->scale_x_y / s->w, input_h * s->scale_x_y / s->h)); //biases_index

            yolo_layer_count++;
            yolo_layers.push_back(s);
        }
        else if (s->name == "dropout")
        {
#if OUTPUT_LAYER_MAP
            printf("dropout\n");
#endif
            s->h = p->out_h;
            s->w = p->out_w;
            s->c = p->out_c;
            s->out_h = s->h;
            s->out_w = s->w;
            s->out_c = p->out_c;
            s->layer_type = "Noop";
        }
        else
        {
#if OUTPUT_LAYER_MAP
            printf("%-8s (unsupported)\n", s->name.c_str());
#endif
        }
    }

    for (auto it = dnet.begin(); it != dnet.end(); it++)
    {
        auto s = *it;
        for (size_t i = 0; i < s->input_blobs.size(); i++)
        {
            auto p = get_section_by_output_blob(dnet, s->input_blobs[i]);
            if (p == NULL || p->layer_type != "Noop")
                continue;
            s->input_blobs[i] = p->input_blobs[0];
        }
    }

    for (auto it = dnet.begin(); it != dnet.end();)
        if ((*it)->layer_type == "Noop")
            it = dnet.erase(it);
        else
            it++;

    for (auto it = dnet.begin(); it != dnet.end(); it++)
    {
        auto s = *it;
        for (std::string output_name : s->output_blobs)
        {
            auto q = get_sections_by_input_blob(dnet, output_name);
            if (q.size() <= 1 || s->layer_type == "Split")
                continue;
            Section* p = new Section;
            p->layer_type = "Split";
            p->layer_name = s->layer_name + "_split";
            p->w = s->w;
            p->h = s->h;
            p->c = s->c;
            p->out_w = s->out_w;
            p->out_h = s->out_h;
            p->out_c = s->out_c;
            p->input_blobs.push_back(output_name);
            for (size_t i = 0; i < q.size(); i++)
            {
                std::string new_output_name = p->layer_name + "_" + std::to_string(i);
                p->output_blobs.push_back(new_output_name);

                for (size_t j = 0; j < q[i]->input_blobs.size(); j++)
                    if (q[i]->input_blobs[j] == output_name)
                        q[i]->input_blobs[j] = new_output_name;
            }
            it = dnet.insert(it + 1, p);
        }
    }

    if (merge_output && yolo_layer_count > 0)
    {
        std::vector<int> masks;
        std::vector<float> scale_x_y;

        Section* s = new Section;
        s->classes = yolo_layers[0]->classes;
        s->anchors = yolo_layers[0]->anchors;
        s->mask = yolo_layers[0]->mask;

        for (auto p : yolo_layers)
        {
            if (s->classes != p->classes)
                error("yolo object classes number not match, output cannot be merged.");

            if (s->anchors.size() != p->anchors.size())
                error("yolo layer anchor count not match, output cannot be merged.");

            for (size_t i = 0; i < s->anchors.size(); i++)
                if (s->anchors[i] != p->anchors[i])
                    error("yolo anchor size not match, output cannot be merged.");

            if (s->ignore_thresh > p->ignore_thresh)
                s->ignore_thresh = p->ignore_thresh;

            for (int m : p->mask)
                masks.push_back(m);

            scale_x_y.push_back(input_w * p->scale_x_y / p->w);
            s->input_blobs.push_back(p->input_blobs[0]);
        }

        for (auto it = dnet.begin(); it != dnet.end();)
            if ((*it)->name == "yolo")
                it = dnet.erase(it);
            else
                it++;

        s->layer_type = "Yolov3DetectionOutput";
        s->layer_name = "detection_out";
        s->output_blobs.push_back("output");
        s->param.push_back(format("0=%d", s->classes));                                                          //num_class
        s->param.push_back(format("1=%d", s->mask.size()));                                                      //num_box
        s->param.push_back(format("2=%f", s->ignore_thresh));                                                    //confidence_threshold
        s->param.push_back(format("-23304=%d%s", s->anchors.size(), array_to_float_string(s->anchors).c_str())); //biases
        s->param.push_back(format("-23305=%d%s", masks.size(), array_to_float_string(masks).c_str()));           //mask
        s->param.push_back(format("-23306=%d%s", scale_x_y.size(), array_to_float_string(scale_x_y).c_str()));   //biases_index

        dnet.push_back(s);
    }
}

void read_to(std::vector<float>& vec, size_t size, FILE* fp)
{
    vec.resize(size);
    size_t read_size = fread(&vec[0], sizeof(float), size, fp);
    if (read_size != size)
        error("\n Warning: Unexpected end of wights-file!\n");
}

void load_weights(const char* filename, std::deque<Section*>& dnet)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL)
        file_error(filename);

    int major, minor, revision;

    fread_or_error(&major, sizeof(int), 1, fp, filename);
    fread_or_error(&minor, sizeof(int), 1, fp, filename);
    fread_or_error(&revision, sizeof(int), 1, fp, filename);
    if ((major * 10 + minor) >= 2)
    {
        uint64_t iseen = 0;
        fread_or_error(&iseen, sizeof(uint64_t), 1, fp, filename);
    }
    else
    {
        uint32_t iseen = 0;
        fread_or_error(&iseen, sizeof(uint32_t), 1, fp, filename);
    }

    for (auto s : dnet)
    {
        if (s->name == "convolutional")
        {
            read_to(s->bias, s->filters, fp);
            if (s->batch_normalize)
            {
                read_to(s->scales, s->filters, fp);
                read_to(s->rolling_mean, s->filters, fp);
                read_to(s->rolling_variance, s->filters, fp);
            }

            if (s->layer_type == "Convolution")
                read_to(s->weights, (size_t)(s->c) * s->filters * s->size * s->size, fp);
            else if (s->layer_type == "ConvolutionDepthWise")
                read_to(s->weights, s->c * s->filters * s->size * s->size / s->groups, fp);
        }
    }

    fclose(fp);
}

int count_output_blob(std::deque<Section*>& dnet)
{
    int count = 0;
    for (auto s : dnet)
        count += (int)s->output_blobs.size();
    return count;
}

int main(int argc, char** argv)
{
    if (!(argc == 3 || argc == 5 || argc == 6))
    {
        fprintf(stderr, "Usage: %s [darknetcfg] [darknetweights] [ncnnparam] [ncnnbin] [merge_output]\n"
                "\t[darknetcfg]     .cfg file of input darknet model.\n"
                "\t[darknetweights] .weights file of input darknet model.\n"
                "\t[cnnparam]       .param file of output ncnn model.\n"
                "\t[ncnnbin]        .bin file of output ncnn model.\n"
                "\t[merge_output]   merge all output yolo layers into one, enabled by default.\n",
                argv[0]);
        return -1;
    }

    const char* darknetcfg = argv[1];
    const char* darknetweights = argv[2];
    const char* ncnn_param = argc >= 5 ? argv[3] : "ncnn.param";
    const char* ncnn_bin = argc >= 5 ? argv[4] : "ncnn.bin";
    int merge_output = argc >= 6 ? atoi(argv[5]) : 1;

    std::deque<Section*> dnet;

    printf("Loading cfg...\n");
    load_cfg(darknetcfg, dnet);
    parse_cfg(dnet, merge_output);

    printf("Loading weights...\n");
    load_weights(darknetweights, dnet);

    FILE* pp = fopen(ncnn_param, "wb");
    if (pp == NULL)
        file_error(ncnn_param);

    FILE* bp = fopen(ncnn_bin, "wb");
    if (bp == NULL)
        file_error(ncnn_bin);

    printf("Converting model...\n");

    fprintf(pp, "7767517\n");
    fprintf(pp, "%d %d\n", (int)dnet.size(), count_output_blob(dnet));

    for (auto s : dnet)
    {
        fprintf(pp, "%-22s %-20s %d %d", s->layer_type.c_str(), s->layer_name.c_str(), (int)s->input_blobs.size(), (int)s->output_blobs.size());
        for (auto b : s->input_blobs)
            fprintf(pp, " %s", b.c_str());
        for (auto b : s->output_blobs)
            fprintf(pp, " %s", b.c_str());
        for (auto p : s->param)
            fprintf(pp, " %s", p.c_str());
        fprintf(pp, "\n");

        if (s->name == "convolutional")
        {
            fseek(bp, 4, SEEK_CUR);
            if (s->weights.size() > 0)
                fwrite(&s->weights[0], sizeof(float), s->weights.size(), bp);
            if (s->scales.size() > 0)
                fwrite(&s->scales[0], sizeof(float), s->scales.size(), bp);
            if (s->rolling_mean.size() > 0)
                fwrite(&s->rolling_mean[0], sizeof(float), s->rolling_mean.size(), bp);
            if (s->rolling_variance.size() > 0)
                fwrite(&s->rolling_variance[0], sizeof(float), s->rolling_variance.size(), bp);
            if (s->bias.size() > 0)
                fwrite(&s->bias[0], sizeof(float), s->bias.size(), bp);
        }
    }
    fclose(pp);

    printf("%d layers, %d blobs generated.\n", (int)dnet.size(), count_output_blob(dnet));
    printf("NOTE: The input of darknet uses: mean_vals=0 and norm_vals=1/255.f.\n");
    if (!merge_output)
        printf("NOTE: There are %d unmerged yolo output layer. Make sure all outputs are processed with nms.\n", yolo_layer_count);
    printf("NOTE: Remeber to use ncnnoptimize for better performance.\n");

    return 0;
}
