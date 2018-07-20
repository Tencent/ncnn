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

#include "layer.h"

#include <stdio.h>
#include <string.h>
#include "cpu.h"

namespace ncnn {

Option::Option()
{
    lightmode = true;
    num_threads = get_cpu_count();
    blob_allocator = 0;
    workspace_allocator = 0;
}

static Option g_default_option;

const Option& get_default_option()
{
    return g_default_option;
}

int set_default_option(const Option& opt)
{
    if (opt.num_threads <= 0)
    {
        fprintf(stderr, "invalid option num_threads %d\n", opt.num_threads);
        return -1;
    }

    g_default_option = opt;

    return 0;
}

Layer::Layer()
{
    one_blob_only = false;
    support_inplace = false;
}

Layer::~Layer()
{
}

int Layer::load_param(const ParamDict& /*pd*/)
{
    return 0;
}

int Layer::load_model(const ModelBin& /*mb*/)
{
    return 0;
}

int Layer::load_scale(const char* scalepath)
{
    std::ifstream in(scalepath);
    std::string line;
    
    std::vector<std::string> objectLine;
    std::vector<std::string> objects;

    // read the file line to strings
    if (in)
    {
        while (getline(in, line))
        {
            objectLine.push_back(line);
        }
    }
    else
    {
        std::cout << "no such file " << scalepath << std::endl;
        return -1;
    }

    for (std::vector<std::string>::iterator iter = objectLine.begin(); iter != objectLine.end(); iter++)
    {
        std::istringstream temp((*iter));

        std::string str1,str2;
        temp >> str1 >> str2;
        objects.push_back(str1);
        objects.push_back(str2);
    }

    //find the layer scales
    std::string layer_name = name;
    bool flag = false;

    //initial the default value of scaleValue in this layer
    scaleValue.name = layer_name;
    scaleValue.dataScale = 1.0f;
    scaleValue.weightScale = 1.0f;

    for (std::vector<std::string>::iterator iter = objects.begin(); iter != objects.end(); iter++)
    {
        if(layer_name == *iter)
        {
            if (flag == false)
            {
                scaleValue.dataScale = stringToNum<float>(*(iter + 1));
                flag = true;
            }
        }

        //weight scale
        std::string param_name = layer_name+"_param_0";
        if(param_name == *iter)
        {
            scaleValue.weightScale = stringToNum<float>(*(iter + 1));
        }
    }    
#if NCNN_INT8_INFO
    fprintf(stderr, "%-28s dataScale:%-12f weightScale:%-12f\n", \
                scaleValue.name.c_str(), scaleValue.dataScale, scaleValue.weightScale);                
#endif
    top_scale = scaleValue.dataScale;

    return 0;
}

int Layer::load_scale_bin(const unsigned char* mem)
{
    stQuantizeParamsBin *scaleBin = (stQuantizeParamsBin*)mem;

    scaleValue.dataScale = scaleBin->dataScale;
    scaleValue.weightScale = scaleBin->weightScale;
#if NCNN_INT8_INFO
    fprintf(stderr, "name_index:%-16d dataScale:%-12f weightScale:%-12f\n", \
                scaleBin->index, scaleValue.dataScale, scaleValue.weightScale);    
#endif
    top_scale = scaleValue.dataScale;

    return 0;
}


int Layer::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blobs = bottom_blobs;
    for (int i = 0; i < (int)top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blobs[i].clone(opt.blob_allocator);
        if (top_blobs[i].empty())
            return -100;
    }

    return forward_inplace(top_blobs, opt);
}

int Layer::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blob = bottom_blob.clone(opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    return forward_inplace(top_blob, opt);
}

int Layer::forward_inplace(std::vector<Mat>& /*bottom_top_blobs*/, const Option& /*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(Mat& /*bottom_top_blob*/, const Option& /*opt*/) const
{
    return -1;
}

#include "layer_declaration.h"

static const layer_registry_entry layer_registry[] =
{
#include "layer_registry.h"
};

static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);

#if NCNN_STRING
int layer_to_index(const char* type)
{
    for (int i=0; i<layer_registry_entry_count; i++)
    {
        if (strcmp(type, layer_registry[i].name) == 0)
            return i;
    }

    return -1;
}

Layer* create_layer(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return 0;

    return create_layer(index);
}
#endif // NCNN_STRING

Layer* create_layer(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    return layer_creator();
}

} // namespace ncnn
