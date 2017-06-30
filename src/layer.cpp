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

namespace ncnn {

Layer::Layer()
{
    one_blob_only = false;
    support_inplace = false;
}

Layer::~Layer()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int Layer::load_param(FILE* /*paramfp*/)
{
    return 0;
}
#endif // NCNN_STRING

int Layer::load_param_bin(FILE* /*paramfp*/)
{
    return 0;
}

int Layer::load_model(FILE* /*binfp*/)
{
    return 0;
}
#endif // NCNN_STDIO

int Layer::load_param(const unsigned char*& /*mem*/)
{
    return 0;
}

int Layer::load_model(const unsigned char*& /*mem*/)
{
    return 0;
}

int Layer::forward(const std::vector<Mat>& /*bottom_blobs*/, std::vector<Mat>& /*top_blobs*/) const
{
    return -1;
}

int Layer::forward(const Mat& /*bottom_blob*/, Mat& /*top_blob*/) const
{
    return -1;
}

int Layer::forward_inplace(std::vector<Mat>& bottom_top_blobs) const
{
    std::vector<Mat> top_blobs;
    int ret = forward(bottom_top_blobs, top_blobs);
    bottom_top_blobs = top_blobs;
    return ret;
}

int Layer::forward_inplace(Mat& bottom_top_blob) const
{
    Mat top_blob;
    int ret = forward(bottom_top_blob, top_blob);
    bottom_top_blob = top_blob;
    return ret;
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
        {
            return i;
        }
    }

    fprintf(stderr, "layer %s not exists\n", type);
    return -1;
}
#endif // NCNN_STRING

Layer* create_layer(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
    {
        fprintf(stderr, "layer index %d not exists\n", index);
        return 0;
    }

    layer_creator_func layer_creator = layer_registry[index].creator;
    if (!layer_creator)
    {
        fprintf(stderr, "layer index %d not enabled\n", index);
        return 0;
    }

    return layer_creator();
}

} // namespace ncnn
