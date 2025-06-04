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

#include <string.h>
#include "c_api.h"

static int test_c_api_0()
{
    ncnn_mat_t a = ncnn_mat_create_1d(2, NULL);
    ncnn_mat_t b = ncnn_mat_create_1d(2, NULL);
    ncnn_mat_t c = 0;

    ncnn_option_t opt = ncnn_option_create();

    // set a and b
    {
        ncnn_mat_fill_float(a, 2.f);
        ncnn_mat_fill_float(b, 3.f);
    }

    // c = a + b
    {
        ncnn_layer_t op = ncnn_layer_create_by_type("BinaryOp");

        // load param
        {
            ncnn_paramdict_t pd = ncnn_paramdict_create();
            ncnn_paramdict_set_int(pd, 0, 0); // op_type = ADD

            op->load_param(op, pd);

            ncnn_paramdict_destroy(pd);
        }

        // load model
        {
            ncnn_modelbin_t mb = ncnn_modelbin_create_from_mat_array(0, 0);

            op->load_model(op, mb);

            ncnn_modelbin_destroy(mb);
        }

        op->create_pipeline(op, opt);

        const ncnn_mat_t bottom_blobs[2] = {a, b};
        ncnn_mat_t top_blobs[1] = {0};
        op->forward_n(op, bottom_blobs, 2, top_blobs, 1, opt);
        c = top_blobs[0];

        op->destroy_pipeline(op, opt);

        ncnn_layer_destroy(op);
    }

    // check c == a + b
    bool success = false;
    if (c)
    {
        int dims = ncnn_mat_get_dims(c);
        int w = ncnn_mat_get_w(c);
        const float* c_data = (const float*)ncnn_mat_get_data(c);

        success = dims == 1 && w == 2 && c_data[0] == 5.f && c_data[1] == 5.f;
    }

    ncnn_option_destroy(opt);

    ncnn_mat_destroy(a);
    ncnn_mat_destroy(b);
    ncnn_mat_destroy(c);

    if (!success)
    {
        fprintf(stderr, "test_c_api_0 failed\n");
    }

    return success ? 0 : -1;
}

static int test_c_api_1()
{
    ncnn_mat_t a = ncnn_mat_create_1d(24, NULL);

    // set a
    {
        const float data[] = {
            0, 1, 2, 3, 4, 5, 6, 7,
            10, 11, 12, 13, 14, 15, 16, 17,
            20, 21, 22, 23, 24, 25, 26, 27
        };

        float* a_data = (float*)ncnn_mat_get_data(a);
        memcpy(a_data, data, 24 * sizeof(float));
    }

    ncnn_mat_t b = ncnn_mat_reshape_3d(a, 4, 2, 3, NULL);
    ncnn_mat_t c = 0;

    ncnn_option_t opt = ncnn_option_create();

    // c = reorg(b, 2)
    {
        ncnn_layer_t op = ncnn_layer_create_by_type("Reorg");

        // load param
        {
            ncnn_paramdict_t pd = ncnn_paramdict_create();
            ncnn_paramdict_set_int(pd, 0, 2); // stride

            op->load_param(op, pd);

            ncnn_paramdict_destroy(pd);
        }

        // load model
        {
            ncnn_modelbin_t mb = ncnn_modelbin_create_from_mat_array(0, 0);

            op->load_model(op, mb);

            ncnn_modelbin_destroy(mb);
        }

        op->create_pipeline(op, opt);

        op->forward_1(op, b, &c, opt);

        op->destroy_pipeline(op, opt);

        ncnn_layer_destroy(op);
    }

    // check c
    bool success = false;
    if (c)
    {
        int dims = ncnn_mat_get_dims(c);
        int w = ncnn_mat_get_w(c);
        int h = ncnn_mat_get_h(c);
        int ch = ncnn_mat_get_c(c);

        success = dims == 3 && w == 2 && h == 1 && ch == 12;

        const float expected[] = {
            0, 2,
            1, 3,
            4, 6,
            5, 7,
            10, 12,
            11, 13,
            14, 16,
            15, 17,
            20, 22,
            21, 23,
            24, 26,
            25, 27
        };
        ncnn_mat_t c2 = 0;
        ncnn_flatten(c, &c2, opt);
        const float* c2_data = (const float*)ncnn_mat_get_data(c2);
        if (memcmp(c2_data, expected, 24) != 0)
        {
            success = false;
        }
        ncnn_mat_destroy(c2);
    }

    ncnn_option_destroy(opt);

    ncnn_mat_destroy(a);
    ncnn_mat_destroy(b);
    ncnn_mat_destroy(c);

    if (!success)
    {
        fprintf(stderr, "test_c_api_1 failed\n");
    }

    return success ? 0 : -1;
}

static int mylayer_forward_inplace_1(const ncnn_layer_t layer, ncnn_mat_t bottom_top_blob, const ncnn_option_t opt)
{
    int w = ncnn_mat_get_w(bottom_top_blob);
    int h = ncnn_mat_get_h(bottom_top_blob);
    int channels = ncnn_mat_get_c(bottom_top_blob);
    int size = w * h;

    #pragma omp parallel for num_threads(ncnn_option_get_num_threads(opt))
    for (int q = 0; q < channels; q++)
    {
        float* ptr = (float*)ncnn_mat_get_channel_data(bottom_top_blob, q);
        for (int i = 0; i < size; i++)
        {
            *ptr = *ptr + 100.f;
            ptr++;
        }
    }

    return 0;
}

static ncnn_layer_t mylayer_creator(void* /*userdata*/)
{
    ncnn_layer_t layer = ncnn_layer_create();

    ncnn_layer_set_one_blob_only(layer, 1);
    ncnn_layer_set_support_inplace(layer, 1);

    layer->forward_inplace_1 = mylayer_forward_inplace_1;

    return layer;
}

static void mylayer_destroyer(ncnn_layer_t layer, void* /*userdata*/)
{
    ncnn_layer_destroy(layer);
}

static size_t emptydr_read(ncnn_datareader_t /*dr*/, void* buf, size_t size)
{
    memset(buf, 0, size);
    return size;
}

static int test_c_api_2()
{
    // datareader from empty
    ncnn_datareader_t emptydr = ncnn_datareader_create();
    {
        emptydr->read = emptydr_read;
    }

    ncnn_allocator_t blob_allocator = ncnn_allocator_create_pool_allocator();
    ncnn_allocator_t workspace_allocator = ncnn_allocator_create_unlocked_pool_allocator();

    ncnn_option_t opt = ncnn_option_create();
    {
        ncnn_option_set_num_threads(opt, 1);

        ncnn_option_set_blob_allocator(opt, blob_allocator);
        ncnn_option_set_workspace_allocator(opt, workspace_allocator);
    }

    ncnn_net_t net = ncnn_net_create();
    {
        ncnn_net_set_option(net, opt);

        ncnn_net_register_custom_layer_by_type(net, "MyLayer", mylayer_creator, mylayer_destroyer, 0);

        const char param_txt[] = "7767517\n2 2\nInput input 0 1 data\nMyLayer mylayer 1 1 data output\n";

        ncnn_net_load_param_memory(net, param_txt);
        ncnn_net_load_model_datareader(net, emptydr);
    }

    ncnn_mat_t a = ncnn_mat_create_1d(24, blob_allocator);

    // set a
    {
        const float data[] = {
            0, 1, 2, 3, 4, 5, 6, 7,
            10, 11, 12, 13, 14, 15, 16, 17,
            20, 21, 22, 23, 24, 25, 26, 27
        };

        float* a_data = (float*)ncnn_mat_get_data(a);
        memcpy(a_data, data, 24 * sizeof(float));
    }

    ncnn_mat_t b = ncnn_mat_reshape_3d(a, 4, 2, 3, blob_allocator);
    ncnn_mat_t c = 0;

    {
        ncnn_extractor_t ex = ncnn_extractor_create(net);

        ncnn_extractor_input(ex, "data", b);

        ncnn_extractor_extract(ex, "output", &c);

        ncnn_extractor_destroy(ex);
    }

    ncnn_net_destroy(net);

    // check c
    bool success = false;
    if (c)
    {
        int dims = ncnn_mat_get_dims(c);
        int w = ncnn_mat_get_w(c);
        int h = ncnn_mat_get_h(c);
        int ch = ncnn_mat_get_c(c);

        success = dims == 3 && w == 4 && h == 2 && ch == 3;

        const float expected[] = {
            100, 101, 102, 103, 104, 105, 106, 107,
            110, 111, 112, 113, 114, 115, 116, 117,
            120, 121, 122, 123, 124, 125, 126, 127
        };
        ncnn_mat_t c2 = 0;
        ncnn_flatten(c, &c2, opt);
        const float* c2_data = (const float*)ncnn_mat_get_data(c2);
        if (memcmp(c2_data, expected, 24) != 0)
        {
            success = false;
        }
        ncnn_mat_destroy(c2);
    }

    ncnn_mat_destroy(a);
    ncnn_mat_destroy(b);
    ncnn_mat_destroy(c);

    ncnn_option_destroy(opt);

    ncnn_allocator_destroy(blob_allocator);
    ncnn_allocator_destroy(workspace_allocator);

    ncnn_datareader_destroy(emptydr);

    if (!success)
    {
        fprintf(stderr, "test_c_api_2 failed\n");
    }

    return success ? 0 : -1;
}

int main()
{
    return test_c_api_0() || test_c_api_1() || test_c_api_2();
}
