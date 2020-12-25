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

#include "c_api.h"

static int test_c_api_0()
{
    ncnn_mat_t a = ncnn_mat_create_1d(2);
    ncnn_mat_t b = ncnn_mat_create_1d(2);
    ncnn_mat_t c = 0;

    // set a and b
    {
        ncnn_mat_fill_float(a, 2.f);
        ncnn_mat_fill_float(b, 3.f);
    }

    // c = a + b
    {
        ncnn_option_t opt = ncnn_option_create();

        ncnn_layer_t op = ncnn_layer_create_by_type("BinaryOp");

        // load param
        {
            ncnn_paramdict_t pd = ncnn_paramdict_create();
            ncnn_paramdict_set_int(pd, 0, 0); // op_type = ADD

            ncnn_layer_load_param(op, pd);

            ncnn_paramdict_destroy(pd);
        }

        // load model
        {
            ncnn_mat_t weights[0];
            ncnn_modelbin_t mb = ncnn_modelbin_from_mat_array(weights, 0);

            ncnn_layer_load_model(op, mb);

            ncnn_modelbin_destroy(mb);
        }

        ncnn_layer_create_pipeline(op, opt);

        const ncnn_mat_t bottom_blobs[2] = {a, b};
        ncnn_mat_t* top_blobs[1] = {&c};
        ncnn_layer_forward_n(op, bottom_blobs, 2, top_blobs, 1, opt);

        ncnn_layer_destroy_pipeline(op, opt);

        ncnn_layer_destroy(op);

        ncnn_option_destroy(opt);
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

    ncnn_mat_destroy(a);
    ncnn_mat_destroy(b);
    ncnn_mat_destroy(c);

    return success ? 0 : -1;
}

int main()
{
    return test_c_api_0();
}
