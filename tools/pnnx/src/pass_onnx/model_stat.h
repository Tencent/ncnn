// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "onnx-ml.pb.h"

namespace pnnx {

namespace onnx2pnnx {

struct ModelStat
{
    ModelStat()
    {
        node_size = 0;
        initializer_size = 0;
        functions_size = 0;

        nn_module_count = 0;
        custom_module_count = 0;
        aten_count = 0;
        prims_count = 0;
        onnx_count = 0;
    }

    int node_size;
    int initializer_size;
    int functions_size;

    int nn_module_count;
    int custom_module_count;
    int aten_count;
    int prims_count;
    int onnx_count;

    std::map<std::string, int> nn_module_op_count;
    std::map<std::string, int> aten_op_count;
    std::map<std::string, int> prims_op_count;
    std::map<std::string, int> onnx_op_count;
};

ModelStat get_model_stat(const onnx::ModelProto& model);

void print_model_stat(const ModelStat& oldstat, const ModelStat& newstat);

} // namespace onnx2pnnx

} // namespace pnnx
