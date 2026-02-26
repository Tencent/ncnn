// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
