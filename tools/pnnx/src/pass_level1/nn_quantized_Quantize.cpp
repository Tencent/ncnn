// Tencent is pleased to support the open source community by making ncnn available.
//
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

#include "pass_level1.h"

#include "../utils.h"

namespace pnnx {

class Quantize : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.quantized.modules.Quantize";
    }

    const char* type_str() const
    {
        return "nn.quantized.Quantize";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        //         mod.dump(true, false, false);

        //         graph->dump();

        const torch::jit::Node* quantize_per_tensor = find_node_by_kind(graph, "aten::quantize_per_tensor");

        //         for (auto aa : quantize_per_tensor->schema().arguments())
        //         {
        //             fprintf(stderr, "arg %s\n", aa.name().c_str());
        //         }

        // scale, zero_point
        op->params["scale"] = quantize_per_tensor->namedInput("scale");
        op->params["zero_point"] = quantize_per_tensor->namedInput("zero_point");
        op->params["dtype"] = "torch.qint8";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Quantize)

} // namespace pnnx
