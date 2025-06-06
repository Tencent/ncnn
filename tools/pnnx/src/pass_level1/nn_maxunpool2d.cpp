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

#include "../pass_level3/fuse_expression.h"

#include "../utils.h"

namespace pnnx {

class MaxUnpool2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.MaxUnpool2d";
    }

    const char* type_str() const
    {
        return "nn.MaxUnpool2d";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        graph->dump();

        {
            Graph pnnx_graph;

            pass_level1(mod, graph, pnnx_graph);

            fuse_expression(pnnx_graph);

            Operator* expr_op = pnnx_graph.ops[2];

            if (expr_op->type == "pnnx.Expression")
            {
                std::string expr = expr_op->params["expr"].s;

                int stride0;
                int stride1;
                int kernel_size0;
                int kernel_size1;
                int padding0;
                int padding1;
                int nscan = sscanf(expr.c_str(), "(int(sub(add(mul(sub(size(@0,2),1),%d),%d),%d)),int(sub(add(mul(sub(size(@1,3),1),%d),%d),%d)))", &stride0, &kernel_size0, &padding0, &stride1, &kernel_size1, &padding1);
                if (nscan == 6)
                {
                    op->params["kernel_size"] = Parameter{kernel_size0, kernel_size1};
                    op->params["stride"] = Parameter{stride0, stride1};
                    op->params["padding"] = Parameter{padding0 / 2, padding1 / 2};
                }
            }
        }

        const torch::jit::Node* max_unpool2d = find_node_by_kind(graph, "aten::max_unpool2d");

        for (auto aa : max_unpool2d->schema().arguments())
        {
            fprintf(stderr, "arg %s\n", aa.name().c_str());
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(MaxUnpool2d)

} // namespace pnnx
