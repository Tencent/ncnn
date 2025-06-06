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

#include "convert_attribute.h"

namespace pnnx {

namespace ncnn {

void convert_attribute(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        if (op->type != "pnnx.Attribute")
            continue;

        op->type = "MemoryData";

        const std::string& key = op->attrs.begin()->first;
        const Attribute& data = op->attrs.begin()->second;

        const int batch_index = op->outputs[0]->params["__batch_index"].i;

        if ((int)data.shape.size() > 5)
        {
            fprintf(stderr, "pnnx attribute %d-rank tensor is not supported yet!\n", (int)data.shape.size());
            return;
        }

        // drop batch index
        std::vector<int> new_shape;
        for (int i = 0; i < (int)data.shape.size(); i++)
        {
            if (i == batch_index && data.shape[i] == 1)
                continue;

            new_shape.push_back(data.shape[i]);
        }

        if (new_shape.size() == 5 && batch_index == 233)
        {
            if (new_shape[0] == 1)
            {
                fprintf(stderr, "assume pnnx attribute 5-rank tensor has batch_index 0\n");
                new_shape.erase(new_shape.begin());
            }
            else
            {
                fprintf(stderr, "pnnx attribute 5-rank tensor is not supported yet!\n");
            }
        }

        if (new_shape.size() == 0)
        {
            // scalar
            op->params["0"] = 1;
        }
        if (new_shape.size() == 1)
        {
            op->params["0"] = new_shape[0];
        }
        if (new_shape.size() == 2)
        {
            op->params["0"] = new_shape[1];
            op->params["1"] = new_shape[0];
        }
        if (new_shape.size() == 3)
        {
            op->params["0"] = new_shape[2];
            op->params["1"] = new_shape[1];
            op->params["2"] = new_shape[0];
        }
        if (new_shape.size() == 4)
        {
            op->params["0"] = new_shape[3];
            op->params["1"] = new_shape[2];
            op->params["11"] = new_shape[1];
            op->params["2"] = new_shape[0];
        }

        if (key != "0")
        {
            op->attrs["0"] = data;
            op->attrs.erase(key);
        }
    }
}

} // namespace ncnn

} // namespace pnnx
