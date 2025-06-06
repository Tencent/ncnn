// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "convert_half_to_float.h"

#include <string.h>

namespace pnnx {

namespace ncnn {

void convert_half_to_float(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        while (1)
        {
            bool matched = false;

            for (auto x : op->attrs)
            {
                const Attribute& attr = x.second;
                if (attr.type != 3)
                    continue;

                matched = true;

                // fp16 -> fp32
                Attribute attr_new;
                attr_new.type = 1;
                attr_new.shape = attr.shape;
                attr_new.data.resize(attr.elemcount() * 4);

                auto p = attr.get_float32_data();
                memcpy((void*)attr_new.data.data(), (const void*)p.data(), attr_new.data.size());

                op->attrs[x.first] = attr_new;

                break;
            }

            if (!matched)
                break;
        }
    }
}

} // namespace ncnn

} // namespace pnnx
