// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

namespace pnnx {

void convert_half_to_float(torch::jit::Module& mod)
{
    for (auto submod : mod.children())
    {
        convert_half_to_float(submod);
    }

    for (auto named_attr : mod.named_attributes(false))
    {
        const std::string& name = named_attr.name;
        auto attr = named_attr.value;

        if (attr.type()->kind() == c10::TypeKind::TensorType)
        {
            auto t = attr.toTensor();

            if (t.scalar_type() == c10::ScalarType::Half)
            {
                at::Tensor t_fp32 = t.toType(c10::ScalarType::Float);

                mod.setattr(name, t_fp32);
            }
        }
    }
}

} // namespace pnnx
