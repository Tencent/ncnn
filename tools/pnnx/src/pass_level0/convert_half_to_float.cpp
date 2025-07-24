// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
            if (t.scalar_type() == c10::ScalarType::BFloat16)
            {
                at::Tensor t_fp32 = t.toType(c10::ScalarType::Float);

                mod.setattr(name, t_fp32);
            }
        }
    }
}

} // namespace pnnx
