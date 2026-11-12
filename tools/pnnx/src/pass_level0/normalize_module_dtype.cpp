// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_module_dtype.h"

namespace pnnx {

void normalize_module_dtype(torch::jit::Module& mod)
{
    for (auto submod : mod.children())
    {
        normalize_module_dtype(submod);
    }

    for (auto named_attr : mod.named_attributes(false))
    {
        const std::string& name = named_attr.name;
        auto attr = named_attr.value;

        if (attr.type()->kind() == c10::TypeKind::TensorType)
        {
            auto t = attr.toTensor();

            if (t.scalar_type() == c10::ScalarType::Half || t.scalar_type() == c10::ScalarType::BFloat16 || t.scalar_type() == c10::ScalarType::Double)
            {
                at::Tensor t_fp32 = t.toType(c10::ScalarType::Float);

                mod.setattr(name, t_fp32);
            }
            if (t.is_complex() && t.scalar_type() != c10::ScalarType::ComplexFloat)
            {
                mod.setattr(name, t.toType(c10::ScalarType::ComplexFloat));
            }
        }
    }
}

} // namespace pnnx
