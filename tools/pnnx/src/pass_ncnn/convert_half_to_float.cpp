// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_half_to_float.h"

#include <algorithm>
#include <stdio.h>

namespace pnnx {

namespace ncnn {

void convert_half_to_float(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        for (auto& x : op->attrs)
        {
            Attribute& attr = x.second;
            if (attr.type != 3 && attr.type != 13)
                continue;

            const int count = attr.elemcount();
            const bool empty = std::find(attr.shape.begin(), attr.shape.end(), 0) != attr.shape.end();
            const bool negative = std::any_of(attr.shape.begin(), attr.shape.end(), [](int dim) {
                return dim < 0;
            });
            const std::vector<float> data = attr.get_float32_data();
            if (negative || (count == 0 && !empty) || data.size() != (size_t)count || attr.data.size() / 2 != data.size() || attr.data.size() % 2 != 0)
            {
                fprintf(stderr, "cannot lower invalid half attribute %s.%s\n", op->name.c_str(), x.first.c_str());
                continue;
            }

            // Python/PNNX retain their original dtype. Only ncnn weights are
            // promoted here; this does not add arbitrary ncnn input dtypes.
            attr.type = 1;
            attr.set_float32_data(data);
        }
    }
}

} // namespace ncnn

} // namespace pnnx
