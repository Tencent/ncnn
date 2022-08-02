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

#include "convert_to_fp16_model.h"

namespace pnnx {

namespace ncnn {

static unsigned short float32_to_float16(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // Some normal fp32 cannot be expressed as normal fp16
            fp16 = (sign << 15) | (0x00 << 10) | 0x00;
        }
        else
        {
            // normal fp16
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

void convert_to_fp16_model(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        bool is_type_flag_fp32 = false;
        for (auto& it : op->attrs)
        {
            Attribute& attr = it.second;

            if (is_type_flag_fp32)
            {
                // fp32 -> fp16
                const float* p = (const float*)attr.data.data();
                int len = attr.data.size() / 4;
                std::vector<char> data_fp16(len * 2);
                unsigned short* p_fp16 = (unsigned short*)data_fp16.data();
                for (int i = 0; i < len; i++)
                {
                    p_fp16[i] = float32_to_float16(p[i]);
                }

                attr.type = 3;
                attr.data = data_fp16;

                is_type_flag_fp32 = false;
                continue;
            }

            if (attr.type == 0 && attr.data == std::vector<char> {0, 0, 0, 0})
            {
                // write fp16 flag
                // unsigned int fp16_flag = 0x01306B47;
                attr.data[0] = 0x47;
                attr.data[1] = 0x6B;
                attr.data[2] = 0x30;
                attr.data[3] = 0x01;

                is_type_flag_fp32 = true;
                continue;
            }
        }
    }
}

} // namespace ncnn

} // namespace pnnx
