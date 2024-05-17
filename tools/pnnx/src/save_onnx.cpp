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

#include "save_onnx.h"

#include "onnx.pb.h"

#include <string.h>
#include <fstream>
#include <iostream>

#include "utils.h"

namespace pnnx {

int save_onnx(const Graph& g, const char* onnxpath, int fp16)
{
    onnx::ModelProto model;

    onnx::GraphProto* gp = model.mutable_graph();

    for (const Operand* x : g.operands)
    {
        onnx::ValueInfoProto* vip = gp->add_value_info();

        vip->set_name(x->name);

        onnx::TypeProto* tp = vip->mutable_type();

        onnx::TypeProto_Tensor* tpt = tp->mutable_tensor_type();

        switch (x->type)
        {
        case 1: // f32
            tpt->set_elem_type(fp16 ? 10 : 1);
            break;
        case 2: // f64
            tpt->set_elem_type(fp16 ? 10 : 11);
            break;
        case 3: // f16
            tpt->set_elem_type(10);
            break;
        case 4: // i32
            tpt->set_elem_type(6);
            break;
        case 5: // i64
            tpt->set_elem_type(7);
            break;
        case 6: // i16
            tpt->set_elem_type(5);
            break;
        case 7: // i8
            tpt->set_elem_type(3);
            break;
        case 8: // u8
            tpt->set_elem_type(2);
            break;
        case 9: // bool
            tpt->set_elem_type(9);
            break;
        case 10: // cp64
            tpt->set_elem_type(14);
            break;
        case 11: // cp128
            tpt->set_elem_type(15);
            break;
        case 12: // cp32
            tpt->set_elem_type(0);
            break;
        default: // null
            tpt->set_elem_type(0);
            break;
        }

        onnx::TensorShapeProto* tsp = tpt->mutable_shape();

        for (auto s : x->shape)
        {
            onnx::TensorShapeProto_Dimension* tspd = tsp->add_dim();

            tspd->set_dim_value(s);
        }
    }

    for (const Operator* op : g.ops)
    {
        onnx::NodeProto* np = gp->add_node();

        np->set_op_type(op->type);
        np->set_name(op->name);

        for (const Operand* oprand : op->inputs)
        {
            np->add_input(oprand->name);
        }

        for (const Operand* oprand : op->outputs)
        {
            np->add_output(oprand->name);
        }

        for (const auto& it : op->params)
        {
            const Parameter& param = it.second;

            onnx::AttributeProto* ap = np->add_attribute();

            ap->set_name(it.first);

            if (param.type == 0)
            {
                ap->set_type(onnx::AttributeProto::STRING);
                ap->set_s("None");
            }
            if (param.type == 1)
            {
                ap->set_type(onnx::AttributeProto::INT);
                if (param.b)
                    ap->set_i(1);
                else
                    ap->set_i(0);
            }
            if (param.type == 2)
            {
                ap->set_type(onnx::AttributeProto::INT);
                ap->set_i(param.i);
            }
            if (param.type == 3)
            {
                ap->set_type(onnx::AttributeProto::FLOAT);
                ap->set_f(param.f);
            }
            if (param.type == 4)
            {
                ap->set_type(onnx::AttributeProto::STRING);
                ap->set_s(param.s);
            }
            if (param.type == 5)
            {
                ap->set_type(onnx::AttributeProto::INTS);
                for (auto i : param.ai)
                {
                    ap->add_ints(i);
                }
            }
            if (param.type == 6)
            {
                ap->set_type(onnx::AttributeProto::FLOATS);
                for (auto f : param.af)
                {
                    ap->add_floats(f);
                }
            }
            if (param.type == 7)
            {
                ap->set_type(onnx::AttributeProto::STRINGS);
                for (auto s : param.as)
                {
                    ap->add_strings(s);
                }
            }
        }

        for (const auto& it : op->attrs)
        {
            onnx::TensorProto* tp = gp->add_initializer();

            tp->set_name(op->name + "." + it.first);

            np->add_input(op->name + "." + it.first);

            const Attribute& attr = it.second;
            for (auto s : attr.shape)
            {
                tp->add_dims(s);
            }

            switch (attr.type)
            {
            case 1: // f32
                tp->set_data_type(fp16 ? 10 : 1);
                break;
            case 2: // f64
                tp->set_data_type(fp16 ? 10 : 11);
                break;
            case 3: // f16
                tp->set_data_type(10);
                break;
            case 4: // i32
                tp->set_data_type(6);
                break;
            case 5: // i64
                tp->set_data_type(7);
                break;
            case 6: // i16
                tp->set_data_type(5);
                break;
            case 7: // i8
                tp->set_data_type(3);
                break;
            case 8: // u8
                tp->set_data_type(2);
                break;
            case 9: // bool
                tp->set_data_type(9);
                break;
            case 10: // cp64
                tp->set_data_type(14);
                break;
            case 11: // cp128
                tp->set_data_type(15);
                break;
            case 12: // cp32
                tp->set_data_type(0);
                break;
            default: // null
                tp->set_data_type(0);
                break;
            }

            std::string* d = tp->mutable_raw_data();
            if (fp16 && attr.type == 1)
            {
                // fp32 to fp16
                const float* p = (const float*)attr.data.data();
                int len = attr.data.size() / 4;
                d->resize(len * 2);
                unsigned short* p_fp16 = (unsigned short*)d->data();
                for (int i = 0; i < len; i++)
                {
                    p_fp16[i] = float32_to_float16(p[i]);
                }
            }
            else if (fp16 && attr.type == 2)
            {
                // fp64 to fp16
                const double* p = (const double*)attr.data.data();
                int len = attr.data.size() / 4;
                d->resize(len);
                unsigned short* p_fp16 = (unsigned short*)d->data();
                for (int i = 0; i < len; i++)
                {
                    p_fp16[i] = float32_to_float16((float)p[i]);
                }
            }
            else
            {
                d->resize(attr.data.size());
                memcpy((void*)d->data(), attr.data.data(), attr.data.size());
            }
        }
    }

    std::fstream output(onnxpath, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!model.SerializeToOstream(&output))
    {
        fprintf(stderr, "write onnx failed\n");
        return -1;
    }

    return 0;
}

} // namespace pnnx
