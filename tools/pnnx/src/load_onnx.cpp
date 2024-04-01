// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "load_onnx.h"

#include "onnx.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include <fstream>

#include "ir.h"

#include "pass_onnx/canonicalize.h"
#include "pass_onnx/dead_code_elimination.h"
#include "pass_onnx/eliminate_noop.h"
#include "pass_onnx/fold_constants.h"
#include "pass_onnx/inline_containers.h"
#include "pass_onnx/model_stat.h"
#include "pass_onnx/shape_inference.h"

#include "pass_onnx.h"

namespace pnnx {

static size_t type_to_elemsize(int type)
{
    if (type == 1) return 4;
    if (type == 2) return 8;
    if (type == 3) return 2;
    if (type == 4) return 4;
    if (type == 5) return 8;
    if (type == 6) return 2;
    if (type == 7) return 1;
    if (type == 8) return 1;
    if (type == 9) return 1;
    if (type == 10) return 8;
    if (type == 11) return 16;
    if (type == 12) return 4;
    return 0; // null
}

static int get_onnx_tensor_type(int32_t dt)
{
    if (dt == onnx::TensorProto::FLOAT) return 1;
    if (dt == onnx::TensorProto::DOUBLE) return 2;
    if (dt == onnx::TensorProto::FLOAT16) return 3;
    if (dt == onnx::TensorProto::INT32) return 4;
    if (dt == onnx::TensorProto::INT64) return 5;
    if (dt == onnx::TensorProto::INT16) return 6;
    if (dt == onnx::TensorProto::INT8) return 7;
    if (dt == onnx::TensorProto::UINT8) return 8;
    if (dt == onnx::TensorProto::BOOL) return 9;
    if (dt == onnx::TensorProto::COMPLEX64) return 10;
    if (dt == onnx::TensorProto::COMPLEX128) return 11;
    return 0; // unknown type
}

Parameter::Parameter(const onnx::AttributeProto& attr)
{
    type = 0;

    switch (attr.type())
    {
    case onnx::AttributeProto::INT:
    {
        type = 2;
        int64_t i64 = attr.i();
        if (i64 == std::numeric_limits<int64_t>::max()) i64 = INT_MAX;
        if (i64 == std::numeric_limits<int64_t>::min()) i64 = INT_MIN;
        i = (int)i64;
        break;
    }
    case onnx::AttributeProto::FLOAT:
    {
        type = 3;
        f = attr.f();
        break;
    }
    case onnx::AttributeProto::STRING:
    {
        type = 4;
        s = attr.s();
        break;
    }
    case onnx::AttributeProto::INTS:
    {
        type = 5;
        for (int i = 0; i < attr.ints().size(); i++)
        {
            int64_t i64 = attr.ints().at(i);
            if (i64 == std::numeric_limits<int64_t>::max()) i64 = INT_MAX;
            if (i64 == std::numeric_limits<int64_t>::min()) i64 = INT_MIN;
            ai.push_back(i64);
        }
        break;
    }
    case onnx::AttributeProto::FLOATS:
    {
        type = 6;
        for (int i = 0; i < attr.floats().size(); i++)
        {
            float f = attr.floats().at(i);
            af.push_back(f);
        }
        break;
    }
    case onnx::AttributeProto::STRINGS:
    {
        type = 7;
        for (int i = 0; i < attr.strings().size(); i++)
        {
            std::string s = attr.strings().at(i);
            as.push_back(s);
        }
        break;
    }
    case onnx::AttributeProto::TENSOR:
    {
        const onnx::TensorProto& tensor = attr.t();

        int64_t numel = 1;
        for (int k = 0; k < tensor.dims_size(); k++)
        {
            numel *= tensor.dims(k);
        }

        if (numel == 1)
        {
            if (tensor.data_type() == onnx::TensorProto::INT32)
            {
                type = 2;
                if (tensor.has_raw_data())
                {
                    // assert tensor.raw_data().size() == 4
                    i = ((int*)tensor.raw_data().data())[0];
                }
                else
                {
                    // assert tensor.int32_data().size() == 1
                    i = tensor.int32_data().at(0);
                }
            }
            else if (tensor.data_type() == onnx::TensorProto::INT64)
            {
                type = 2;
                int64_t i64;
                if (tensor.has_raw_data())
                {
                    // assert tensor.raw_data().size() == 8
                    i64 = ((int64_t*)tensor.raw_data().data())[0];
                }
                else
                {
                    // assert tensor.int64_data().size() == 1
                    i64 = tensor.int64_data().at(0);
                }
                if (i64 == std::numeric_limits<int64_t>::max()) i64 = INT_MAX;
                if (i64 == std::numeric_limits<int64_t>::min()) i64 = INT_MIN;
                i = (int)i64;
            }
            else if (tensor.data_type() == onnx::TensorProto::FLOAT)
            {
                type = 3;
                if (tensor.has_raw_data())
                {
                    // assert tensor.raw_data().size() == 4
                    f = ((float*)tensor.raw_data().data())[0];
                }
                else
                {
                    // assert tensor.float_data().size() == 1
                    f = tensor.float_data().at(0);
                }
            }
            else
            {
                fprintf(stderr, "unknown Node attribute tensor data type %d\n", (int)tensor.data_type());
            }
        }
        else
        {
            // constant tensor will become pnnx attribute node later
            type = 8;
        }
        break;
    }
    default:
    {
        fprintf(stderr, "unknown Node attribute type %d\n", (int)attr.type());
        break;
    }
    }
}

Parameter::Parameter(const onnx2pnnx::OnnxAttributeProxy& attr)
    : Parameter(attr.attr)
{
}

Attribute::Attribute(const onnx::TensorProto& t)
{
    type = get_onnx_tensor_type(t.data_type());

    const int ndim = (int)t.dims_size();

    if (ndim == 0)
    {
        shape = {1};

        data.resize(type_to_elemsize(type));

        if (t.has_raw_data())
        {
            // assert t.raw_data().size() == type_to_elemsize(type)
            memcpy((void*)data.data(), (const void*)t.raw_data().data(), t.raw_data().size());
        }
        else if (t.data_type() == onnx::TensorProto::INT64)
        {
            int64_t i = t.int64_data().at(0);
            memcpy((void*)data.data(), (const void*)&i, data.size());
        }
        else if (t.data_type() == onnx::TensorProto::INT32)
        {
            int i = t.int32_data().at(0);
            memcpy((void*)data.data(), (const void*)&i, data.size());
        }
        else if (t.data_type() == onnx::TensorProto::DOUBLE)
        {
            double f = t.double_data().at(0);
            memcpy((void*)data.data(), (const void*)&f, data.size());
        }
        else if (t.data_type() == onnx::TensorProto::FLOAT)
        {
            float f = t.float_data().at(0);
            memcpy((void*)data.data(), (const void*)&f, data.size());
        }
        else
        {
            fprintf(stderr, "unknown Attribute tensor scalar type %d\n", type);
        }

        return;
    }

    shape.resize(ndim);
    for (int i = 0; i < ndim; i++)
        shape[i] = t.dims(i);

    if (shape.size() > 0)
    {
        data.resize(elemcount() * type_to_elemsize(type));

        if (t.has_raw_data())
        {
            memcpy((void*)data.data(), (const void*)t.raw_data().data(), data.size());
        }
        else if (t.data_type() == onnx::TensorProto::INT64)
        {
            memcpy((void*)data.data(), (const void*)t.int64_data().data(), data.size());
        }
        else if (t.data_type() == onnx::TensorProto::INT32)
        {
            memcpy((void*)data.data(), (const void*)t.int32_data().data(), data.size());
        }
        else if (t.data_type() == onnx::TensorProto::DOUBLE)
        {
            memcpy((void*)data.data(), (const void*)t.double_data().data(), data.size());
        }
        else if (t.data_type() == onnx::TensorProto::FLOAT)
        {
            memcpy((void*)data.data(), (const void*)t.float_data().data(), data.size());
        }
        else
        {
            fprintf(stderr, "unknown Attribute tensor scalar type %d\n", type);
        }
    }
}

Operand* Graph::new_operand(const onnx::ValueInfoProto& value)
{
    Operand* r = new Operand;
    r->name = value.name();

    int32_t et = value.type().tensor_type().elem_type();
    r->type = get_onnx_tensor_type(et);

    const onnx::TensorShapeProto& tensor_shape = value.type().tensor_type().shape();
    r->shape.resize(tensor_shape.dim_size());
    for (int z = 0; z < tensor_shape.dim_size(); z++)
    {
        r->shape[z] = tensor_shape.dim(z).dim_value();
    }

    operands.push_back(r);
    return r;
}

Operand* Graph::new_operand(const onnx::TensorProto& t)
{
    Operand* r = new Operand;
    r->name = t.name();

    r->type = get_onnx_tensor_type(t.data_type());

    const int ndim = (int)t.dims_size();
    if (ndim == 0)
    {
        r->shape = {1};
    }
    else
    {
        r->shape.resize(ndim);
        for (int i = 0; i < ndim; i++)
            r->shape[i] = t.dims(i);
    }

    operands.push_back(r);
    return r;
}

static bool read_proto_from_binary(const char* filepath, onnx::ModelProto* message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

#if GOOGLE_PROTOBUF_VERSION >= 3011000
    codedstr.SetTotalBytesLimit(INT_MAX);
#else
    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

static double get_current_time()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return usec.count() / 1000.0;
}

int load_onnx(const std::string& onnxpath, Graph& pnnx_graph)
{
    onnx::ModelProto model;

    bool s1 = read_proto_from_binary(onnxpath.c_str(), &model);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }

    fprintf(stderr, "############# pass_level0 onnx \n");

    onnx2pnnx::ModelStat oldstat = onnx2pnnx::get_model_stat(model);

    fprintf(stderr, "%-30s", "inline_containers ... ");

    double t0 = get_current_time();

    onnx2pnnx::inline_containers(model);

    double t1 = get_current_time();

    fprintf(stderr, "%10.2fms\n", t1 - t0);

    fprintf(stderr, "%-30s", "eliminate_noop ... ");

    t0 = get_current_time();

    onnx2pnnx::eliminate_noop(model);

    t1 = get_current_time();

    fprintf(stderr, "%10.2fms\n", t1 - t0);

    fprintf(stderr, "%-30s", "dead_code_elimination ... ");

    t0 = get_current_time();

    onnx2pnnx::dead_code_elimination(model);

    t1 = get_current_time();

    fprintf(stderr, "%10.2fms\n", t1 - t0);

    fprintf(stderr, "%-30s", "fold_constants ... ");

    t0 = get_current_time();

    onnx2pnnx::fold_constants(model);

    t1 = get_current_time();

    fprintf(stderr, "%10.2fms\n", t1 - t0);

    fprintf(stderr, "%-30s", "dead_code_elimination ... ");

    t0 = get_current_time();

    onnx2pnnx::dead_code_elimination(model);

    t1 = get_current_time();

    fprintf(stderr, "%10.2fms\n", t1 - t0);

    fprintf(stderr, "%-30s", "canonicalize ... ");

    t0 = get_current_time();

    onnx2pnnx::canonicalize(model);

    t1 = get_current_time();

    fprintf(stderr, "%10.2fms\n", t1 - t0);

    fprintf(stderr, "%-30s", "shape_inference ... ");

    t0 = get_current_time();

    onnx2pnnx::shape_inference(model);

    t1 = get_current_time();

    fprintf(stderr, "%10.2fms\n", t1 - t0);

    // save
    std::fstream output("tmp2.onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    if (!model.SerializeToOstream(&output))
    {
        fprintf(stderr, "write onnx failed\n");
        return -1;
    }

    onnx2pnnx::ModelStat newstat = onnx2pnnx::get_model_stat(model);

    onnx2pnnx::print_model_stat(oldstat, newstat);

    fprintf(stderr, "############# pass_level1 onnx\n");

    pass_onnx(model, pnnx_graph);

    return 0;
}

} // namespace pnnx
