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

#include "onnx-ml.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include <fstream>

#include <onnxruntime_c_api.h>

#include "ir.h"

#include "pass_onnx/canonicalize.h"
#include "pass_onnx/dead_code_elimination.h"
#include "pass_onnx/eliminate_initializer_input.h"
#include "pass_onnx/eliminate_noop.h"
#include "pass_onnx/fold_constants.h"
#include "pass_onnx/inline_containers.h"
#include "pass_onnx/inline_if_graph.h"
#include "pass_onnx/model_stat.h"
#include "pass_onnx/shape_inference.h"
#include "pass_onnx/fuse_constant_as_attribute.h"

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
        if (!tensor_shape.dim(z).has_dim_value())
        {
            r->shape[z] = -1;
        }
        else
        {
            r->shape[z] = tensor_shape.dim(z).dim_value();
        }
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

static const char* get_onnx_tensor_elem_data_type_str(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return "i8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return "u8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return "i16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return "u16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return "i32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return "u32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "i64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return "u64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return "f16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "f32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return "f64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return "bf16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        return "c64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        return "c128";
    default:
        break;
    }

    // unknown
    fprintf(stderr, "unsupported tensor elem data type %d\n", (int)type);
    return "";
}

static bool check_input_shape(onnx::ModelProto& model, const std::vector<std::vector<int64_t> >& input_shapes, const std::vector<std::string>& input_types)
{
    const onnx::GraphProto& graph = model.graph();

    if (!input_shapes.empty() && (int)input_shapes.size() != graph.input_size())
    {
        fprintf(stderr, "input_shape expect %d tensors but got %d\n", graph.input_size(), (int)input_shapes.size());
        return false;
    }

    for (int i = 0; i < graph.input_size(); i++)
    {
        const onnx::ValueInfoProto& value = graph.input(i);
        const onnx::TensorShapeProto& tsp = value.type().tensor_type().shape();

        ONNXTensorElementDataType datatype = (ONNXTensorElementDataType)value.type().tensor_type().elem_type();

        bool matched = true;

        if (input_shapes.empty())
        {
            // dynamic dimension size
            for (int j = 0; j < tsp.dim_size(); j++)
            {
                if (!tsp.dim(j).has_dim_value())
                {
                    matched = false;
                    break;
                }
            }
        }
        else
        {
            if ((int)input_shapes[i].size() != tsp.dim_size())
            {
                matched = false;
            }
            else
            {
                for (int j = 0; j < tsp.dim_size(); j++)
                {
                    if (!tsp.dim(j).has_dim_value())
                        continue;

                    int64_t ds = tsp.dim(j).dim_value();
                    if (ds == -1)
                        continue;

                    if (input_shapes[i][j] != ds)
                        matched = false;
                }
            }

            if (input_types[i] != get_onnx_tensor_elem_data_type_str(datatype))
                matched = false;
        }

        if (!matched)
        {
            fprintf(stderr, "input_shapes[%d] expect [", i);
            for (int j = 0; j < tsp.dim_size(); j++)
            {
                if (tsp.dim(j).has_dim_value())
                {
                    int64_t ds = tsp.dim(j).dim_value();
                    fprintf(stderr, "%ld", ds);
                }
                else
                {
                    fprintf(stderr, "?");
                }
                if (j + 1 != tsp.dim_size())
                    fprintf(stderr, ",");
            }
            fprintf(stderr, "]%s but got ", get_onnx_tensor_elem_data_type_str(datatype));
            if (input_shapes.empty())
            {
                fprintf(stderr, "nothing\n");
            }
            else
            {
                fprintf(stderr, "[");
                for (size_t j = 0; j < input_shapes[i].size(); j++)
                {
                    fprintf(stderr, "%ld", input_shapes[i][j]);
                    if (j + 1 != input_shapes[i].size())
                        fprintf(stderr, ",");
                }
                fprintf(stderr, "]%s\n", input_types[i].c_str());
            }

            return false;
        }
    }

    return true;
}

int load_onnx(const std::string& onnxpath, Graph& pnnx_graph,
              const std::vector<std::vector<int64_t> >& input_shapes,
              const std::vector<std::string>& input_types,
              const std::vector<std::vector<int64_t> >& input_shapes2,
              const std::vector<std::string>& input_types2)
{
    onnx::ModelProto model;

    bool s1 = read_proto_from_binary(onnxpath.c_str(), &model);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }

    onnx2pnnx::eliminate_initializer_input(model);

    // input shape sanity check
    if (!check_input_shape(model, input_shapes, input_types))
    {
        return -1;
    }
    if (!input_shapes2.empty() && !check_input_shape(model, input_shapes2, input_types2))
    {
        return -1;
    }

    fprintf(stderr, "############# pass_level0 onnx \n");

    onnx2pnnx::ModelStat oldstat = onnx2pnnx::get_model_stat(model);

    double t0 = 0;
    double t1 = 0;

    int inlined = 0;

    do
    {
        fprintf(stderr, "%-34s", "inline_containers ... ");

        t0 = get_current_time();

        onnx2pnnx::inline_containers(model);

        t1 = get_current_time();

        fprintf(stderr, "%8.2fms\n", t1 - t0);

        fprintf(stderr, "%-34s", "eliminate_noop ... ");

        t0 = get_current_time();

        onnx2pnnx::eliminate_noop(model);

        t1 = get_current_time();

        fprintf(stderr, "%8.2fms\n", t1 - t0);

        fprintf(stderr, "%-34s", "fold_constants ... ");

        t0 = get_current_time();

        onnx2pnnx::fold_constants(model, input_shapes, input_types, input_shapes2, input_types2);

        t1 = get_current_time();

        fprintf(stderr, "%8.2fms\n", t1 - t0);

        fprintf(stderr, "%-34s", "canonicalize ... ");

        t0 = get_current_time();

        onnx2pnnx::canonicalize(model);

        t1 = get_current_time();

        fprintf(stderr, "%8.2fms\n", t1 - t0);

        fprintf(stderr, "%-34s", "shape_inference ... ");

        t0 = get_current_time();

        onnx2pnnx::shape_inference(model, input_shapes, input_types, input_shapes2, input_types2);

        t1 = get_current_time();

        fprintf(stderr, "%8.2fms\n", t1 - t0);

        fprintf(stderr, "%-34s", "fold_constants_dynamic_shape ... ");

        t0 = get_current_time();

        onnx2pnnx::fold_constants_dynamic_shape(model, input_shapes, input_types);

        t1 = get_current_time();

        fprintf(stderr, "%8.2fms\n", t1 - t0);

        fprintf(stderr, "%-34s", "inline_if_graph ... ");

        t0 = get_current_time();

        inlined = onnx2pnnx::inline_if_graph(model);

        t1 = get_current_time();

        fprintf(stderr, "%8.2fms\n", t1 - t0);

    } while (inlined);

    fprintf(stderr, "%-34s", "fuse_constant_as_attribute ... ");

    t0 = get_current_time();

    onnx2pnnx::fuse_constant_as_attribute(model);

    t1 = get_current_time();

    fprintf(stderr, "%8.2fms\n", t1 - t0);

    fprintf(stderr, "%-34s", "eliminate_noop_with_shape ... ");

    t0 = get_current_time();

    onnx2pnnx::eliminate_noop_with_shape(model);

    t1 = get_current_time();

    fprintf(stderr, "%8.2fms\n", t1 - t0);

    // save
    {
        std::string simonnx_path;
        if (onnxpath.size() > 5 && onnxpath.substr(onnxpath.size() - 5) == ".onnx")
        {
            simonnx_path = onnxpath.substr(0, onnxpath.size() - 5) + ".pnnxsim.onnx";
        }
        else
        {
            simonnx_path = onnxpath + ".pnnxsim.onnx";
        }
        std::fstream output(simonnx_path, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!model.SerializeToOstream(&output))
        {
            fprintf(stderr, "write pnnxsim onnx failed\n");
            return -1;
        }
    }

    onnx2pnnx::ModelStat newstat = onnx2pnnx::get_model_stat(model);

    onnx2pnnx::print_model_stat(oldstat, newstat);

    fprintf(stderr, "############# pass_level1 onnx\n");

    pass_onnx(model, pnnx_graph);

    return 0;
}

} // namespace pnnx
