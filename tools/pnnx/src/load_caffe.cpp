// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_caffe.h"

#include <fstream>
#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <google/protobuf/text_format.h>

#include "caffe.pb.h"
#include "pass_caffe.h"

namespace pnnx {

static std::string caffe_identifier(const std::string& name, size_t index)
{
    std::string s = "caffe_" + std::to_string(index) + "_";
    for (char c : name)
    {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')
            s += c;
        else
            s += '_';
    }
    return s;
}

static int load_caffe_parameters(const google::protobuf::Message& message, const std::string& prefix, Operator* op)
{
    const google::protobuf::Descriptor* descriptor = message.GetDescriptor();
    const google::protobuf::Reflection* reflection = message.GetReflection();
    for (int i = 0; i < descriptor->field_count(); i++)
    {
        const google::protobuf::FieldDescriptor* field = descriptor->field(i);
        const std::string key = prefix + field->name();
        // interp uses field presence to distinguish explicit size from zoom/shrink
        if (descriptor == caffe::InterpParameter::descriptor() && !reflection->HasField(message, field))
            continue;
        if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE)
        {
            if (field->is_repeated())
            {
                for (int j = 0; j < reflection->FieldSize(message, field); j++)
                    if (load_caffe_parameters(reflection->GetRepeatedMessage(message, field, j), key + "." + std::to_string(j) + ".", op) != 0)
                        return -1;
            }
            else if (reflection->HasField(message, field))
            {
                if (load_caffe_parameters(reflection->GetMessage(message, field), key + ".", op) != 0)
                    return -1;
            }
            continue;
        }

        if (op->params.count(key))
        {
            fprintf(stderr, "caffe layer %s (%s) has duplicate parameter %s\n", op->name.c_str(), op->type.c_str(), key.c_str());
            return -1;
        }

        const int value_count = field->is_repeated() ? reflection->FieldSize(message, field) : 1;
        for (int j = 0; j < value_count; j++)
        {
            bool overflow = false;
            if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_INT64)
            {
                const int64_t v = field->is_repeated() ? reflection->GetRepeatedInt64(message, field, j) : reflection->GetInt64(message, field);
                overflow = v < INT_MIN || v > INT_MAX;
            }
            if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_UINT32)
            {
                const uint64_t v = field->is_repeated() ? reflection->GetRepeatedUInt32(message, field, j) : reflection->GetUInt32(message, field);
                overflow = v > INT_MAX;
            }
            if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_UINT64)
            {
                const uint64_t v = field->is_repeated() ? reflection->GetRepeatedUInt64(message, field, j) : reflection->GetUInt64(message, field);
                overflow = v > INT_MAX;
            }
            if (overflow)
            {
                fprintf(stderr, "caffe layer %s (%s) parameter %s exceeds integer range\n", op->name.c_str(), op->type.c_str(), key.c_str());
                return -1;
            }
        }

        if (field->is_repeated())
        {
            std::vector<int> ints;
            std::vector<float> floats;
            std::vector<std::string> strings;
            for (int j = 0; j < reflection->FieldSize(message, field); j++)
            {
                switch (field->cpp_type())
                {
                case google::protobuf::FieldDescriptor::CPPTYPE_INT32:
                    ints.push_back(reflection->GetRepeatedInt32(message, field, j));
                    break;
                case google::protobuf::FieldDescriptor::CPPTYPE_INT64:
                    ints.push_back(reflection->GetRepeatedInt64(message, field, j));
                    break;
                case google::protobuf::FieldDescriptor::CPPTYPE_UINT32:
                    ints.push_back(reflection->GetRepeatedUInt32(message, field, j));
                    break;
                case google::protobuf::FieldDescriptor::CPPTYPE_UINT64:
                    ints.push_back(reflection->GetRepeatedUInt64(message, field, j));
                    break;
                case google::protobuf::FieldDescriptor::CPPTYPE_BOOL:
                    ints.push_back(reflection->GetRepeatedBool(message, field, j));
                    break;
                case google::protobuf::FieldDescriptor::CPPTYPE_ENUM:
                    ints.push_back(reflection->GetRepeatedEnum(message, field, j)->number());
                    break;
                case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
                    floats.push_back(reflection->GetRepeatedFloat(message, field, j));
                    break;
                case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
                    floats.push_back(reflection->GetRepeatedDouble(message, field, j));
                    break;
                case google::protobuf::FieldDescriptor::CPPTYPE_STRING:
                    strings.push_back(reflection->GetRepeatedString(message, field, j));
                    break;
                default:
                    break;
                }
            }
            if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_STRING)
                op->params[key] = strings;
            else if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_FLOAT || field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE)
                op->params[key] = floats;
            else
                op->params[key] = ints;
        }
        else
        {
            if (!reflection->HasField(message, field) && !field->has_default_value())
                continue;
            switch (field->cpp_type())
            {
            case google::protobuf::FieldDescriptor::CPPTYPE_INT32:
                op->params[key] = reflection->GetInt32(message, field);
                break;
            case google::protobuf::FieldDescriptor::CPPTYPE_INT64:
                op->params[key] = (int64_t)reflection->GetInt64(message, field);
                break;
            case google::protobuf::FieldDescriptor::CPPTYPE_UINT32:
                op->params[key] = (int64_t)reflection->GetUInt32(message, field);
                break;
            case google::protobuf::FieldDescriptor::CPPTYPE_UINT64:
                op->params[key] = (int64_t)reflection->GetUInt64(message, field);
                break;
            case google::protobuf::FieldDescriptor::CPPTYPE_BOOL:
                op->params[key] = reflection->GetBool(message, field);
                break;
            case google::protobuf::FieldDescriptor::CPPTYPE_ENUM:
                op->params[key] = reflection->GetEnum(message, field)->number();
                break;
            case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
                op->params[key] = reflection->GetFloat(message, field);
                break;
            case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
                op->params[key] = reflection->GetDouble(message, field);
                break;
            case google::protobuf::FieldDescriptor::CPPTYPE_STRING:
                op->params[key] = reflection->GetString(message, field);
                break;
            default:
                break;
            }
        }
    }
    return 0;
}

static bool caffe_rule_matches(const caffe::NetStateRule& rule, const caffe::NetState& state)
{
    if (rule.has_phase() && rule.phase() != caffe::TEST)
        return false;
    if (rule.has_min_level() && state.level() < rule.min_level())
        return false;
    if (rule.has_max_level() && state.level() > rule.max_level())
        return false;
    std::set<std::string> stages(state.stage().begin(), state.stage().end());
    for (int i = 0; i < rule.stage_size(); i++)
        if (!stages.count(rule.stage(i)))
            return false;
    for (int i = 0; i < rule.not_stage_size(); i++)
        if (stages.count(rule.not_stage(i)))
            return false;
    return true;
}

static int load_caffe_blob(const caffe::BlobProto& blob, Attribute& attr)
{
    attr.type = 1;
    if (blob.has_shape())
    {
        for (int i = 0; i < blob.shape().dim_size(); i++)
        {
            if (blob.shape().dim(i) <= 0 || blob.shape().dim(i) > INT_MAX)
                return -1;
            attr.shape.push_back((int)blob.shape().dim(i));
        }
    }
    else
    {
        attr.shape = {blob.num(), blob.channels(), blob.height(), blob.width()};
    }
    // pnnx attributes store a scalar weight as a one-element tensor
    if (attr.shape.empty())
        attr.shape.push_back(1);
    size_t count = 1;
    for (int dim : attr.shape)
    {
        if (dim <= 0 || count > (size_t)INT_MAX / dim)
            return -1;
        count *= dim;
    }
    if ((blob.data_size() && blob.double_data_size()) || count != (size_t)(blob.data_size() + blob.double_data_size()))
        return -1;
    attr.data.resize(count * sizeof(float));
    for (size_t i = 0; i < count; i++)
    {
        float value = blob.data_size() ? blob.data(i) : (float)blob.double_data(i);
        memcpy(attr.data.data() + i * sizeof(float), &value, sizeof(float));
    }
    return 0;
}

int load_caffe(const std::string& caffeprotopath, Graph& g,
               const std::vector<std::vector<int64_t> >& input_shapes,
               const std::vector<std::string>& input_types)
{
    fprintf(stderr, "############# pass_level0 caffe\n");
    const std::string caffemodelpath = caffeprotopath.substr(0, caffeprotopath.size() - 9) + ".caffemodel";
    std::ifstream protofile(caffeprotopath.c_str(), std::ios::binary);
    if (!protofile)
    {
        fprintf(stderr, "open caffe prototxt %s failed\n", caffeprotopath.c_str());
        return -1;
    }
    std::string prototext((std::istreambuf_iterator<char>(protofile)), std::istreambuf_iterator<char>());
    caffe::NetParameter net;
    if (!google::protobuf::TextFormat::ParseFromString(prototext, &net))
    {
        fprintf(stderr, "parse caffe prototxt %s failed\n", caffeprotopath.c_str());
        return -1;
    }
    std::ifstream modelfile(caffemodelpath.c_str(), std::ios::binary);
    caffe::NetParameter weights;
    if (!modelfile || !weights.ParseFromIstream(&modelfile))
    {
        fprintf(stderr, "open or parse caffe weights %s failed\n", caffemodelpath.c_str());
        return -1;
    }
    if (net.layers_size() || weights.layers_size())
    {
        fprintf(stderr, "caffe deprecated V1 layers format is not supported yet\n");
        return -1;
    }
    std::map<std::string, const caffe::LayerParameter*> weight_layers;
    for (int i = 0; i < weights.layer_size(); i++)
    {
        const caffe::LayerParameter& layer = weights.layer(i);
        if (!weight_layers.insert(std::make_pair(layer.name(), &layer)).second)
        {
            fprintf(stderr, "duplicate caffe weight layer %s (%s)\n", layer.name().c_str(), layer.type().c_str());
            return -1;
        }
    }

    std::map<std::string, Operand*> blobs;
    std::set<std::string> blob_names;
    for (int i = 0; i < net.input_size(); i++)
        blob_names.insert(net.input(i));
    for (int i = 0; i < net.layer_size(); i++)
        for (int j = 0; j < net.layer(i).top_size(); j++)
            blob_names.insert(net.layer(i).top(j));
    std::vector<std::pair<std::string, std::vector<int64_t> > > input_defs;
    if ((net.input_shape_size() && net.input_shape_size() != net.input_size()) || (net.input_dim_size() && net.input_dim_size() != net.input_size() * 4) || (net.input_shape_size() && net.input_dim_size()))
    {
        fprintf(stderr, "invalid caffe net input shape declarations\n");
        return -1;
    }
    for (int i = 0; i < net.input_size(); i++)
    {
        std::vector<int64_t> shape;
        if (net.input_shape_size())
            shape.assign(net.input_shape(i).dim().begin(), net.input_shape(i).dim().end());
        else if (net.input_dim_size())
            for (int j = 0; j < 4; j++)
                shape.push_back(net.input_dim(i * 4 + j));
        input_defs.push_back(std::make_pair(net.input(i), shape));
    }

    std::vector<const caffe::LayerParameter*> layers;
    std::set<std::string> layer_names;
    for (int i = 0; i < net.layer_size(); i++)
    {
        const caffe::LayerParameter& layer = net.layer(i);
        bool include = layer.include_size() == 0;
        for (int j = 0; j < layer.include_size(); j++)
            include |= caffe_rule_matches(layer.include(j), net.state());
        for (int j = 0; j < layer.exclude_size(); j++)
            if (caffe_rule_matches(layer.exclude(j), net.state()))
                include = false;
        if (!include)
            continue;
        if (!layer_names.insert(layer.name()).second)
        {
            fprintf(stderr, "duplicate caffe layer %s (%s)\n", layer.name().c_str(), layer.type().c_str());
            return -1;
        }
        if (layer.type() == "Input" || layer.type() == "MemoryData")
        {
            if (layer.bottom_size() || !layer.top_size())
                return -1;
            for (int j = 0; j < layer.top_size(); j++)
            {
                std::vector<int64_t> shape;
                if (layer.type() == "Input")
                {
                    const caffe::InputParameter& p = layer.input_param();
                    if (p.shape_size() != 0 && p.shape_size() != 1 && p.shape_size() != layer.top_size())
                    {
                        fprintf(stderr, "caffe layer %s (Input) shape count mismatch\n", layer.name().c_str());
                        return -1;
                    }
                    if (p.shape_size())
                    {
                        const caffe::BlobShape& s = p.shape(p.shape_size() == 1 ? 0 : j);
                        shape.assign(s.dim().begin(), s.dim().end());
                    }
                }
                else
                {
                    const caffe::MemoryDataParameter& p = layer.memory_data_param();
                    if (j == 0)
                        shape = {p.batch_size(), p.channels(), p.height(), p.width()};
                    else if (j == 1)
                        shape = {p.batch_size()};
                    else
                        return -1;
                }
                input_defs.push_back(std::make_pair(layer.top(j), shape));
            }
            continue;
        }
        layers.push_back(&layer);
    }
    if ((!input_shapes.empty() && input_shapes.size() != input_defs.size()) || (!input_types.empty() && input_types.size() != input_defs.size()))
    {
        fprintf(stderr, "caffe input count mismatch, model expects %zu inputs\n", input_defs.size());
        return -1;
    }
    for (size_t i = 0; i < input_defs.size(); i++)
    {
        const std::string& name = input_defs[i].first;
        const std::vector<int64_t>& shape = input_shapes.empty() ? input_defs[i].second : input_shapes[i];
        if (blobs.count(name) || (!input_types.empty() && input_types[i] != "f32"))
        {
            fprintf(stderr, "duplicate caffe input or unsupported input type for %s (expected f32)\n", name.c_str());
            return -1;
        }
        std::string input_name = "input" + std::to_string(i);
        while (layer_names.count(input_name))
            input_name += "_input";
        layer_names.insert(input_name);
        Operator* op = g.new_operator("pnnx.Input", input_name);
        Operand* r = g.new_operand(name);
        r->producer = op;
        r->type = 1;
        r->params["__batch_index"] = 0;
        for (int64_t dim : shape)
        {
            if (dim <= 0 || dim > INT_MAX)
            {
                fprintf(stderr, "invalid caffe input shape for %s; provide inputshape\n", name.c_str());
                return -1;
            }
            r->shape.push_back((int)dim);
        }
        op->outputs.push_back(r);
        blobs[name] = r;
    }

    for (const caffe::LayerParameter* lp : layers)
    {
        const caffe::LayerParameter& layer = *lp;
        Operator* op = g.new_operator("caffe." + layer.type(), layer.name());
        for (int j = 0; j < layer.bottom_size(); j++)
        {
            auto it = blobs.find(layer.bottom(j));
            if (it == blobs.end())
            {
                fprintf(stderr, "caffe layer %s (%s) missing bottom %s\n", layer.name().c_str(), layer.type().c_str(), layer.bottom(j).c_str());
                return -1;
            }
            op->inputs.push_back(it->second);
            it->second->consumers.push_back(op);
        }
        std::set<std::string> tops;
        for (int j = 0; j < layer.top_size(); j++)
        {
            const std::string& name = layer.top(j);
            if (!tops.insert(name).second)
            {
                fprintf(stderr, "caffe layer %s (%s) has repeated top %s\n", layer.name().c_str(), layer.type().c_str(), name.c_str());
                return -1;
            }
            auto it = blobs.find(name);
            if (it != blobs.end())
            {
                bool inplace = false;
                for (int k = 0; k < layer.bottom_size(); k++)
                    inplace |= layer.bottom(k) == name;
                if (!inplace)
                {
                    fprintf(stderr, "caffe layer %s (%s) duplicates top %s\n", layer.name().c_str(), layer.type().c_str(), name.c_str());
                    return -1;
                }
                std::string version_name = name + "_caffe_" + std::to_string(g.operands.size());
                while (blob_names.count(version_name))
                    version_name += "_inplace";
                blob_names.insert(version_name);
                it->second->name = version_name;
            }
            Operand* r = g.new_operand(name);
            r->producer = op;
            r->type = 1;
            op->outputs.push_back(r);
            blobs[name] = r;
        }
        const google::protobuf::Descriptor* descriptor = layer.GetDescriptor();
        const google::protobuf::Reflection* reflection = layer.GetReflection();
        std::string parameter_type = layer.type();
        if (parameter_type == "Deconvolution" || parameter_type == "ConvolutionDepthwise" || parameter_type == "DepthwiseConvolution")
            parameter_type = "Convolution";
        else if (parameter_type == "ReLU6")
            parameter_type = "ReLU";
        else if (parameter_type == "LSTM")
            parameter_type = "Recurrent";
        parameter_type += "Parameter";
        for (int j = 0; j < descriptor->field_count(); j++)
        {
            const google::protobuf::FieldDescriptor* field = descriptor->field(j);
            if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE && !field->is_repeated()
                    && field->message_type()->name() == parameter_type && reflection->HasField(layer, field))
            {
                if (load_caffe_parameters(reflection->GetMessage(layer, field), "", op) != 0)
                    return -1;
                break;
            }
        }
        if (layer.type() == "Concat" && layer.has_concat_param() && layer.concat_param().has_concat_dim() && !layer.concat_param().has_axis())
            op->params.erase("axis");
        if (layer.type() == "Slice" && layer.has_slice_param() && layer.slice_param().has_slice_dim() && !layer.slice_param().has_axis())
            op->params.erase("axis");
        const caffe::LayerParameter* wl = &layer;
        auto wit = weight_layers.find(layer.name());
        if (wit != weight_layers.end())
        {
            wl = wit->second;
            if (wl->type() != layer.type())
            {
                fprintf(stderr, "caffe layer %s (%s) weight layer type mismatch %s\n", layer.name().c_str(), layer.type().c_str(), wl->type().c_str());
                return -1;
            }
        }
        for (int j = 0; j < wl->blobs_size(); j++)
        {
            Attribute attr;
            if (load_caffe_blob(wl->blobs(j), attr) != 0)
            {
                fprintf(stderr, "caffe layer %s (%s) invalid weight blob %d\n", layer.name().c_str(), layer.type().c_str(), j);
                return -1;
            }
            op->attrs[std::to_string(j)] = attr;
        }
    }

    // resolve outputs before eliminating Silence and inplace identities
    const std::vector<Operand*> operands = g.operands;
    int output_index = 0;
    for (Operand* r : operands)
    {
        if (!r->consumers.empty())
            continue;
        bool final_version = false;
        for (const auto& it : blobs)
            final_version |= it.second == r;
        if (!final_version)
            continue;
        std::string output_name = "output" + std::to_string(output_index++);
        while (layer_names.count(output_name))
            output_name += "_output";
        layer_names.insert(output_name);
        Operator* op = g.new_operator("pnnx.Output", output_name);
        op->inputs.push_back(r);
        r->consumers.push_back(op);
    }
    if (output_index == 0)
    {
        fprintf(stderr, "caffe inference graph has no outputs\n");
        return -1;
    }
    for (auto it = g.ops.begin(); it != g.ops.end();)
    {
        Operator* op = *it;
        if (op->type != "caffe.Silence")
        {
            ++it;
            continue;
        }
        for (Operand* r : op->inputs)
            r->remove_consumer(op);
        it = g.ops.erase(it);
        delete op;
    }
    if (pass_caffe(g) != 0)
        return -1;

    // distinct Caffe names may become the same Python identifier after sanitizing
    for (size_t i = 0; i < g.ops.size(); i++)
        g.ops[i]->name = caffe_identifier(g.ops[i]->name, i);
    for (size_t i = 0; i < g.operands.size(); i++)
        g.operands[i]->name = caffe_identifier(g.operands[i]->name, i);

    return 0;
}

} // namespace pnnx
