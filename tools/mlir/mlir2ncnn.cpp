// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <stdio.h>

#include <map>
#include <set>

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "tf_dialect.h"
#include "ncnn_dialect.h"

static std::string get_mlir_value_uniq_id(const mlir::Value& value)
{
    if (value.getLoc().isa<mlir::FileLineColLoc>())
    {
        mlir::FileLineColLoc floc = value.getLoc().cast<mlir::FileLineColLoc>();

        return floc.getFilename().str() + ":" + std::to_string(floc.getLine()) + ":" + std::to_string(floc.getColumn());
    }

    if (value.getLoc().isa<mlir::FusedLoc>())
    {
        mlir::FileLineColLoc floc = value.getLoc().cast<mlir::FusedLoc>().getLocations().front().cast<mlir::FileLineColLoc>();

        return floc.getFilename().str() + ":" + std::to_string(floc.getLine()) + ":" + std::to_string(floc.getColumn());
    }

    fprintf(stderr, "unhandled get_mlir_value_uniq_id\n");
    return std::string();
}

static std::string get_attr_s(const mlir::Attribute& attr)
{
    std::string s;

    if (attr.isa<mlir::StringAttr>())
    {
        mlir::StringAttr a = attr.cast<mlir::StringAttr>();

        s = a.getValue().str();
    }

    return s;
}

static int get_attr_b(const mlir::Attribute& attr)
{
    int i;

    if (attr.isa<mlir::BoolAttr>())
    {
        mlir::BoolAttr a = attr.cast<mlir::BoolAttr>();

        i = a.getValue() ? 1 : 0;
    }
    else
    {
        fprintf(stderr, "not BoolAttr\n");
    }

    return i;
}

static int get_attr_i(const mlir::Attribute& attr)
{
    int i;

    if (attr.isa<mlir::IntegerAttr>())
    {
        mlir::IntegerAttr a = attr.cast<mlir::IntegerAttr>();

        i = (int)a.getInt();
    }
    else
    {
        fprintf(stderr, "not IntegerAttr\n");
    }

    return i;
}

static float get_attr_f(const mlir::Attribute& attr)
{
    float f;

    if (attr.isa<mlir::FloatAttr>())
    {
        mlir::FloatAttr a = attr.cast<mlir::FloatAttr>();

        f = (float)a.getValueAsDouble();
    }
    else
    {
        fprintf(stderr, "not FloatAttr\n");
    }

    return f;
}

static std::vector<int> get_attr_ai(const mlir::Attribute& attr)
{
    std::vector<int> v;

    if (attr.isa<mlir::ArrayAttr>())
    {
        mlir::ArrayAttr a = attr.cast<mlir::ArrayAttr>();

        const int array_size = a.getValue().size();

        v.resize(array_size);
        for (int j = 0; j < array_size; j++)
        {
            if (a[j].isa<mlir::IntegerAttr>())
            {
                int64_t ii = a[j].cast<mlir::IntegerAttr>().getInt();
                v[j] = std::max(std::min(ii, (int64_t)INT_MAX), (int64_t)INT_MIN);
            }
        }
    }
    else if (attr.isa<mlir::DenseIntElementsAttr>())
    {
        mlir::DenseIntElementsAttr ai = attr.cast<mlir::DenseIntElementsAttr>();

        for (auto ii : ai.getIntValues())
        {
            v.push_back(ii.getSExtValue());
        }
    }
    else
    {
        fprintf(stderr, "not ArrayAttr or DenseIntElementsAttr\n");
    }

    return v;
}

static std::vector<float> get_attr_af(const mlir::Attribute& attr)
{
    std::vector<float> v;

    if (attr.isa<mlir::ArrayAttr>())
    {
        mlir::ArrayAttr a = attr.cast<mlir::ArrayAttr>();

        const int array_size = a.getValue().size();

        v.resize(array_size);
        for (int j = 0; j < array_size; j++)
        {
            if (a[j].isa<mlir::FloatAttr>())
            {
                double ff = a[j].cast<mlir::FloatAttr>().getValueAsDouble();
                v[j] = ff;
            }
        }
    }
    else if (attr.isa<mlir::DenseFPElementsAttr>())
    {
        mlir::DenseFPElementsAttr af = attr.cast<mlir::DenseFPElementsAttr>();

        for (auto ff : af.getFloatValues())
        {
            v.push_back(ff.convertToFloat());
        }
    }
    else
    {
        fprintf(stderr, "not ArrayAttr or DenseFPElementsAttr\n");
    }

    return v;
}

static std::string get_operation_attr_s(const mlir::Operation& _operation, const char* key)
{
    mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

    mlir::Attribute attr = operation.getAttr(key);

    return get_attr_s(attr);
}

static int get_operation_attr_b(const mlir::Operation& _operation, const char* key)
{
    mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

    mlir::Attribute attr = operation.getAttr(key);

    return get_attr_b(attr);
}

static int get_operation_attr_i(const mlir::Operation& _operation, const char* key)
{
    mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

    mlir::Attribute attr = operation.getAttr(key);

    return get_attr_i(attr);
}

static float get_operation_attr_f(const mlir::Operation& _operation, const char* key)
{
    mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

    mlir::Attribute attr = operation.getAttr(key);

    return get_attr_f(attr);
}

static std::vector<int> get_operation_attr_ai(const mlir::Operation& _operation, const char* key)
{
    mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

    mlir::Attribute attr = operation.getAttr(key);

    return get_attr_ai(attr);
}

static std::vector<float> get_operation_attr_af(const mlir::Operation& _operation, const char* key)
{
    mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

    mlir::Attribute attr = operation.getAttr(key);

    return get_attr_af(attr);
}

int main(int argc, char** argv)
{
    const char* mlirpath = argv[1];
    const char* ncnn_prototxt = argc >= 4 ? argv[2] : "ncnn.param";
    const char* ncnn_modelbin = argc >= 4 ? argv[3] : "ncnn.bin";

    mlir::MLIRContext context;

    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::TF::TensorFlowDialect>();
    context.getOrLoadDialect<mlir::ncnn::NCNNDialect>();

    mlir::OwningModuleRef m = mlir::parseSourceFile(mlirpath, &context);

    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Add a run of the canonicalizer to optimize the mlir module.
    pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    pm.run(*m);

    //     m->dump();

    mlir::FuncOp main_fn = m->lookupSymbol<mlir::FuncOp>("main");

    auto& bb = main_fn.getBlocks().front();

    //     bb.dump();

    FILE* pp = fopen(ncnn_prototxt, "wb");
    FILE* bp = fopen(ncnn_modelbin, "wb");

    // node reference
    std::map<std::string, int> node_reference;

    // weight node and weight reshape node
    std::map<std::string, mlir::Attribute> weights;

    // weight node before BinaryOp
    std::map<std::string, mlir::Attribute> binaryop_weights;

    fprintf(pp, "7767517\n");

    const mlir::Block::OpListType& operations = bb.getOperations();

    int node_count = operations.size();

    // global definition line
    // [layer count] [blob count]
    std::set<std::string> blob_names;
    for (const mlir::Operation& _operation : operations)
    {
        mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

        std::string op = operation.getName().getStringRef().str();

        int num_input = (int)operation.getNumOperands();
        int num_output = (int)operation.getNumResults();

        if (op == "tf.Const")
        {
            // weight
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(0));
            weights[output_name] = operation.getAttr("value");
            continue;
        }
        else
        {
            bool isBinaryOp = false;
            // TODO add more binaryop
            if (op == "tf.BiasAdd" || op == "tf.AddV2" || op == "tf.Sub" || op == "tf.Maximum" || op == "tf.Minimum" || op == "tf.Mul")
            {
                isBinaryOp = true;
            }

            if (isBinaryOp)
            {
                // check weights
                for (int j = 0; j < num_input; j++)
                {
                    std::string input_name = get_mlir_value_uniq_id(operation.getOperand(j));

                    std::map<std::string, mlir::Attribute>::iterator it = weights.find(input_name);
                    if (it != weights.end())
                    {
                        // binary op with weight, insert MemoryData layer and const blob
                        binaryop_weights[input_name] = it->second;
                        weights.erase(it);
                    }
                }
            }
        }

        for (int j = 0; j < num_input; j++)
        {
            std::string input_name = get_mlir_value_uniq_id(operation.getOperand(j));

            // check weight
            if (weights.find(input_name) != weights.end())
            {
                continue;
            }

            blob_names.insert(input_name);

            if (node_reference.find(input_name) == node_reference.end())
            {
                node_reference[input_name] = 1;
            }
            else
            {
                node_reference[input_name] = node_reference[input_name] + 1;
            }
        }

        for (int j = 0; j < num_output; j++)
        {
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(j));

            blob_names.insert(output_name);
        }
    }

    // remove node_reference entry with reference equals to one
    int splitncnn_blob_count = 0;
    std::map<std::string, int>::iterator it = node_reference.begin();
    while (it != node_reference.end())
    {
        if (it->second == 1)
        {
            node_reference.erase(it++);
        }
        else
        {
            splitncnn_blob_count += it->second;
            //             fprintf(stderr, "%s %d\n", it->first.c_str(), it->second);
            ++it;
        }
    }

    fprintf(pp, "%lu %lu\n", node_count + node_reference.size() - weights.size(), blob_names.size() + splitncnn_blob_count);

    int internal_split = 0;

    // model op
    int g_opid = 0;

    for (const mlir::Operation& _operation : operations)
    {
        mlir::Operation& operation = const_cast<mlir::Operation&>(_operation);

        std::string op = operation.getName().getStringRef().str();

        int opid = g_opid++;

        int num_input = (int)operation.getNumOperands();
        int num_output = (int)operation.getNumResults();

        for (int i = 0; i < (int)operation.getNumOperands(); i++)
        {
            std::string input_name = get_mlir_value_uniq_id(operation.getOperand(i));

            // check weight
            if (weights.find(input_name) != weights.end())
            {
                num_input--;
            }
        }

        if (op == "std.return")
        {
            fprintf(pp, "%-16s", "Noop");
        }
        else if (op == "ncnn.BinaryOp")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (op == "ncnn.KerasConv2D")
        {
            fprintf(pp, "%-16s", "Convolution");
        }
        else if (op == "ncnn.KerasDense")
        {
            fprintf(pp, "%-16s", "InnerProduct");
        }
        else if (op == "ncnn.KerasBatchNorm")
        {
            fprintf(pp, "%-16s", "BatchNorm");
        }
        else if (op == "ncnn.InstanceNorm")
        {
            fprintf(pp, "%-16s", "InstanceNorm");
        }
        else if (op == "ncnn.InstanceNormAffine")
        {
            fprintf(pp, "%-16s", "InstanceNorm");
        }
        else if (op == "tf.AddN")
        {
            fprintf(pp, "%-16s", "Eltwise");
        }
        else if (op == "tf.AddV2")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (op == "tf.AvgPool")
        {
            fprintf(pp, "%-16s", "Pooling");
        }
        else if (op == "tf.BiasAdd")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (op == "tf.ConcatV2")
        {
            fprintf(pp, "%-16s", "Concat");
        }
        else if (op == "tf.Const")
        {
            // check weight before BinaryOp
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(0));
            if (binaryop_weights.find(output_name) != binaryop_weights.end())
            {
                fprintf(pp, "%-16s", "MemoryData");
            }
            else
            {
                continue;
            }
        }
        else if (op == "tf.Conv2D")
        {
            fprintf(pp, "%-16s", "Convolution");
        }
        else if (op == "tf.Conv2DBackpropInput")
        {
            fprintf(pp, "%-16s", "Deconvolution");
        }
        else if (op == "tf.DepthwiseConv2dNative")
        {
            fprintf(pp, "%-16s", "ConvolutionDepthWise");
        }
        else if (op == "tf.Identity")
        {
            fprintf(pp, "%-16s", "Noop");
        }
        else if (op == "tf.LeakyRelu")
        {
            fprintf(pp, "%-16s", "ReLU");
        }
        else if (op == "tf.MatMul")
        {
            int transpose_a = get_operation_attr_b(operation, "transpose_a");
            int transpose_b = get_operation_attr_b(operation, "transpose_b");

            if (transpose_a == 0 && transpose_b == 1)
            {
                // InnerProduct-like A * B + C
                fprintf(pp, "%-16s", "InnerProduct");
            }
            else
            {
                fprintf(pp, "%-16s", "Gemm");
            }
        }
        else if (op == "tf.Maximum")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (op == "tf.MaxPool")
        {
            fprintf(pp, "%-16s", "Pooling");
        }
        else if (op == "tf.Mean")
        {
            std::string reduction_indices_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& R = weights[reduction_indices_name];

            std::vector<int> v = get_attr_ai(R);

            int keep_dims = get_operation_attr_b(operation, "keep_dims");

            if (keep_dims == 0 && v.size() == 2 && v[0] == 1 && v[1] == 2)
            {
                // global avg pooling style nhwc -> nc
                fprintf(pp, "%-16s", "Pooling");
            }
            else
            {
                fprintf(stderr, "tf.Mean is not global avg pooling\n");
                fprintf(pp, "%-16s", "Reduction");
            }
        }
        else if (op == "tf.Minimum")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (op == "tf.Mul")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (op == "tf.Pad")
        {
            fprintf(pp, "%-16s", "Padding");
        }
        else if (op == "tf.Placeholder")
        {
            fprintf(pp, "%-16s", "Input");
        }
        else if (op == "tf.Relu")
        {
            fprintf(pp, "%-16s", "ReLU");
        }
        else if (op == "tf.Relu6")
        {
            fprintf(pp, "%-16s", "Clip");
        }
        else if (op == "tf.Reshape")
        {
            fprintf(pp, "%-16s", "Reshape");
        }
        else if (op == "tf.ResizeBilinear")
        {
            fprintf(pp, "%-16s", "Interp");
        }
        else if (op == "tf.ResizeNearestNeighbor")
        {
            fprintf(pp, "%-16s", "Interp");
        }
        else if (op == "tf.Sigmoid")
        {
            fprintf(pp, "%-16s", "Sigmoid");
        }
        else if (op == "tf.Softmax")
        {
            fprintf(pp, "%-16s", "Softmax");
        }
        else if (op == "tf.StridedSlice")
        {
            fprintf(pp, "%-16s", "Crop");
        }
        else if (op == "tf.Sub")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (op == "tf.Tanh")
        {
            fprintf(pp, "%-16s", "TanH");
        }
        else
        {
            // TODO
            fprintf(stderr, "%s not supported yet!\n", op.c_str());
            fprintf(pp, "%-16s", op.c_str());
        }

        fprintf(pp, " op_%d %d %d", opid, num_input, num_output);

        for (int i = 0; i < (int)operation.getNumOperands(); i++)
        {
            std::string input_name = get_mlir_value_uniq_id(operation.getOperand(i));

            // check weight
            if (weights.find(input_name) != weights.end())
            {
                continue;
            }

            if (node_reference.find(input_name) != node_reference.end())
            {
                int refidx = node_reference[input_name] - 1;
                node_reference[input_name] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                input_name = input_name + splitsuffix;
            }

            fprintf(pp, " %s", input_name.c_str());
        }

        for (int i = 0; i < num_output; i++)
        {
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(i));
            fprintf(pp, " %s", output_name.c_str());
        }

        if (op == "std.return")
        {
        }
        else if (op == "ncnn.BinaryOp")
        {
            int op_type = get_operation_attr_i(operation, "op_type");
            int with_scalar = get_operation_attr_i(operation, "with_scalar");
            float b = get_operation_attr_f(operation, "b");

            fprintf(pp, " 0=%d", op_type);
            fprintf(pp, " 1=%d", with_scalar);
            fprintf(pp, " 2=%e", b);
        }
        else if (op == "ncnn.KerasConv2D")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            std::string bias_name = get_mlir_value_uniq_id(operation.getOperand(2));
            const mlir::Attribute& W = weights[weight_name];
            const mlir::Attribute& B = weights[bias_name];

            llvm::ArrayRef<int64_t> shape = W.getType().cast<mlir::RankedTensorType>().getShape();

            //             assert(shape.size() == 4)

            // kh-kw-inch-outch
            int kernel_size_h = shape[0];
            int kernel_size_w = shape[1];
            int num_input = shape[2];
            int num_output = shape[3];
            int weight_data_size = kernel_size_h * kernel_size_w * num_input * num_output;

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 11=%d", kernel_size_h);
            fprintf(pp, " 6=%d", weight_data_size);

            std::vector<int> dilations = get_operation_attr_ai(operation, "dilations");
            std::vector<int> strides = get_operation_attr_ai(operation, "strides");
            std::string padding = get_operation_attr_s(operation, "padding");

            if (dilations.size() == 4)
            {
                fprintf(pp, " 2=%d", dilations[2]);
                fprintf(pp, " 12=%d", dilations[1]);
            }

            if (strides.size() == 4)
            {
                fprintf(pp, " 3=%d", strides[2]);
                fprintf(pp, " 13=%d", strides[1]);
            }

            if (padding == "EXPLICIT")
            {
                // nhwc = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
                std::vector<int> explicit_paddings = get_operation_attr_ai(operation, "explicit_paddings");

                fprintf(pp, " 4=%d", explicit_paddings[4]);
                fprintf(pp, " 15=%d", explicit_paddings[5]);
                fprintf(pp, " 14=%d", explicit_paddings[2]);
                fprintf(pp, " 16=%d", explicit_paddings[3]);
            }
            else if (padding == "VALID")
            {
                fprintf(pp, " 4=%d", 0);
            }
            else if (padding == "SAME")
            {
                fprintf(pp, " 4=%d", -233);
            }

            fprintf(pp, " 5=1"); // bias_term

            std::vector<float> v = get_attr_af(W);
            std::vector<float> bv = get_attr_af(B);

            // reorder h-w-i-o to o-i-h-w
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                float tmp;
                for (int p = 0; p < num_output; p++)
                {
                    for (int q = 0; q < num_input; q++)
                    {
                        for (int i = 0; i < kernel_size_h; i++)
                        {
                            for (int j = 0; j < kernel_size_w; j++)
                            {
                                tmp = v[i * kernel_size_w * num_input * num_output + j * num_input * num_output + q * num_output + p];
                                fwrite(&tmp, sizeof(float), 1, bp);
                            }
                        }
                    }
                }
            }

            fwrite(bv.data(), sizeof(float), bv.size(), bp);
        }
        else if (op == "ncnn.KerasDense")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            std::string bias_name = get_mlir_value_uniq_id(operation.getOperand(2));
            const mlir::Attribute& W = weights[weight_name];
            const mlir::Attribute& B = weights[bias_name];

            llvm::ArrayRef<int64_t> shape = W.getType().cast<mlir::RankedTensorType>().getShape();

            //             assert(shape.size() == 2)

            // inch-outch
            int num_input = shape[0];
            int num_output = shape[1];
            int weight_data_size = shape[0] * shape[1];

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=1"); // bias_term
            fprintf(pp, " 2=%d", weight_data_size);

            std::vector<float> v = get_attr_af(W);
            std::vector<float> bv = get_attr_af(B);

            // reorder i-o to o-i
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                float tmp;
                for (int p = 0; p < num_output; p++)
                {
                    for (int q = 0; q < num_input; q++)
                    {
                        tmp = v[q * num_output + p];
                        fwrite(&tmp, sizeof(float), 1, bp);
                    }
                }
            }

            fwrite(bv.data(), sizeof(float), bv.size(), bp);
        }
        else if (op == "ncnn.KerasBatchNorm")
        {
            std::string gamma_name = get_mlir_value_uniq_id(operation.getOperand(1));
            std::string bias_name = get_mlir_value_uniq_id(operation.getOperand(2));
            const mlir::Attribute& W = weights[gamma_name];
            const mlir::Attribute& B = weights[bias_name];

            std::vector<float> v = get_attr_af(W);
            std::vector<float> bv = get_attr_af(B);

            int channels = v.size();

            fprintf(pp, " 0=%d", channels);

            std::vector<float> mean(channels, 0.f);
            std::vector<float> var(channels, 1.f);

            fwrite(v.data(), sizeof(float), channels, bp);
            fwrite(mean.data(), sizeof(float), channels, bp);
            fwrite(var.data(), sizeof(float), channels, bp);
            fwrite(bv.data(), sizeof(float), channels, bp);
        }
        else if (op == "ncnn.InstanceNorm")
        {
            float eps = get_operation_attr_f(operation, "epsilon");

            fprintf(pp, " 0=0"); // channels
            fprintf(pp, " 1=%e", eps);
            fprintf(pp, " 2=0"); // affine
        }
        else if (op == "ncnn.InstanceNormAffine")
        {
            float eps = get_operation_attr_f(operation, "epsilon");

            std::string gamma_name = get_mlir_value_uniq_id(operation.getOperand(1));
            std::string beta_name = get_mlir_value_uniq_id(operation.getOperand(2));
            const mlir::Attribute& G = weights[gamma_name];
            const mlir::Attribute& B = weights[beta_name];

            std::vector<float> gv = get_attr_af(G);
            std::vector<float> bv = get_attr_af(B);

            int channels = gv.size();

            fprintf(pp, " 0=%d", channels);
            fprintf(pp, " 1=%e", eps);
            fprintf(pp, " 2=1"); // affine

            fwrite(gv.data(), sizeof(float), gv.size(), bp);
            fwrite(bv.data(), sizeof(float), bv.size(), bp);
        }
        else if (op == "tf.AddN")
        {
            int op_type = 1;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (op == "tf.AddV2")
        {
            int op_type = 0;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (op == "tf.AvgPool")
        {
            std::vector<int> ksize = get_operation_attr_ai(operation, "ksize");
            std::vector<int> strides = get_operation_attr_ai(operation, "strides");
            std::string padding = get_operation_attr_s(operation, "padding");

            fprintf(pp, " 0=1"); // avg pool

            if (ksize.size() == 4)
            {
                fprintf(pp, " 1=%d", ksize[2]);
                fprintf(pp, " 11=%d", ksize[1]);
            }

            if (strides.size() == 4)
            {
                fprintf(pp, " 2=%d", strides[2]);
                fprintf(pp, " 12=%d", strides[1]);
            }

            int pad_mode = 1;
            if (padding == "VALID")
            {
                pad_mode = 1;
            }
            else if (padding == "SAME")
            {
                pad_mode = 2;
            }

            fprintf(pp, " 5=%d", pad_mode);
        }
        else if (op == "tf.ConcatV2")
        {
            std::string axis_name = get_mlir_value_uniq_id(operation.getOperand(operation.getNumOperands() - 1));
            const mlir::Attribute& A = weights[axis_name];

            int axis = get_attr_ai(A)[0];

            // axis nhc to nhw
            // axis nhwc to nchw
            int dims = operation.getOperand(0).getType().cast<mlir::RankedTensorType>().getShape().size();

            if (dims == 2 && axis == 1)
            {
                axis = 0;
            }
            if (dims == 3 && axis == 1)
            {
                axis = 1;
            }
            if (dims == 3 && axis == 2)
            {
                axis = 0;
            }
            if (dims == 4 && axis == 1)
            {
                axis = 1;
            }
            if (dims == 4 && axis == 2)
            {
                axis = 2;
            }
            if (dims == 4 && axis == 3)
            {
                axis = 0;
            }

            fprintf(pp, " 0=%d", axis);
        }
        else if (op == "tf.Const")
        {
            // check weight before BinaryOp
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(0));
            if (binaryop_weights.find(output_name) != binaryop_weights.end())
            {
                const mlir::Attribute& M = binaryop_weights[output_name];

                llvm::ArrayRef<int64_t> shape = M.getType().cast<mlir::RankedTensorType>().getShape();

                // c wc hwc
                if (shape.size() == 0)
                {
                    // scalar
                    fprintf(pp, " 0=1");
                }
                else if (shape.size() == 1)
                {
                    fprintf(pp, " 0=%d", (int)shape[0]);
                }
                else if (shape.size() == 2)
                {
                    fprintf(pp, " 0=%d", (int)shape[1]);
                    fprintf(pp, " 1=%d", (int)shape[0]);
                }
                else if (shape.size() == 3)
                {
                    fprintf(pp, " 0=%d", (int)shape[1]);
                    fprintf(pp, " 1=%d", (int)shape[0]);
                    fprintf(pp, " 2=%d", (int)shape[2]);
                }

                std::vector<float> v = get_attr_af(M);

                if (shape.size() != 3)
                {
                    fwrite(v.data(), sizeof(float), v.size(), bp);
                }
                else
                {
                    int w = (int)shape[1];
                    int h = (int)shape[0];
                    int c = (int)shape[2];

                    float tmp;
                    // h-w-c to c-h-w
                    for (int p = 0; p < c; p++)
                    {
                        for (int i = 0; i < h; i++)
                        {
                            for (int j = 0; j < w; j++)
                            {
                                tmp = v[i * w * c + j * c + p];
                                fwrite(&tmp, sizeof(float), 1, bp);
                            }
                        }
                    }
                }
            }
        }
        else if (op == "tf.Conv2D")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& W = weights[weight_name];

            llvm::ArrayRef<int64_t> shape = W.getType().cast<mlir::RankedTensorType>().getShape();

            //             assert(shape.size() == 4)

            // kh-kw-inch-outch
            int kernel_size_h = shape[0];
            int kernel_size_w = shape[1];
            int num_input = shape[2];
            int num_output = shape[3];
            int weight_data_size = kernel_size_h * kernel_size_w * num_input * num_output;

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 11=%d", kernel_size_h);
            fprintf(pp, " 6=%d", weight_data_size);

            std::vector<int> dilations = get_operation_attr_ai(operation, "dilations");
            std::vector<int> strides = get_operation_attr_ai(operation, "strides");
            std::string padding = get_operation_attr_s(operation, "padding");

            if (dilations.size() == 4)
            {
                fprintf(pp, " 2=%d", dilations[2]);
                fprintf(pp, " 12=%d", dilations[1]);
            }

            if (strides.size() == 4)
            {
                fprintf(pp, " 3=%d", strides[2]);
                fprintf(pp, " 13=%d", strides[1]);
            }

            if (padding == "EXPLICIT")
            {
                // nhwc = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
                std::vector<int> explicit_paddings = get_operation_attr_ai(operation, "explicit_paddings");

                fprintf(pp, " 4=%d", explicit_paddings[4]);
                fprintf(pp, " 15=%d", explicit_paddings[5]);
                fprintf(pp, " 14=%d", explicit_paddings[2]);
                fprintf(pp, " 16=%d", explicit_paddings[3]);
            }
            else if (padding == "VALID")
            {
                fprintf(pp, " 4=%d", 0);
            }
            else if (padding == "SAME")
            {
                fprintf(pp, " 4=%d", -233);
            }

            std::vector<float> v = get_attr_af(W);

            // reorder h-w-i-o to o-i-h-w
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                float tmp;
                for (int p = 0; p < num_output; p++)
                {
                    for (int q = 0; q < num_input; q++)
                    {
                        for (int i = 0; i < kernel_size_h; i++)
                        {
                            for (int j = 0; j < kernel_size_w; j++)
                            {
                                tmp = v[i * kernel_size_w * num_input * num_output + j * num_input * num_output + q * num_output + p];
                                fwrite(&tmp, sizeof(float), 1, bp);
                            }
                        }
                    }
                }
            }
        }
        else if (op == "tf.Conv2DBackpropInput")
        {
            std::string output_shape_name = get_mlir_value_uniq_id(operation.getOperand(0));
            const std::vector<int> output_shape = get_attr_ai(weights[output_shape_name]);

            //             assert(output_shape.size() == 4)

            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& W = weights[weight_name];

            llvm::ArrayRef<int64_t> shape = W.getType().cast<mlir::RankedTensorType>().getShape();

            //             assert(shape.size() == 4)

            // kh-kw-outch-inch
            int kernel_size_h = shape[0];
            int kernel_size_w = shape[1];
            int num_output = shape[2];
            int num_input = shape[3];
            int weight_data_size = kernel_size_h * kernel_size_w * num_input * num_output;

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 11=%d", kernel_size_h);
            fprintf(pp, " 6=%d", weight_data_size);

            std::vector<int> dilations = get_operation_attr_ai(operation, "dilations");
            std::vector<int> strides = get_operation_attr_ai(operation, "strides");
            std::string padding = get_operation_attr_s(operation, "padding");

            if (dilations.size() == 4)
            {
                fprintf(pp, " 2=%d", dilations[2]);
                fprintf(pp, " 12=%d", dilations[1]);
            }

            if (strides.size() == 4)
            {
                fprintf(pp, " 3=%d", strides[2]);
                fprintf(pp, " 13=%d", strides[1]);
            }

            if (padding == "EXPLICIT")
            {
                // nhwc = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
                std::vector<int> explicit_paddings = get_operation_attr_ai(operation, "explicit_paddings");

                fprintf(pp, " 4=%d", explicit_paddings[4]);
                fprintf(pp, " 15=%d", explicit_paddings[5]);
                fprintf(pp, " 14=%d", explicit_paddings[2]);
                fprintf(pp, " 16=%d", explicit_paddings[3]);
            }
            else if (padding == "VALID")
            {
                fprintf(pp, " 4=%d", 0);
            }
            else if (padding == "SAME")
            {
                fprintf(pp, " 4=%d", -233);

                fprintf(pp, " 20=%d", output_shape[2]);
                fprintf(pp, " 21=%d", output_shape[1]);
            }

            std::vector<float> v = get_attr_af(W);

            // reorder h-w-o-i to o-i-h-w
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                float tmp;
                for (int p = 0; p < num_output; p++)
                {
                    for (int q = 0; q < num_input; q++)
                    {
                        for (int i = 0; i < kernel_size_h; i++)
                        {
                            for (int j = 0; j < kernel_size_w; j++)
                            {
                                tmp = v[i * kernel_size_w * num_output * num_input + j * num_output * num_input + p * num_input + q];
                                fwrite(&tmp, sizeof(float), 1, bp);
                            }
                        }
                    }
                }
            }
        }
        else if (op == "tf.DepthwiseConv2dNative")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& W = weights[weight_name];

            llvm::ArrayRef<int64_t> shape = W.getType().cast<mlir::RankedTensorType>().getShape();

            //             assert(shape.size() == 4)

            // kh-kw-inch-cm
            int kernel_size_h = shape[0];
            int kernel_size_w = shape[1];
            int num_input = shape[2];
            int channel_multiplier = shape[3];

            int num_output = num_input * channel_multiplier;
            int group = num_input;

            int weight_data_size = kernel_size_h * kernel_size_w * num_input * channel_multiplier;

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 11=%d", kernel_size_h);
            fprintf(pp, " 6=%d", weight_data_size);
            fprintf(pp, " 7=%d", group);

            std::vector<int> dilations = get_operation_attr_ai(operation, "dilations");
            std::vector<int> strides = get_operation_attr_ai(operation, "strides");
            std::string padding = get_operation_attr_s(operation, "padding");

            if (dilations.size() == 4)
            {
                fprintf(pp, " 2=%d", dilations[2]);
                fprintf(pp, " 12=%d", dilations[1]);
            }

            if (strides.size() == 4)
            {
                fprintf(pp, " 3=%d", strides[2]);
                fprintf(pp, " 13=%d", strides[1]);
            }

            if (padding == "EXPLICIT")
            {
                // nhwc = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
                std::vector<int> explicit_paddings = get_operation_attr_ai(operation, "explicit_paddings");

                fprintf(pp, " 4=%d", explicit_paddings[4]);
                fprintf(pp, " 15=%d", explicit_paddings[5]);
                fprintf(pp, " 14=%d", explicit_paddings[2]);
                fprintf(pp, " 16=%d", explicit_paddings[3]);
            }
            else if (padding == "VALID")
            {
                fprintf(pp, " 4=%d", 0);
            }
            else if (padding == "SAME")
            {
                fprintf(pp, " 4=%d", -233);
            }

            std::vector<float> v = get_attr_af(W);

            // reorder h-w-i-cm to i-cm-h-w
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                float tmp;
                for (int p = 0; p < num_input; p++)
                {
                    for (int q = 0; q < channel_multiplier; q++)
                    {
                        for (int i = 0; i < kernel_size_h; i++)
                        {
                            for (int j = 0; j < kernel_size_w; j++)
                            {
                                tmp = v[i * kernel_size_w * channel_multiplier * num_input + j * channel_multiplier * num_input + p * channel_multiplier + q];
                                fwrite(&tmp, sizeof(float), 1, bp);
                            }
                        }
                    }
                }
            }
        }
        else if (op == "tf.Identity")
        {
        }
        else if (op == "tf.LeakyRelu")
        {
            float alpha = get_operation_attr_f(operation, "alpha");

            fprintf(pp, " 0=%e", alpha);
        }
        else if (op == "tf.MatMul")
        {
            int transpose_a = get_operation_attr_b(operation, "transpose_a");
            int transpose_b = get_operation_attr_b(operation, "transpose_b");

            if (transpose_a == 0 && transpose_b == 1)
            {
                // InnerProduct-like A * B + C
                std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
                const mlir::Attribute& W = weights[weight_name];

                llvm::ArrayRef<int64_t> shape = W.getType().cast<mlir::RankedTensorType>().getShape();

                //             assert(shape.size() == 2)

                // inch-outch
                int num_input = shape[0];
                int num_output = shape[1];
                int weight_data_size = shape[0] * shape[1];

                fprintf(pp, " 0=%d", num_output);
                fprintf(pp, " 2=%d", weight_data_size);

                std::vector<float> v = get_attr_af(W);

                // reorder i-o to o-i
                {
                    int quantize_tag = 0;
                    fwrite(&quantize_tag, sizeof(int), 1, bp);

                    float tmp;
                    for (int p = 0; p < num_output; p++)
                    {
                        for (int q = 0; q < num_input; q++)
                        {
                            tmp = v[q * num_output + p];
                            fwrite(&tmp, sizeof(float), 1, bp);
                        }
                    }
                }
            }
            else
            {
                // gemm
                fprintf(pp, " 0=1.0"); // alpha
                fprintf(pp, " 1=1.0"); // beta
                fprintf(pp, " 2=%d", transpose_a);
                fprintf(pp, " 3=%d", transpose_b);
            }
        }
        else if (op == "tf.Maximum")
        {
            int op_type = 4;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (op == "tf.MaxPool")
        {
            std::vector<int> ksize = get_operation_attr_ai(operation, "ksize");
            std::vector<int> strides = get_operation_attr_ai(operation, "strides");
            std::string padding = get_operation_attr_s(operation, "padding");

            fprintf(pp, " 0=0"); // max pool

            if (ksize.size() == 4)
            {
                fprintf(pp, " 1=%d", ksize[2]);
                fprintf(pp, " 11=%d", ksize[1]);
            }

            if (strides.size() == 4)
            {
                fprintf(pp, " 2=%d", strides[2]);
                fprintf(pp, " 12=%d", strides[1]);
            }

            int pad_mode = 1;
            if (padding == "VALID")
            {
                pad_mode = 1;
            }
            else if (padding == "SAME")
            {
                pad_mode = 2;
            }

            fprintf(pp, " 5=%d", pad_mode);
        }
        else if (op == "tf.Mean")
        {
            std::string reduction_indices_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& R = weights[reduction_indices_name];

            std::vector<int> v = get_attr_ai(R);

            int keep_dims = get_operation_attr_b(operation, "keep_dims");

            if (keep_dims == 0 && v.size() == 2 && v[0] == 1 && v[1] == 2)
            {
                // global avg pooling style nhwc -> nc
                int pool = 1;
                int global_pool = 1;

                fprintf(pp, " 0=%d", pool);
                fprintf(pp, " 4=%d", global_pool);
            }
            else
            {
                // TODO
            }
        }
        else if (op == "tf.Minimum")
        {
            int op_type = 5;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (op == "tf.Mul")
        {
            int op_type = 2;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (op == "tf.Pad")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& P = weights[weight_name];

            std::vector<int> v = get_attr_ai(P);

            // nhwc = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
            fprintf(pp, " 0=%d", v[2]);
            fprintf(pp, " 1=%d", v[3]);
            fprintf(pp, " 2=%d", v[4]);
            fprintf(pp, " 3=%d", v[5]);
        }
        else if (op == "tf.Placeholder")
        {
        }
        else if (op == "tf.Relu")
        {
        }
        else if (op == "tf.Relu6")
        {
            float min = 0.f;
            float max = 6.f;
            fprintf(pp, " 0=%e", min);
            fprintf(pp, " 1=%e", max);
        }
        else if (op == "tf.Reshape")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& S = weights[weight_name];

            std::vector<int> v = get_attr_ai(S);

            int size = v.size();

            // n h w c
            // n h c
            // n c
            if (size == 4)
            {
                fprintf(pp, " 0=%d 1=%d 2=%d", v[2], v[1], v[3]);
            }
            if (size == 3)
            {
                fprintf(pp, " 0=%d 1=%d 2=-233", v[1], v[2]);
            }
            if (size == 2)
            {
                fprintf(pp, " 0=%d 1=-233 2=-233", v[1]);
            }

            // FIXME may not always be the case
            fprintf(pp, " 3=1");
        }
        else if (op == "tf.ResizeBilinear")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& P = weights[weight_name];

            std::vector<int> size = get_attr_ai(P);

            int align_corners = get_operation_attr_b(operation, "align_corners");
            int half_pixel_centers = get_operation_attr_b(operation, "half_pixel_centers");
            if (!(align_corners == 0 && half_pixel_centers == 1))
            {
                fprintf(stderr, "Unsupported ResizeBilinear align_corners %d half_pixel_centers %d !\n", align_corners, half_pixel_centers);
            }

            fprintf(pp, " 0=2"); // bilinear
            fprintf(pp, " 3=%d 4=%d", size[1], size[0]);
        }
        else if (op == "tf.ResizeNearestNeighbor")
        {
            std::string weight_name = get_mlir_value_uniq_id(operation.getOperand(1));
            const mlir::Attribute& P = weights[weight_name];

            std::vector<int> size = get_attr_ai(P);

            int align_corners = get_operation_attr_b(operation, "align_corners");
            int half_pixel_centers = get_operation_attr_b(operation, "half_pixel_centers");
            if (!(align_corners == 0 && half_pixel_centers == 1))
            {
                fprintf(stderr, "Unsupported ResizeNearestNeighbor align_corners %d half_pixel_centers %d !\n", align_corners, half_pixel_centers);
            }

            fprintf(pp, " 0=1"); // nearest
            fprintf(pp, " 3=%d 4=%d", size[1], size[0]);
        }
        else if (op == "tf.Sigmoid")
        {
        }
        else if (op == "tf.Softmax")
        {
        }
        else if (op == "tf.StridedSlice")
        {
            std::string begin_name = get_mlir_value_uniq_id(operation.getOperand(1));
            std::string end_name = get_mlir_value_uniq_id(operation.getOperand(2));
            std::string strides_name = get_mlir_value_uniq_id(operation.getOperand(3));
            const mlir::Attribute& B = weights[begin_name];
            const mlir::Attribute& E = weights[end_name];
            const mlir::Attribute& S = weights[strides_name];

            std::vector<int> begin = get_attr_ai(B);
            std::vector<int> end = get_attr_ai(E);
            std::vector<int> strides = get_attr_ai(S);

            int begin_mask = get_operation_attr_i(operation, "begin_mask");
            int end_mask = get_operation_attr_i(operation, "end_mask");
            int ellipsis_mask = get_operation_attr_i(operation, "ellipsis_mask");
            int new_axis_mask = get_operation_attr_i(operation, "new_axis_mask");
            int shrink_axis_mask = get_operation_attr_i(operation, "shrink_axis_mask");

            int dims = strides.size();

            // assert strides == 1
            for (int i = 0; i < dims; i++)
            {
                if (strides[i] != 1)
                    fprintf(stderr, "Unsupported StridedSlice strides !\n");
            }

            for (int i = 0; i < dims; i++)
            {
                // TODO strides[i] < 0
                if (begin_mask & (1 << i))
                {
                    begin[i] = 0;
                }
                if (end_mask & (1 << i))
                {
                    end[i] = -233;
                }
                if (ellipsis_mask & (1 << i))
                {
                    begin[i] = 0;
                    end[i] = -233;
                }
            }

            if (new_axis_mask)
            {
                fprintf(stderr, "Unsupported StridedSlice new_axis_mask !\n");
            }

            if (shrink_axis_mask)
            {
                fprintf(stderr, "Unsupported StridedSlice shrink_axis_mask !\n");
            }

            // n h w c
            // n h c
            // n c
            if (dims == 4)
            {
                fprintf(pp, " -23309=3,%d,%d,%d", begin[3], begin[1], begin[2]);
                fprintf(pp, " -23310=3,%d,%d,%d", end[3], end[1], end[2]);
            }
            if (dims == 3)
            {
                fprintf(pp, " -23309=2,%d,%d", begin[2], begin[1]);
                fprintf(pp, " -23310=2,%d,%d", end[2], end[1]);
            }
            if (dims == 2)
            {
                fprintf(pp, " -23309=1,%d", begin[1]);
                fprintf(pp, " -23310=1,%d", end[1]);
            }
        }
        else if (op == "tf.Sub")
        {
            int op_type = 1;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (op == "tf.Tanh")
        {
        }

#if 0
        for (const mlir::NamedAttribute& attr : operation.getAttrs())
        {
            const mlir::Identifier& identifier = attr.first;
            const mlir::Attribute& attr = attr.second;

            fprintf(pp, " %s=", identifier.c_str());

            if (attr.isa<mlir::AffineMapAttr>())
            {
                fprintf(pp, "AffineMap");
            }
            if (attr.isa<mlir::ArrayAttr>())
            {
//                 fprintf(pp, "Array");
                mlir::ArrayAttr a = attr.cast<mlir::ArrayAttr>();
                int array_size = a.getValue().size();
                for (int t=0; t<array_size; t++)
                {
                    if (a[t].isa<mlir::IntegerAttr>())
                    {
                        int64_t ii = a[t].cast<mlir::IntegerAttr>().getInt();
                        fprintf(pp, "%lld,", ii);
                    }
                }
            }
            if (attr.isa<mlir::BoolAttr>())
            {
//                 fprintf(pp, "Bool");
                mlir::BoolAttr a = attr.cast<mlir::BoolAttr>();
                fprintf(pp, "%d", a.getValue() ? 1 : 0);
            }
            if (attr.isa<mlir::DictionaryAttr>())
            {
                fprintf(pp, "Dictionary");
            }
            if (attr.isa<mlir::FloatAttr>())
            {
                fprintf(pp, "Float");
            }
            if (attr.isa<mlir::IntegerAttr>())
            {
                fprintf(pp, "Integer");
            }
            if (attr.isa<mlir::IntegerSetAttr>())
            {
                fprintf(pp, "IntegerSet");
            }
            if (attr.isa<mlir::OpaqueAttr>())
            {
                fprintf(pp, "Opaque");
            }
            if (attr.isa<mlir::StringAttr>())
            {
//                 fprintf(pp, "String");
                mlir::StringAttr s = attr.cast<mlir::StringAttr>();
                fprintf(pp, "%s", s.getValue().empty() ? "" : s.getValue().data());
            }
            if (attr.isa<mlir::SymbolRefAttr>())
            {
                fprintf(pp, "SymbolRef");
            }
            if (attr.isa<mlir::FlatSymbolRefAttr>())
            {
                fprintf(pp, "FlatSymbolRef");
            }
            if (attr.isa<mlir::TypeAttr>())
            {
                fprintf(pp, "Type");
            }
            if (attr.isa<mlir::UnitAttr>())
            {
                fprintf(pp, "Unit");
            }
            if (attr.isa<mlir::ElementsAttr>())
            {
                fprintf(pp, "Elements");
            }
            if (attr.isa<mlir::DenseElementsAttr>())
            {
                fprintf(pp, "DenseElements");
            }
            if (attr.isa<mlir::DenseFPElementsAttr>())
            {
                fprintf(pp, "DenseFPElements");
            }
            if (attr.isa<mlir::DenseIntElementsAttr>())
            {
                fprintf(pp, "DenseIntElements");
            }
            if (attr.isa<mlir::OpaqueElementsAttr>())
            {
                fprintf(pp, "OpaqueElements");
            }
            if (attr.isa<mlir::SparseElementsAttr>())
            {
                fprintf(pp, "SparseElements");
            }
            if (attr.isa<mlir::SplatElementsAttr>())
            {
                fprintf(pp, "SplatElements");
            }

        }
#endif

        fprintf(pp, "\n");

        for (int j = 0; j < num_output; j++)
        {
            std::string output_name = get_mlir_value_uniq_id(operation.getResult(j));
            if (node_reference.find(output_name) != node_reference.end())
            {
                int refcount = node_reference[output_name];
                if (refcount > 1)
                {
                    char splitname[256];
                    sprintf(splitname, "splitncnn_%d", internal_split);
                    fprintf(pp, "%-16s %-24s %d %d", "Split", splitname, 1, refcount);

                    fprintf(pp, " %s", output_name.c_str());

                    for (int k = 0; k < refcount; k++)
                    {
                        fprintf(pp, " %s_splitncnn_%d", output_name.c_str(), k);
                    }
                    fprintf(pp, "\n");

                    internal_split++;
                }
            }
        }
    }

    fclose(pp);
    fclose(bp);

    return 0;
}
