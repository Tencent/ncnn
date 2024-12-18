# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import onnx

def export(ptpath_or_model, output_model_path, inputs=None, inputs2=None, input_shapes=None, input_types=None,
           input_shapes2=None, input_types2=None, device=None, customop=None, moduleop=None,
           optlevel=None, pnnxparam=None, pnnxbin=None, pnnxpy=None, pnnxonnx=None,
           ncnnparam=None, ncnnbin=None, ncnnpy=None, check_trace=True, fp16=True):

    if (inputs is None) and (input_shapes is None):
        raise Exception("inputs or input_shapes should be specified.")
    if not (input_shapes is None) and (input_types is None):
        raise Exception("when input_shapes is specified, then input_types should be specified correspondingly.")

    # Check if the input is a path or a model instance
    if isinstance(ptpath_or_model, torch.nn.Module):
        ptpath_or_model.eval()
        mod = torch.jit.trace(ptpath_or_model, inputs, check_trace=check_trace)
        mod.save(output_model_path)
        from . import convert
        return convert(output_model_path, inputs, inputs2, input_shapes, input_types,
                        input_shapes2, input_types2, device, customop, moduleop,
                        optlevel, pnnxparam, pnnxbin, pnnxpy, pnnxonnx,
                        ncnnparam, ncnnbin, ncnnpy, fp16)
    elif isinstance(ptpath_or_model, str):
        ptpath = ptpath_or_model
    else:
        raise TypeError("`ptpath_or_model` must be a PyTorch model instance or a file path.")

    # If it is a PyTorch model file (.pt)
    if ptpath.endswith('.pt'):
        model = torch.jit.load(ptpath)
        model.eval()
        mod = torch.jit.trace(model, inputs, check_trace=check_trace)
        mod.save(output_model_path)
        ptpath = output_model_path
        from . import convert
        return convert(ptpath, inputs, inputs2, input_shapes, input_types,
                       input_shapes2, input_types2, device, customop, moduleop,
                       optlevel, pnnxparam, pnnxbin, pnnxpy, pnnxonnx,
                       ncnnparam, ncnnbin, ncnnpy, fp16)

    # If the input is an ONNX model file (.onnx)
    elif ptpath.endswith('.onnx'):
        model = onnx.load(ptpath)
        onnx.checker.check_model(model)
        from . import convert
        return convert(ptpath, inputs=inputs, inputs2=inputs2,
                       input_shapes=input_shapes, input_types=input_types,
                       input_shapes2=input_shapes2, input_types2=input_types2,
                       device=device, customop=customop, moduleop=moduleop,
                       optlevel=optlevel, pnnxparam=pnnxparam, pnnxbin=pnnxbin,
                       pnnxpy=pnnxpy, pnnxonnx=pnnxonnx, ncnnparam=ncnnparam,
                       ncnnbin=ncnnbin, ncnnpy=ncnnpy, fp16=fp16)
    else:
        raise Exception("Unsupported model file type. Only .pt and .onnx files are supported.")

