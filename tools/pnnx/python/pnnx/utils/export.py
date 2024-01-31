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
import os

def export(model, ptpath, inputs = None, inputs2 = None, input_shapes = None, input_types = None,
           input_shapes2 = None, input_types2 = None, device = None, customop = None,
           moduleop = None, optlevel = None, pnnxparam = None, pnnxbin = None,
           pnnxpy = None, pnnxonnx = None, ncnnparam = None, ncnnbin = None, ncnnpy = None,
           check_trace = True, fp16 = True):
    if (inputs is None) and (input_shapes is None):
        raise Exception("inputs or input_shapes should be specified.")
    if not (input_shapes is None) and (input_types is None):
        raise Exception("when input_shapes is specified, then input_types should be specified correspondingly.")

    model.eval()
    mod = torch.jit.trace(model, inputs, check_trace=check_trace)
    mod.save(ptpath)

    from . import convert
    return convert(ptpath, inputs, inputs2, input_shapes, input_types, input_shapes2, input_types2, device, customop, moduleop, optlevel, pnnxparam, pnnxbin, pnnxpy, pnnxonnx, ncnnparam, ncnnbin, ncnnpy, fp16)
