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
from .utils import check_type, get_shape_from_inputs, get_type_from_inputs, generate_inputs_arg, str_in_list_to_str
import subprocess
from .. import EXEC_PATH

def convert(ptpath, inputs = None, inputs2 = None, input_shapes = None, input_types = None,
            input_shapes2 = None, input_types2 = None, device = None, customop = None,
            moduleop = None, optlevel = None, pnnxparam = None, pnnxbin = None,
            pnnxpy = None, pnnxonnx = None, ncnnparam = None, ncnnbin = None, ncnnpy = None, fp16 = True):

    check_type(ptpath, "modelname", [str], "str")
    check_type(inputs, "inputs", [torch.Tensor, tuple, list], "torch.Tensor or tuple/list of torch.Tensor")
    check_type(inputs2, "inputs2", [torch.Tensor, tuple, list], "torch.Tensor or tuple/list of torch.Tensor")
    check_type(input_shapes, "input_shapes", [list], "list of list with int type inside")
    check_type(input_types, "input_types", [str, list], "str or list of str")
    check_type(input_shapes2, "input_shapes2", [list], "list of list with int type inside")
    check_type(input_types2, "input_types2", [str, list], "str or  list of str")
    check_type(device, "device", [str], "str")
    check_type(customop, "customop", [str, list], "str or list of str")
    check_type(moduleop, "moduleop", [str, list], "str or list of str")
    check_type(optlevel, "optlevel", [int], "int")

    if input_shapes2 is None:
        input_shapes2 = []
    elif type(input_shapes2[0])!= list:
        input_shapes2 = [input_shapes2]
    if input_types2 is None:
        input_types2 = []
    elif type(input_types2) != list:
        input_types2 = [input_types2]
    if customop is None:
        customop = []
    elif type(customop) != list:
        customop = [customop]
    if moduleop is None:
        moduleop = []
    elif type(moduleop) != list:
        moduleop = [moduleop]
    if device is None:
        device = "cpu"
    if optlevel is None:
        optlevel = 2
    if type(inputs) == torch.Tensor:
        inputs = [inputs]
    if type(inputs2) == torch.Tensor:
        inputs2 = [inputs2]

    if not (inputs is None):

        if device is None:
            try:
                devicename = str(next(model.parameters()).device)
                if ("cpu" in devicename):
                    device = "cpu"
                elif ("cuda" in devicename):
                    device = "gpu"
            except: # model without parameters
                device = "cpu"

        input_shapes = get_shape_from_inputs(inputs)
        input_types = get_type_from_inputs(inputs)

        if not (inputs2 is None):
            input_shapes2 = get_shape_from_inputs(inputs2)
            input_types2 = get_type_from_inputs(inputs2)

        input_arg1 = generate_inputs_arg(input_shapes, input_types)

        command_list = [EXEC_PATH, ptpath, "inputshape=" + input_arg1,
                        "device=" + device,
                        "optlevel=" + str(optlevel)]
        if not (len(input_shapes2) == 0):
            input_arg2 = generate_inputs_arg(input_shapes2, input_types2)
            command_list.append("inputshape2=" + input_arg2)
        if not (len(customop) == 0):
            command_list.append("customop=" + str_in_list_to_str(customop))
        if not (len(moduleop) == 0):
            command_list.append("moduleop=" + str_in_list_to_str(moduleop))

        if not (pnnxparam is None):
            command_list.append("pnnxparam=" + pnnxparam)
        if not (pnnxbin is None):
            command_list.append("pnnxbin=" + pnnxbin)
        if not (pnnxpy is None):
            command_list.append("pnnxpy=" + pnnxpy)
        if not (pnnxonnx is None):
            command_list.append("pnnxonnx=" + pnnxonnx)
        if not (ncnnparam is None):
            command_list.append("ncnnparam=" + ncnnparam)
        if not (ncnnbin is None):
            command_list.append("ncnnbin=" + ncnnbin)
        if not (ncnnpy is None):
            command_list.append("ncnnpy=" + ncnnpy)
        if not (fp16 is True):
            command_list.append("fp16=0")
        current_dir = os.getcwd()
        subprocess.run(command_list, stdout=subprocess.PIPE, text=True, cwd=current_dir)

    else: # use input_shapes and input_types
        if (input_shapes is None) or (input_types is None):
            raise Exception("input_shapes and input_types should be specified together.")

        if device is None:
            try:
                devicename = str(next(model.parameters()).device)
                if ("cpu" in devicename):
                    device = "cpu"
                elif ("cuda" in devicename):
                    device = "gpu"
            except: # model without parameters
                device = "cpu"

        input_arg1 = generate_inputs_arg(input_shapes, input_types)

        command_list = [EXEC_PATH, ptpath, "inputshape=" + input_arg1,
                        "device=" + device,
                        "optlevel=" + str(optlevel)]
        if not (len(input_shapes2) == 0):
            input_arg2 = generate_inputs_arg(input_shapes2, input_types2)
            command_list.append("inputshape2=" + input_arg2)
        if not (len(customop) == 0):
            command_list.append("customop=" + str_in_list_to_str(customop))
        if not (len(moduleop) == 0):
            command_list.append("moduleop=" + str_in_list_to_str(moduleop))
        if not (pnnxparam is None):
            command_list.append("pnnxparam=" + pnnxparam)
        if not (pnnxbin is None):
            command_list.append("pnnxbin=" + pnnxbin)
        if not (pnnxpy is None):
            command_list.append("pnnxpy=" + pnnxpy)
        if not (pnnxonnx is None):
            command_list.append("pnnxonnx=" + pnnxonnx)
        if not (ncnnparam is None):
            command_list.append("ncnnparam=" + ncnnparam)
        if not (ncnnbin is None):
            command_list.append("ncnnbin=" + ncnnbin)
        if not (ncnnpy is None):
            command_list.append("ncnnpy=" + ncnnpy)
        current_dir = os.getcwd()
        subprocess.run(command_list, stdout=subprocess.PIPE, text=True, cwd=current_dir)

    # return pnnx model
    if pnnxpy is None:
        pnnxpy = os.path.splitext(ptpath)[0] + '_pnnx.py'

    pnnx_module_name = os.path.splitext(pnnxpy)[0]

    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(pnnx_module_name, pnnxpy)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[pnnx_module_name] = foo
    spec.loader.exec_module(foo)
    return foo.Model()
