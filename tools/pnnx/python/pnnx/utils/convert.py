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

import pnnx
import torch
import os

def check_type(data, dataname, types, typesname):
    if not(data is None):
        if (type(data) in types):
            return True
        else:
            raise Exception(dataname + " should be "+ typesname + ".")
    else:
        return True

def convert(ptpath, input_shapes, input_types, input_shapes2 = None,
            input_types2 = None, device = None, customop_modules = None,
            module_operators = None, optlevel = None, pnnxparam = None,
            pnnxbin = None, pnnxpy = None, pnnxonnx = None, ncnnparam = None,
            ncnnbin = None, ncnnpy = None):

    check_type(ptpath, "modelname", [str], "str")
    check_type(input_shapes, "input_shapes", [list], "list of list with int type inside")
    check_type(input_types, "input_types", [str, list], "str or list of str")
    check_type(input_shapes2, "input_shapes2", [list], "list of list with int type inside")
    check_type(input_types2, "input_types2", [str, list], "str or  list of str")
    check_type(device, "device", [str], "str")
    check_type(customop_modules, "customop_modules", [str, list], "str or list of str")
    check_type(module_operators, "module_operators", [str, list], "str or list of str")
    check_type(optlevel, "optlevel", [int], "int")

    if input_shapes2 is None:
        input_shapes2 = []
    elif type(input_shapes2[0])!= list:
        input_shapes2 = [input_shapes2]
    if input_types2 is None:
        input_types2 = []
    elif type(input_types2) != list:
        input_types2 = [input_types2]
    if customop_modules is None:
        customop_modules = []
    elif type(customop_modules) != list:
        customop_modules = [customop_modules]
    if module_operators is None:
        module_operators = []
    elif type(module_operators) != list:
        module_operators = [module_operators]
    if device is None:
        device = "cpu"
    if optlevel is None:
        optlevel = 2
    if pnnxparam is None:
        pnnxparam = ""
    if pnnxbin is None:
        pnnxbin = ""
    if pnnxpy is None:
        pnnxpy = ""
    if pnnxonnx is None:
        pnnxonnx = ""
    if ncnnparam is None:
        ncnnparam = ""
    if ncnnbin is None:
        ncnnbin = ""
    if ncnnpy is None:
        ncnnpy = ""
    if type(input_shapes[0]) != list:
        input_shapes = [input_shapes]
    if type(input_types) != list:
        input_types = [input_types]

    if len(input_shapes) != len(input_types):
        raise Exception("input_shapes should has the same length with input_types!")
    if len(input_shapes2) != len(input_types2):
        raise Exception("input_shapes2 should has the same length with input_types2!")

    pnnx.pnnx_export(ptpath, input_shapes, input_types, input_shapes2,
                     input_types2, device, customop_modules, module_operators,
                     optlevel, pnnxparam,pnnxbin, pnnxpy, pnnxonnx, ncnnparam,
                     ncnnbin, ncnnpy)