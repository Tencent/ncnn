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

import os
from .utils import check_type, generate_inputs_arg, str_in_list_to_str
import subprocess
from .. import EXEC_PATH

def convert(ptpath, input_shapes, input_types, input_shapes2 = None,
            input_types2 = None, device = None, customop = None,
            moduleop = None, optlevel = None, pnnxparam = None,
            pnnxbin = None, pnnxpy = None, pnnxonnx = None, ncnnparam = None,
            ncnnbin = None, ncnnpy = None):

    check_type(ptpath, "modelname", [str], "str")
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

    if type(input_shapes[0]) != list:
        input_shapes = [input_shapes]
    if type(input_types) != list:
        input_types = [input_types]

    if len(input_shapes) != len(input_types):
        raise Exception("input_shapes should has the same length with input_types!")
    if len(input_shapes2) != len(input_types2):
        raise Exception("input_shapes2 should has the same length with input_types2!")

    input_arg1 = generate_inputs_arg(input_shapes, input_types)

    command_list = [EXEC_PATH, ptpath, "inputshape="+input_arg1,
                    "device="+device,
                    "optlevel="+str(optlevel)]
    if not (len(input_shapes2) == 0):
        input_arg2 = generate_inputs_arg(input_shapes2, input_types2)
        command_list.append("inputshape2=" + input_arg2)
    if not (len(customop) == 0):
        command_list.append("customop=" + str_in_list_to_str(customop))
    if not (len(moduleop) == 0):
        command_list.append("moduleop=" + str_in_list_to_str(moduleop))
    if not(pnnxparam is None):
        command_list.append("pnnxparam="+pnnxparam)
    if not(pnnxbin is None):
        command_list.append("pnnxbin="+pnnxbin)
    if not(pnnxpy is None):
        command_list.append("pnnxpy="+pnnxpy)
    if not(pnnxonnx is None):
        command_list.append("pnnxonnx="+pnnxonnx)
    if not(ncnnparam is None):
        command_list.append("ncnnparam="+ncnnparam)
    if not(ncnnbin is None):
        command_list.append("ncnnbin="+ncnnbin)
    if not(ncnnpy is None):
        command_list.append("ncnnpy="+ncnnpy)
    current_dir = os.getcwd()
    subprocess.run(command_list, stdout=subprocess.PIPE, text=True, cwd=current_dir)