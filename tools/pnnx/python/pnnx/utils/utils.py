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

def check_type(data, dataname, types, typesname):
    if not(data is None):
        if (type(data) in types):
            return True
        else:
            raise Exception(dataname + " should be "+ typesname + ".")
    else:
        return True

def get_shape_from_inputs(inputs):
    shapes = []
    for item in inputs:
        sub_shapes = []
        for l in item.shape:
            sub_shapes.append(l)
        shapes.append(sub_shapes)
    return shapes

def input_torch_type_to_str(tensor):
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float:
        return "f32"
    if tensor.dtype == torch.float64 or tensor.dtype == torch.double:
        return "f64"
    if tensor.dtype == torch.float16 or tensor.dtype == torch.half:
        return "f16"
    if tensor.dtype == torch.uint8:
        return "u8"
    if tensor.dtype == torch.int8:
        return "i8"
    if tensor.dtype == torch.int16 or tensor.dtype == torch.short:
        return "i16"
    if tensor.dtype == torch.int32 or tensor.dtype == torch.int:
        return "i32"
    if tensor.dtype == torch.int64 or tensor.dtype == torch.long:
        return "i64"
    if tensor.dtype == torch.complex32:
        return "c32"
    if tensor.dtype == torch.complex64:
        return "c64"
    if tensor.dtype == torch.complex128:
        return "c128"

    return "f32"

def get_type_from_inputs(inputs):
    types = []
    for item in inputs:
        types.append(input_torch_type_to_str(item))
    return types

def generate_inputs_arg(inputs, input_shapes):
    generated_arg = ""
    for i in range(0, len(inputs) - 1):
        generated_arg += "["
        for j in range(0, len(inputs[i]) - 1):
            generated_arg += str(inputs[i][j]) + ','
        generated_arg += str(inputs[i][-1])
        generated_arg += "]"
        generated_arg += input_shapes[i]
        generated_arg += ","
    generated_arg += "["
    for j in range(0, len(inputs[-1]) - 1):
        generated_arg += str(inputs[-1][j]) + ','
    generated_arg += str(inputs[-1][-1])
    generated_arg += "]"
    generated_arg += input_shapes[-1]
    return generated_arg

def str_in_list_to_str(input_list):
    generated_str = ""
    for i in range(0, len(input_list) - 1):
        generated_str += input_list[i] + ','
    generated_str += input_list[-1]
    return generated_str

