# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

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
