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


def export(model, filename, inputs = None, input_shapes = None, input_shapes2 = None,
           input_types = None, input_types2 = None, device = None, module_operators = None,
           optlevel = None, pnnxparam = None, pnnxbin = None, pnnxpy = None,
           pnnxonnx = None, ncnnparam = None, ncnnbin = None, ncnnpy = None):
    if (inputs is None) and (input_shapes is None):
        raise Exception("inputs or input_shapes should be specified.")
    if not (input_shapes is None) and (input_types is None):
        raise Exception("when input_shapes is specified, then input_types should be specified correspondingly.")

    check_type(filename, "filename", [str], "str")
    check_type(inputs, "inputs", [tuple, list], "tuple or list of torch.Tensor")
    check_type(input_shapes, "input_shapes", [list], "list of list with int type inside")
    check_type(input_types, "input_types", [list], "list of str")
    check_type(input_shapes2, "input_shapes2", [list], "list of list with int type inside")
    check_type(input_types2, "input_types2", [list], "list of str")
    check_type(device, "device", [str], "str")
    check_type(module_operators, "module_operators", [list], "list of str")
    check_type(optlevel, "optlevel", [int], "int")

    if input_shapes2 is None:
        input_shapes2 = []
    if input_types2 is None:
        input_types2 = []
    if module_operators is None:
        module_operators = []
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


    if not (inputs is None):
        mod = torch.jit.trace(model, inputs)
        mod.save(filename)
        current_path = os.path.abspath(filename)

        devicename = str(next(model.parameters()).device)
        if ("cpu" in devicename):
            device = "cpu"
        elif ("cuda" in devicename):
            device = "gpu"

        input_shapes = get_shape_from_inputs(inputs)
        input_types = get_type_from_inputs(inputs)

        pnnx.pnnx_export(current_path, input_shapes, input_types, input_shapes2,
                         input_types2, device, module_operators, optlevel, pnnxparam,
                         pnnxbin, pnnxpy, pnnxonnx, ncnnparam, ncnnbin, ncnnpy)
    else: # use input_shapes and input_types
        if (input_shapes is None) or (input_types is None):
            raise Exception("input_shapes and input_types should be specified together.")

        mod = torch.jit.trace(model, inputs)
        mod.save(filename)
        current_path = os.path.abspath(filename)

        devicename = next(net.parameters()).device
        if ("cpu" in devicename):
            device = "cpu"
        elif ("cuda" in devicename):
            device = "gpu"

        pnnx.pnnx_export(current_path, input_shapes, input_types, input_shapes2,
                         input_types2, device, module_operators, optlevel, pnnxparam,
                         pnnxbin, pnnxpy, pnnxonnx, ncnnparam, ncnnbin, ncnnpy)