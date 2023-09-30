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
            input_types2 = None, device = None, module_operators = None,
           optlevel = None, pnnxparam = None, pnnxbin = None, pnnxpy = None,
           pnnxonnx = None, ncnnparam = None, ncnnbin = None, ncnnpy = None):

    check_type(ptpath, "modelname", [str], "str")
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
    if device is None:
        device = "cpu"
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

    pnnx.pnnx_export(ptpath, input_shapes, input_types, input_shapes2,
                     input_types2, device, module_operators, optlevel, pnnxparam,
                     pnnxbin, pnnxpy, pnnxonnx, ncnnparam, ncnnbin, ncnnpy)