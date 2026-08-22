# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import pathlib
import re
import subprocess
import sys

import torch


class PnnxTestError(Exception):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def _inputshape(inputs):
    return "inputshape=" + ",".join("[" + ",".join(str(x) for x in tensor.shape) + "]" for tensor in inputs)


def _load_model(path):
    spec = importlib.util.spec_from_file_location("pnnx_test_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model()


def _check_output(expected, actual, rtol=1e-5, atol=1e-6, reshape=False):
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(expected) != len(actual):
            raise AssertionError("output structure differs")
        for a, b in zip(expected, actual):
            _check_output(a, b, rtol, atol, reshape)
        return
    if reshape and expected.numel() == actual.numel():
        actual = actual.reshape(expected.shape)
    if actual.shape != expected.shape:
        raise AssertionError("output shape differs: %s != %s" % (actual.shape, expected.shape))
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError("maximum absolute difference is %g" % torch.max(torch.abs(actual - expected)).item())


def _flatten_output(output):
    if isinstance(output, (tuple, list)):
        result = []
        for value in output:
            result.extend(_flatten_output(value))
        return result
    return [output]


def _run_ncnn(workdir, inputs, output_count):
    import ncnn
    if not hasattr(ncnn, "Net") or not hasattr(ncnn, "Mat"):
        raise RuntimeError("complete ncnn Python binding is required")

    source = (workdir / "model_ncnn.py").read_text()
    input_batch_index = {int(i): int(axis) for i, axis in re.findall(r'ex\.input\("in(\d+)", .*batch_index=(-?\d+)', source)}
    output_batch_index = {int(i): int(axis) for i, axis in re.findall(r'out(\d+)\.numpy\(batch_index=(-?\d+)\)', source)}
    outputs = []
    with ncnn.Net() as net:
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        if net.load_param(str(workdir / "model.ncnn.param")) != 0 or net.load_model(str(workdir / "model.ncnn.bin")) != 0:
            raise RuntimeError("failed to load ncnn model")
        with net.create_extractor() as ex:
            for i, tensor in enumerate(inputs):
                if ex.input("in%d" % i, ncnn.Mat(tensor.numpy(), batch_index=input_batch_index[i]).clone()) != 0:
                    raise RuntimeError("failed to set ncnn input in%d" % i)
            for i in range(output_count):
                ret, output = ex.extract("out%d" % i)
                if ret != 0:
                    raise RuntimeError("failed to extract ncnn output out%d" % i)
                outputs.append(torch.from_numpy(output.numpy(batch_index=output_batch_index[i])))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def run_test(pnnx, workdir, format, model, inputs):
    workdir = pathlib.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    model.eval()

    model_path = workdir / ("model.pt2" if format == "pt2" else "model.pt")
    try:
        if format == "pt2":
            exported = torch.export.export(model, inputs)
            torch.export.save(exported, model_path)
        else:
            torch.jit.trace(model, inputs).save(str(model_path))
    except Exception as e:
        raise PnnxTestError("export", str(e)) from e

    command = [str(pathlib.Path(pnnx).resolve()), model_path.name]
    if format == "torchscript":
        command.append(_inputshape(inputs))
    process = subprocess.run(command, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        stage = "level2" if "############# pass_level2" in process.stdout else "load"
        raise PnnxTestError(stage, process.stdout)

    previous_workdir = pathlib.Path.cwd()
    try:
        os.chdir(workdir)
        converted = _load_model(workdir / "model_pnnx.py")
        converted.eval()
        with torch.no_grad():
            expected = model(*inputs)
            actual = converted(*inputs)
    except Exception as e:
        raise PnnxTestError("pnnx-runtime", str(e)) from e
    finally:
        os.chdir(previous_workdir)

    try:
        _check_output(expected, actual)
    except Exception as e:
        raise PnnxTestError("numeric-diff", "pnnx: " + str(e)) from e

    try:
        ncnn_output = _run_ncnn(workdir, inputs, len(_flatten_output(expected)))
    except Exception as e:
        raise PnnxTestError("ncnn-runtime", str(e)) from e
    try:
        _check_output(expected, ncnn_output, 1e-3, 1e-3, True)
    except Exception as e:
        raise PnnxTestError("numeric-diff", "ncnn: " + str(e)) from e


def main(test, pnnx, workdir, format, case):
    if format == "pt2" and tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
        print("PT2 test skipped: PyTorch 2.6 or newer is required")
        return 77

    torch.manual_seed(0)
    model, inputs = test(case)
    try:
        run_test(pnnx, workdir, format, model, inputs)
    except PnnxTestError as e:
        print("pnnx test failed: format=%s case=%s stage=%s\n%s" % (format, case, e.stage, e), file=sys.stderr)
        return 1
    return 0
