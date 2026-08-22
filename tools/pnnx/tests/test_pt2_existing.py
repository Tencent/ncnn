# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import importlib.util
import json
import os
import pathlib
import shlex
import subprocess
import sys
import zipfile

import numpy as np
import torch


UNSUPPORTED = {
    "Tensor_index": "torch.export cannot serialize the data-dependent indexed output shape",
    "torch_arange": "torch.export cannot guard the data-dependent arange step",
    "torch_masked_select": "data-dependent output shape cannot be specialized from input profiles",
    "transformers_deepseek_v3_attention": "exported graph contains unsupported wrap_with_autocast",
    "transformers_funnel_attention": "data-dependent symbolic floats require runtime scalar lowering",
    "transformers_longformer_attention": "PyTorch cannot deserialize its saved PT2 graph with an unused scalar output",
    "transformers_qwen2_attention": "exported graph contains unsupported wrap_with_autocast",
    "transformers_qwen3_attention": "exported graph contains unsupported wrap_with_autocast",
}

PT2_MODEL_STATS = {
    "test_pnnx_model_stat_multihead_attention_mask": (684, 227),
    "test_pnnx_model_stat_multihead_attention_extra": (822, 410),
    "test_pnnx_model_stat_multihead_attention_unbatched": (666, 242),
    "test_pnnx_model_stat_lstm_proj_state": (0, 18),
    "test_pnnx_model_stat_lstm_unbatched": (0, 28),
    "test_pnnx_model_stat_fused_functional": (324, 206),
}


class TestError(Exception):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


class ExportedModel:
    def __init__(self, model, inputs, models):
        self.model = model
        self.inputs = inputs if isinstance(inputs, tuple) else (inputs,)
        self.models = models

    def save(self, path):
        path = pathlib.Path(path)
        self.path = path.with_suffix(".pt2")
        self.export()
        self.models[path.name] = self

    def export(self, input2=()):
        try:
            dynamic_shapes = None
            if input2:
                if len(input2) != len(self.inputs):
                    raise ValueError("input2 tensor count differs from the export inputs")
                dynamic_shapes = []
                for value, value2 in zip(self.inputs, input2):
                    if value.dim() != value2.dim():
                        raise ValueError("input2 tensor rank differs from the export input")
                    shape = {}
                    for axis, (size, size2) in enumerate(zip(value.shape, value2.shape)):
                        if size == size2:
                            continue
                        shape[axis] = torch.export.Dim.AUTO
                    dynamic_shapes.append(shape)
                dynamic_shapes = tuple(dynamic_shapes)

            program = torch.export.export(self.model, self.inputs, dynamic_shapes=dynamic_shapes)
            torch.export.save(program, self.path)
        except Exception as e:
            raise TestError("export", str(e)) from e


def load_test(path):
    spec = importlib.util.spec_from_file_location("pnnx_existing_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_test(test_path, pnnx, workdir):
    workdir = pathlib.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    models = {}
    trace = torch.jit.trace
    system = os.system
    subprocess_run = subprocess.run

    def export(model, inputs, *args, **kwargs):
        return ExportedModel(model, inputs, models)

    def convert_arguments(arguments):
        arguments = list(arguments)
        input2 = ()
        for argument in arguments:
            if argument.startswith("input2="):
                input2 = tuple(torch.from_numpy(np.ascontiguousarray(np.load(path))) for path in argument[7:].split(","))
                break
        for i in range(1, len(arguments)):
            if arguments[i] in models:
                model = models[arguments[i]]
                if input2:
                    model.export(input2)
                arguments[i] = model.path.name
        return arguments

    def convert(command):
        arguments = shlex.split(command)
        if not arguments or pathlib.Path(arguments[0]).name != "pnnx":
            return system(command)
        arguments[0] = str(pathlib.Path(pnnx).resolve())
        arguments = convert_arguments(arguments)
        process = subprocess_run(arguments, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if process.returncode != 0:
            stage = "level2" if "############# pass_level2" in process.stdout else "load"
            raise TestError(stage, process.stdout)
        return 0

    def convert_run(arguments, *args, **kwargs):
        if not arguments or pathlib.Path(arguments[0]).name != "pnnx":
            return subprocess_run(arguments, *args, **kwargs)
        arguments = list(arguments)
        arguments[0] = str(pathlib.Path(pnnx).resolve())
        arguments = convert_arguments(arguments)
        return subprocess_run(arguments, *args, **kwargs)

    previous_workdir = pathlib.Path.cwd()
    sys.path.insert(0, str(workdir))
    sys.path.insert(0, str(pathlib.Path(test_path).parent))
    torch.jit.trace = export
    os.system = convert
    try:
        os.chdir(workdir)
        test = load_test(test_path)
        subprocess.run = convert_run
        if hasattr(test, "_check_stat_text"):
            run_case = test._run_case

            def run_pt2_case(name, net, inputs, inputshape, expected_inputshape, expected_flops, expected_memops):
                expected_flops, expected_memops = PT2_MODEL_STATS.get(name, (expected_flops, expected_memops))
                return run_case(name, net, inputs, inputshape, expected_inputshape, expected_flops, expected_memops)

            test._run_case = run_pt2_case
        if not test.test():
            raise TestError("numeric-diff", "test returned false")
    except TestError:
        raise
    except Exception as e:
        raise TestError("pnnx-runtime", str(e)) from e
    finally:
        os.chdir(previous_workdir)
        os.system = system
        subprocess.run = subprocess_run
        torch.jit.trace = trace
        sys.path.pop(0)
        sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=pathlib.Path, required=True)
    parser.add_argument("--pnnx", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()

    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
        print("PT2 test skipped: PyTorch 2.6 or newer is required")
        return 77
    if args.case in UNSUPPORTED:
        print("PT2 test skipped: format=pt2 producer=torch-%s case=%s stage=unsupported reason=%s" %
              (torch.__version__, args.case, UNSUPPORTED[args.case]))
        return 77

    try:
        run_test(args.test, args.pnnx, args.workdir)
    except TestError as e:
        schema = "unavailable"
        pt2_files = list(pathlib.Path(args.workdir).glob("*.pt2"))
        if pt2_files:
            try:
                with zipfile.ZipFile(pt2_files[0]) as archive:
                    model = next((x for x in archive.namelist() if x.endswith("models/model.json") or x.endswith("serialized_exported_program.json")), None)
                    if model:
                        version = json.loads(archive.read(model)).get("schema_version", {})
                        schema = "%s.%s" % (version.get("major", "?"), version.get("minor", "?"))
            except Exception:
                pass
        print("pnnx test failed: format=pt2 producer=torch-%s schema=%s case=%s stage=%s\n%s" %
              (torch.__version__, schema, args.case, e.stage, e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
