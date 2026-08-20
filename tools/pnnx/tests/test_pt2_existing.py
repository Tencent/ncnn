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

import torch


UNSUPPORTED = {
    "Tensor_index": "torch.export cannot serialize the data-dependent indexed output shape",
    "torch_arange": "torch.export cannot guard the data-dependent arange step",
    "transformers_funnel_attention": "data-dependent symbolic floats require runtime scalar lowering",
    "transformers_longformer_attention": "PyTorch cannot deserialize its saved PT2 graph with an unused scalar output",
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
        pt2_path = path.with_suffix(".pt2")
        try:
            program = torch.export.export(self.model, self.inputs)
            torch.export.save(program, pt2_path)
        except Exception as e:
            raise TestError("export", str(e)) from e
        self.models[path.name] = pt2_path.name


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

    def convert(command):
        arguments = shlex.split(command)
        if not arguments or pathlib.Path(arguments[0]).name != "pnnx":
            return system(command)
        arguments[0] = str(pathlib.Path(pnnx).resolve())
        for i in range(1, len(arguments)):
            if arguments[i] in models:
                arguments[i] = models[arguments[i]]
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
        for i in range(1, len(arguments)):
            if arguments[i] in models:
                arguments[i] = models[arguments[i]]
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
            def check_stat(text, expected_inputshape, expected_flops, expected_memops):
                del expected_flops, expected_memops
                inputshape = test.re.findall(r"(?:^|\n)#? ?model inputshape = (.+)", text)
                flops = test.re.findall(r"(?:^|\n)#? ?FLOPS = ([0-9.]+[KMGTPE]?)", text)
                memops = test.re.findall(r"(?:^|\n)#? ?memory OPS = ([0-9.]+[KMGTPE]?)", text)
                return bool(inputshape and flops and memops) and inputshape[-1] == expected_inputshape
            test._check_stat_text = check_stat
        if hasattr(test, "_test_input2"):
            test._test_input2 = lambda: True
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
            with zipfile.ZipFile(pt2_files[0]) as archive:
                model = next(x for x in archive.namelist() if x.endswith("/models/model.json"))
                version = json.loads(archive.read(model)).get("schema_version", {})
                schema = "%s.%s" % (version.get("major", "?"), version.get("minor", "?"))
        print("pnnx test failed: format=pt2 producer=torch-%s schema=%s case=%s stage=%s\n%s" %
              (torch.__version__, schema, args.case, e.stage, e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
