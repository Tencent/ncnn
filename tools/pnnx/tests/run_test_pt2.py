# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import runpy
import shutil
import sys
import traceback

import torch


original_system = os.system
original_remove_parametrizations = torch.nn.utils.parametrize.remove_parametrizations


class TorchExportUnavailable(Exception):
    pass


def has_torch_export():
    version = torch.__version__.split("+", 1)[0].split(".")
    try:
        torch_version = tuple(int(x) for x in version[:2])
    except ValueError:
        torch_version = (0, 0)

    return torch_version >= (2, 2) and hasattr(torch, "export") and hasattr(torch.export, "save")


class ExportedProgramTrace:
    def __init__(self, module, example_inputs, example_kwarg_inputs=None):
        self.module = module
        self.example_inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
        self.example_kwarg_inputs = example_kwarg_inputs or {}

    def save(self, path):
        for submodule in self.module.modules():
            if torch.nn.utils.parametrize.is_parametrized(submodule, "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(submodule, "weight", leave_parametrized=True)
                continue
            try:
                torch.nn.utils.remove_weight_norm(submodule)
            except (AttributeError, ValueError):
                pass

        # torch.export itself must see the original torch.jit.trace function.
        torch.jit.trace = original_trace
        try:
            try:
                exported_program = torch.export.export(
                    self.module,
                    self.example_inputs,
                    kwargs=self.example_kwarg_inputs,
                )
            except Exception as e:
                raise TorchExportUnavailable(str(e)) from e
        finally:
            torch.jit.trace = exported_program_trace

        basename, _ = os.path.splitext(path)
        pt2path = basename + ".pt2"
        try:
            torch.export.save(exported_program, pt2path)
        except Exception as e:
            raise TorchExportUnavailable(str(e)) from e

        # Existing tests invoke pnnx with their historical .pt path and expect
        # output files with the same basename. pnnx detects archive contents,
        # so copying the PT2 archive here preserves those test expectations.
        shutil.copyfile(pt2path, path)


def exported_program_trace(module, example_inputs=None, *args, **kwargs):
    del args
    return ExportedProgramTrace(module, example_inputs, kwargs.get("example_kwarg_inputs"))


def portable_system(command):
    if os.name == "nt" and command.startswith("../src/pnnx "):
        command = "..\\src\\pnnx.exe " + command[len("../src/pnnx "):]
    return original_system(command)


def idempotent_remove_parametrizations(module, tensor_name, leave_parametrized=True):
    if not torch.nn.utils.parametrize.is_parametrized(module, tensor_name):
        return module
    return original_remove_parametrizations(module, tensor_name, leave_parametrized=leave_parametrized)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_test_pt2.py TEST_SCRIPT")

    if not has_torch_export():
        print("PT2 test skipped because torch.export requires PyTorch 2.2 or later", file=sys.stderr)
        raise SystemExit(77)

    test_script = os.path.abspath(sys.argv[1])
    test_name = os.path.basename(test_script)
    if os.name == "nt" and test_name in ("test_torch_fft_hfftn.py", "test_torch_fft_ihfftn.py", "test_torch_fft_irfftn.py"):
        print("PT2 test skipped because this PyTorch Windows build crashes while evaluating the reference FFT model", file=sys.stderr)
        raise SystemExit(77)

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.dirname(test_script))

    original_trace = torch.jit.trace
    torch.jit.trace = exported_program_trace
    torch.nn.utils.parametrize.remove_parametrizations = idempotent_remove_parametrizations
    os.system = portable_system
    try:
        runpy.run_path(test_script, run_name="__main__")
    except TorchExportUnavailable as e:
        print("PT2 test skipped because torch.export.export rejected the model: " + str(e), file=sys.stderr)
        raise SystemExit(77)
    except ModuleNotFoundError as e:
        if e.name and not e.name.endswith("_pnnx"):
            print("PT2 test skipped because an optional Python dependency is unavailable: " + e.name, file=sys.stderr)
            raise SystemExit(77)
        raise
    except (ImportError, AttributeError) as e:
        frames = traceback.extract_tb(e.__traceback__)
        if any("onnx" in frame.filename.lower() for frame in frames):
            print("PT2 test skipped because the optional ONNX Python environment is unavailable: " + str(e), file=sys.stderr)
            raise SystemExit(77)
        raise
