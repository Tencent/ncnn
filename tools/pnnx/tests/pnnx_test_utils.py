# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import math
import os
import subprocess
import sys

import torch


def has_exported_program():
    return callable(getattr(getattr(torch, "export", None), "save", None))


def model_formats():
    requested = os.environ.get("PNNX_TEST_FORMAT")
    if requested == "torchscript":
        return ("torchscript",)
    if requested == "pt2":
        if not has_exported_program():
            raise RuntimeError("PT2 tests require torch.export.save in torch " + torch.__version__)
        return ("pt2",)
    if requested:
        raise RuntimeError("unknown PNNX_TEST_FORMAT " + requested)
    if has_exported_program():
        return ("torchscript", "pt2")
    print("SKIP PT2: torch.export.save is unavailable in torch " + torch.__version__)
    return ("torchscript",)


def find_pnnx():
    candidates = [
        os.path.join("..", "src", "pnnx"),
        os.path.join("..", "src", "pnnx.exe"),
        os.path.join("..", "src", "Release", "pnnx.exe"),
        os.path.join("src", "pnnx"),
        os.path.join("src", "pnnx.exe"),
        os.path.join("src", "Release", "pnnx.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError("pnnx executable was not found")


def export_model(model, inputs, name, model_format, dynamic_shapes=None, check_trace=True):
    if model_format == "torchscript":
        path = name + "_torchscript.pt"
    elif model_format == "pt2":
        if not has_exported_program():
            raise RuntimeError("torch.export.save is unavailable in torch " + torch.__version__)
        path = name + "_pt2.pt2"
    else:
        raise ValueError("unknown model format " + model_format)

    _remove_file(path)
    try:
        if model_format == "torchscript":
            torch.jit.trace(model, inputs, check_trace=check_trace).save(path)
        else:
            torch.export.save(
                torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes),
                path,
            )
    except BaseException:
        _remove_file(path)
        raise
    return path


_OUTPUT_SUFFIXES = (
    ("pnnxparam", ".pnnx.param"),
    ("pnnxbin", ".pnnx.bin"),
    ("pnnxpy", "_pnnx.py"),
    ("ncnnparam", ".ncnn.param"),
    ("ncnnbin", ".ncnn.bin"),
    ("ncnnpy", "_ncnn.py"),
)


def _remove_file(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _remove_python_cache(path):
    # Exact cache names only: never glob a prefix or remove a shared __pycache__.
    for optimization in ("", "1", "2"):
        _remove_file(importlib.util.cache_from_source(path, optimization=optimization))


def _cleanup_outputs(output_prefix):
    for _, suffix in _OUTPUT_SUFFIXES:
        path = output_prefix + suffix
        _remove_file(path)
        if suffix.endswith(".py"):
            _remove_python_cache(path)


def _timeout_seconds(timeout):
    # Applies to every converter invocation, including expected failures.
    value = os.environ.get("PNNX_TEST_TIMEOUT", "300") if timeout is None else timeout
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError("PNNX_TEST_TIMEOUT/timeout must be a finite positive number") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("PNNX_TEST_TIMEOUT/timeout must be a finite positive number")
    return seconds


def _output_text(output):
    # TimeoutExpired can carry bytes even with text=True, or None before output.
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output or ""


def _process_diagnostic(command, returncode, stdout, stderr):
    return ("command: " + "\n  ".join(map(str, command)) + "\nreturn code: " + str(returncode)
            + "\nstdout:\n" + _output_text(stdout) + "\nstderr:\n" + _output_text(stderr))


def _reject_crash(result):
    code = result.returncode
    # main()/load_exported_program() use return -1 for ordinary diagnostics.
    # Windows preserves that DWORD (some wrappers expose it signed); POSIX
    # truncates it to 255. Neither is an exception or 128 + a valid signal.
    if code == 0xffffffff or (sys.platform == "win32" and code == -1):
        return
    # Negative POSIX signals and signed Windows statuses; unsigned NTSTATUS
    # exceptions (including warning-severity breakpoint/single-step statuses).
    # Reserve shell-style 128+signal exits for signals 1..64 too, including
    # Linux realtime signals. Unlike 255, these must not satisfy a failure needle.
    if (code < 0 or 129 <= code <= 192 or 0x80000000 <= code <= 0xffffffff
            or code == 0x40000015):  # STATUS_FATAL_APP_EXIT
        raise RuntimeError("pnnx crashed\n" + _process_diagnostic(
            result.args, code, result.stdout, result.stderr))


def run_pnnx(model_path, output_prefix, arguments=(), capture_output=False, timeout=None):
    """Run with a bounded lifetime; capture diagnostics even when echoing output.

    Ordinary nonzero exits are returned for existing negative-test callers.
    Crashes, launch errors and timeouts always raise, before any caller can
    treat their output as an expected unsupported-feature diagnostic.
    """
    seconds = _timeout_seconds(timeout)
    arguments = tuple(arguments)
    output_keys = {key for key, _ in _OUTPUT_SUFFIXES}
    if any(argument.split("=", 1)[0] in output_keys for argument in arguments):
        raise ValueError("output arguments must use the chosen output_prefix")
    command = [
        find_pnnx(),
        model_path,
        *(key + "=" + output_prefix + suffix for key, suffix in _OUTPUT_SUFFIXES),
        *arguments,
    ]
    _cleanup_outputs(output_prefix)
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=seconds,
        )
    except subprocess.TimeoutExpired as exception:
        _cleanup_outputs(output_prefix)
        raise RuntimeError("pnnx timed out after " + str(seconds) + " seconds\n"
                           + _process_diagnostic(command, "timeout", exception.stdout,
                                                 exception.stderr)) from exception
    except OSError as exception:
        _cleanup_outputs(output_prefix)
        raise RuntimeError("could not launch pnnx: " + str(exception)
                           + "\ncommand: " + "\n  ".join(map(str, command))) from exception

    if not capture_output:
        print(_output_text(result.stdout), end="", file=sys.stdout)
        print(_output_text(result.stderr), end="", file=sys.stderr)
    if result.returncode != 0:
        _cleanup_outputs(output_prefix)
    _reject_crash(result)
    return result


def convert_model(model_path, output_prefix, arguments=(), timeout=None):
    result = run_pnnx(model_path, output_prefix, arguments, timeout=timeout)
    generated_path = output_prefix + "_pnnx.py"
    if result.returncode != 0:
        raise RuntimeError("pnnx conversion failed for " + model_path + "\n"
                           + _process_diagnostic(result.args, result.returncode,
                                                 result.stdout, result.stderr))
    if not os.path.isfile(generated_path):
        _cleanup_outputs(output_prefix)
        raise RuntimeError("pnnx did not generate " + generated_path + "\n"
                           + _process_diagnostic(result.args, result.returncode,
                                                 result.stdout, result.stderr))
    return generated_path


def import_model(path, module_name=None):
    name = module_name or os.path.splitext(os.path.basename(path))[0]
    # A regenerated source may have the same timestamp and size as the old one.
    _remove_python_cache(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def export_convert_import(model, inputs, name, model_format, arguments=(), check_trace=True):
    model_path = export_model(model, inputs, name, model_format, check_trace=check_trace)
    output_prefix = name + "_" + model_format
    generated_path = convert_model(model_path, output_prefix, arguments)
    return import_model(generated_path, output_prefix + "_pnnx")


def _compare_outputs(expected, actual, compare, path="output", top_level=True):
    # The generated PNNX boundary represents a single output as a tensor, even
    # when the eager model returns (tensor,). Preserve that historical contract
    # in both directions, but only at the root and only for a tensor singleton.
    # Do not flatten nested trees or silently equate lists with tuples.
    if top_level:
        if isinstance(expected, tuple) and len(expected) == 1 and torch.is_tensor(actual):
            if torch.is_tensor(expected[0]):
                expected = expected[0]
        if isinstance(actual, tuple) and len(actual) == 1 and torch.is_tensor(expected):
            if torch.is_tensor(actual[0]):
                actual = actual[0]

    if isinstance(expected, (tuple, list)):
        kind = tuple if isinstance(expected, tuple) else list
        if not isinstance(actual, kind) or len(expected) != len(actual):
            print(path + ": output structure mismatch")
            return False
        return all(_compare_outputs(a, b, compare, path + "[" + str(i) + "]", False)
                   for i, (a, b) in enumerate(zip(expected, actual)))
    if not torch.is_tensor(expected) or not torch.is_tensor(actual):
        print(path + ": expected matching tensor leaves")
        return False
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        print(path + ": tensor metadata mismatch: "
              + str(expected.dtype) + " " + str(tuple(expected.shape)) + " != "
              + str(actual.dtype) + " " + str(tuple(actual.shape)))
        return False
    # Keep the caller's exact comparator/tolerances; metadata checks must not
    # replace torch.equal with allclose or permit allclose broadcasting.
    if not compare(expected, actual):
        print(path + ": tensor values differ")
        return False
    return True


def test_model_formats(model, inputs, expected, name, compare=torch.equal, check_trace=True,
                       unsupported_by_torch_export=None, unsupported_by_pnnx_pt2=None,
                       converted_inputs=None, torchscript_inputs=None, pt2_inputs=None):
    for needle in (unsupported_by_torch_export, unsupported_by_pnnx_pt2):
        if needle is not None and (not isinstance(needle, str) or not needle.strip()):
            raise ValueError("expected failure must be a nonempty diagnostic string")
    if unsupported_by_torch_export is not None and unsupported_by_pnnx_pt2 is not None:
        raise ValueError("expected failures must identify exactly one failing stage")
    converted_inputs = inputs if converted_inputs is None else converted_inputs
    torchscript_inputs = converted_inputs if torchscript_inputs is None else torchscript_inputs
    pt2_inputs = converted_inputs if pt2_inputs is None else pt2_inputs
    for model_format in model_formats():
        if model_format == "pt2" and unsupported_by_torch_export:
            try:
                export_model(model, inputs, name, "pt2")
            except Exception as exception:
                if unsupported_by_torch_export in str(exception):
                    print("UNSUPPORTED_BY_TORCH_EXPORT: " + unsupported_by_torch_export)
                    continue
                raise
            raise RuntimeError("torch.export unexpectedly supports " + name)

        if model_format == "pt2" and unsupported_by_pnnx_pt2:
            model_path = export_model(model, inputs, name, "pt2")
            result = run_pnnx(model_path, name + "_pt2", capture_output=True)
            _reject_crash(result)
            output = _output_text(result.stdout) + "\n" + _output_text(result.stderr)
            if result.returncode != 0 and unsupported_by_pnnx_pt2 in output:
                print("UNSUPPORTED_BY_PNNX_PT2: " + unsupported_by_pnnx_pt2)
                continue
            if result.returncode == 0:
                reason = "pnnx unexpectedly supports PT2 conversion for "
            else:
                reason = "unexpected pnnx PT2 failure for "
            raise RuntimeError(reason + name + "\n" + _process_diagnostic(
                result.args, result.returncode, result.stdout, result.stderr))

        generated_model = export_convert_import(model, inputs, name, model_format, check_trace=check_trace)
        model_inputs = torchscript_inputs if model_format == "torchscript" else pt2_inputs
        result = generated_model(*model_inputs)
        if not _compare_outputs(expected, result, compare):
            return False

    return True