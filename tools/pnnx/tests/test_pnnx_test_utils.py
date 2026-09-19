#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import torch

import run_pt2_test
import pnnx_test_utils

from pnnx_test_utils import convert_and_import
from pnnx_test_utils import LEGACY_PT2_UNSUPPORTED
from pnnx_test_utils import SUPPORTED
from pnnx_test_utils import pt2_producer_status
from pt2_expectations import EXPORT_UNSUPPORTED
from pt2_expectations import PASS
from pt2_expectations import PT2_EXPECTED_FAILURES
from pt2_expectations import PT2_FRONTEND_UNSUPPORTED
from pt2_expectations import PNNX_LOWERING_UNSUPPORTED


class Pt2ProducerStatusTest(unittest.TestCase):
    def assert_status(self, torch_version, expected):
        with mock.patch.object(torch, "__version__", torch_version):
            self.assertEqual(pt2_producer_status(), expected)

    def test_supported_raw_payload_producers(self):
        self.assert_status("2.9.0", SUPPORTED)
        self.assert_status("2.12.1+cu126", SUPPORTED)
        self.assert_status("2.12.2", SUPPORTED)
        self.assert_status("2.13.0+cpu", SUPPORTED)
        self.assert_status("2.13.1", SUPPORTED)
        self.assert_status("2.14.0", SUPPORTED)
        self.assert_status("3.0.0.dev20260827", SUPPORTED)

    def test_unsupported_producers(self):
        self.assert_status("2.8.0", LEGACY_PT2_UNSUPPORTED)

    def test_unsupported_pt2_producers_exit_with_ctest_skip_code(self):
        # CTest reserves 77 for tests skipped by an unsupported producer.
        with mock.patch.dict("os.environ", {"PNNX_TEST_FORMAT": "pt2"}):
            with mock.patch.object(torch, "__version__", "2.8.0"):
                with self.assertRaises(SystemExit) as raised:
                    convert_and_import(None, (), "producer_gate")
        self.assertEqual(raised.exception.code, 77)

    def test_expected_pt2_failures_exit_with_ctest_skip_code(self):
        failure = RuntimeError("PendingUnbackedSymbolNotFound: Pending unbacked symbols")
        with mock.patch.dict(os.environ, {"PNNX_TEST_FORMAT": "pt2"}):
            with mock.patch.object(pnnx_test_utils, "pt2_producer_status", return_value=SUPPORTED):
                with mock.patch.object(torch, "export", create=True) as exporter:
                    exporter.export.side_effect = failure
                    with self.assertRaises(SystemExit) as raised:
                        convert_and_import(None, (), "test_Tensor_index")
        self.assertEqual(raised.exception.code, 77)

    def test_expected_failure_table_contract(self):
        categories = {
            EXPORT_UNSUPPORTED,
            PT2_FRONTEND_UNSUPPORTED,
            PNNX_LOWERING_UNSUPPORTED,
        }
        test_dir = Path(__file__).resolve().parent
        for name, (category, diagnostic) in PT2_EXPECTED_FAILURES.items():
            with self.subTest(name=name):
                self.assertTrue((test_dir / (name + ".py")).is_file())
                self.assertIn(category, categories)
                self.assertIsInstance(diagnostic, str)
                self.assertTrue(diagnostic)

    def test_expected_failure_stage_diagnostic_and_xpass_are_strict(self):
        name = "test_Tensor_index"
        diagnostic = PT2_EXPECTED_FAILURES[name][1]
        with self.assertRaisesRegex(AssertionError, "failure category changed"):
            pnnx_test_utils._handle_pt2_failure(
                name, PT2_FRONTEND_UNSUPPORTED, diagnostic
            )
        with self.assertRaisesRegex(AssertionError, "diagnostic changed"):
            pnnx_test_utils._handle_pt2_failure(
                name, EXPORT_UNSUPPORTED, "different failure"
            )
        with self.assertRaisesRegex(AssertionError, "conversion now passes"):
            pnnx_test_utils._handle_pt2_conversion_success(name)

        self.assertEqual(pnnx_test_utils.pt2_expectation("ordinary_test"), (PASS, ""))


class Pt2RunnerTest(unittest.TestCase):
    def test_unsupported_producer_does_not_start_test_script(self):
        arguments = [
            "run_pt2_test.py",
            "--pnnx",
            "pnnx",
            "--build-dir",
            "build",
            "legacy_early_success.py",
        ]
        with mock.patch.object(run_pt2_test, "pt2_producer_status", return_value=LEGACY_PT2_UNSUPPORTED):
            with mock.patch.object(run_pt2_test.torch, "__version__", "2.8.0"):
                with mock.patch.object(run_pt2_test.sys, "argv", arguments):
                    with mock.patch.object(run_pt2_test.subprocess, "call") as call:
                        result = run_pt2_test.main()

        self.assertEqual(result, 77)
        call.assert_not_called()

    def test_supported_producer_runs_script_with_pt2_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pnnx_path = Path(temp_dir) / "bin" / "pnnx"
            build_dir = Path(temp_dir) / "build"
            script_path = Path(temp_dir) / "test_example.py"
            arguments = [
                "run_pt2_test.py",
                "--pnnx",
                str(pnnx_path),
                "--build-dir",
                str(build_dir),
                str(script_path),
            ]
            with mock.patch.dict(os.environ, {"PYTHONPATH": "existing"}, clear=True):
                with mock.patch.object(run_pt2_test, "pt2_producer_status", return_value=SUPPORTED):
                    with mock.patch.object(run_pt2_test.sys, "argv", arguments):
                        with mock.patch.object(run_pt2_test.subprocess, "call", return_value=13) as call:
                            result = run_pt2_test.main()

        self.assertEqual(result, 13)
        command = call.call_args.args[0]
        environment = call.call_args.kwargs["env"]
        self.assertEqual(command, [run_pt2_test.sys.executable, str(script_path.resolve())])
        self.assertEqual(environment["PNNX_TEST_FORMAT"], "pt2")
        self.assertEqual(environment["PNNX_TEST_PNNX"], str(pnnx_path.resolve()))
        self.assertEqual(
            environment["PYTHONPATH"],
            str(build_dir.resolve()) + os.pathsep + "existing",
        )


class Pt2GeneratedArtifactTest(unittest.TestCase):
    def test_converter_exit_status_precedes_expected_diagnostic(self):
        diagnostic = "load exported program failed: dynamic tensor shapes are unsupported"
        expectation = {"exit_status": (PT2_FRONTEND_UNSUPPORTED, "dynamic tensor shapes are unsupported")}
        with tempfile.TemporaryDirectory() as temp_dir:
            basename = str(Path(temp_dir) / "exit_status")
            with mock.patch.dict(os.environ, {"PNNX_TEST_FORMAT": "pt2"}):
                with mock.patch.object(pnnx_test_utils, "pt2_producer_status", return_value=SUPPORTED):
                    with mock.patch.object(torch, "export", create=True):
                        with mock.patch.dict(PT2_EXPECTED_FAILURES, expectation, clear=True):
                            # main returns -1 as 255 on POSIX or 0xffffffff on Windows.
                            for code in (255, 0xffffffff, -11, -6, 0xc0000005, 0xc0000409):
                                with self.subTest(returncode=code):
                                    completed = subprocess.CompletedProcess([], code, "", diagnostic)
                                    with mock.patch.object(pnnx_test_utils.subprocess, "run", return_value=completed):
                                        try:
                                            convert_and_import(None, (), "exit_status", output_basename=basename)
                                        except (RuntimeError, SystemExit) as error:
                                            if code in (255, 0xffffffff):
                                                self.assertIsInstance(error, SystemExit)
                                                self.assertEqual(error.code, 77)
                                            else:
                                                self.assertIsInstance(error, RuntimeError)
                                                self.assertIn(str(code), str(error))
                                        else:
                                            self.fail("converter failure was ignored")

    def test_failed_pnnx_conversion_removes_stale_generated_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                stale_paths = [
                    Path("stale_pt2.pt2"),
                    Path("stale_pt2_pnnx.py"),
                    Path("stale_pt2_ncnn.py"),
                    Path("stale_pt2.pnnx.param"),
                    Path("stale_pt2.pnnx.bin"),
                    Path("stale_pt2.ncnn.param"),
                    Path("stale_pt2.ncnn.bin"),
                ]
                for path in stale_paths:
                    path.write_bytes(b"stale")

                failed = subprocess.CompletedProcess([], 1, "", "conversion failed")
                with mock.patch.dict(
                    os.environ,
                    {
                        "PNNX_TEST_FORMAT": "pt2",
                        "PNNX_TEST_PNNX": "pnnx",
                    },
                ):
                    with mock.patch.object(pnnx_test_utils, "pt2_producer_status", return_value=SUPPORTED):
                        with mock.patch.object(torch, "export", create=True):
                            with mock.patch("pnnx_test_utils.subprocess.run", return_value=failed):
                                with self.assertRaises(AssertionError):
                                    convert_and_import(None, (), "stale")

                self.assertFalse(any(path.exists() for path in stale_paths))
            finally:
                os.chdir(original_cwd)


class NcnnTestRuntimeTest(unittest.TestCase):
    option_names = (
        "use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic",
        "use_bf16_storage",
    )

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "precision_ncnn.py"
        self.source = (
            "import ncnn\n"
            "def test_inference():\n"
            "    with ncnn.Net() as net:\n"
            "        net.load_param('model.param')\n"
            "        net.load_model('model.bin')\n"
            "        return net\n"
        )
        self.path.write_text(self.source)
        self.binding = types.ModuleType("ncnn")
        option_names = self.option_names

        class Net:
            def __init__(self, *args, **kwargs):
                self.arguments = (args, kwargs)
                self.opt = types.SimpleNamespace(**dict.fromkeys(option_names, True))
                self.observed_options = []

            def __enter__(self):
                self.observed_options.append(vars(self.opt).copy())
                return self

            def __exit__(self, *args):
                return False

            def load_param(self, path):
                self.observed_options.append(vars(self.opt).copy())

            def load_model(self, path):
                self.observed_options.append(vars(self.opt).copy())

        self.binding.Net = Net
        self.binding.Mat = object()
        self.original_net = Net

    def import_module(self, path=None):
        with mock.patch.dict(pnnx_test_utils.sys.modules, {"ncnn": self.binding}):
            module = pnnx_test_utils._import_generated_module(
                path or self.path, "native_precision_regression"
            )
        self.addCleanup(pnnx_test_utils.sys.modules.pop, module.__name__, None)
        return module

    def test_fp32_options_are_set_before_context_and_model_loading(self):
        net = self.import_module().test_inference()
        self.assertEqual(len(net.observed_options), 3)
        for observed in net.observed_options:
            self.assertEqual(observed, dict.fromkeys(self.option_names, False))

    def test_native_binding_defaults_are_not_changed_globally(self):
        self.import_module().test_inference()
        self.assertIs(self.binding.Net, self.original_net)
        self.assertEqual(
            vars(self.binding.Net().opt), dict.fromkeys(self.option_names, True)
        )

    def test_other_binding_attributes_are_forwarded(self):
        module = self.import_module()
        self.assertIs(module.ncnn.Mat, self.binding.Mat)

    def test_constructor_arguments_and_independent_options_are_preserved(self):
        module = self.import_module()
        first = module.ncnn.Net("test", marker=True)
        second = module.ncnn.Net()
        self.assertEqual(first.arguments, (("test",), {"marker": True}))
        self.assertIsNot(first.opt, second.opt)
        first.opt.use_fp16_storage = True
        self.assertFalse(second.opt.use_fp16_storage)

    def test_pnnx_module_import_does_not_wrap_native_binding(self):
        path = self.path.with_name("precision_pnnx.py")
        path.write_text(self.source)
        module = self.import_module(path)
        self.assertIs(module.ncnn, self.binding)
        self.assertTrue(module.test_inference().opt.use_fp16_storage)

    def test_generated_deployment_source_is_not_modified(self):
        original = self.path.read_bytes()
        self.import_module().test_inference()
        self.assertEqual(self.path.read_bytes(), original)

    def test_native_inference_errors_are_not_suppressed(self):
        module = self.import_module()
        with mock.patch.object(self.original_net, "load_model", side_effect=RuntimeError("load failed")):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                module.test_inference()


if __name__ == "__main__":
    unittest.main()
