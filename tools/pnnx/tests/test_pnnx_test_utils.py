#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import torch

import run_pt2_test

from pnnx_test_utils import convert_and_import
from pnnx_test_utils import LEGACY_PT2_UNSUPPORTED
from pnnx_test_utils import SUPPORTED
from pnnx_test_utils import pt2_producer_status


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
            with mock.patch.object(torch.export, "export", side_effect=failure):
                with self.assertRaises(SystemExit) as raised:
                    convert_and_import(None, (), "test_Tensor_index")
        self.assertEqual(raised.exception.code, 77)


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
                    with mock.patch.object(torch.export, "export", return_value=object()):
                        with mock.patch.object(torch.export, "save"):
                            with mock.patch("pnnx_test_utils.subprocess.run", return_value=failed):
                                with self.assertRaises(AssertionError):
                                    convert_and_import(None, (), "stale")

                self.assertFalse(any(path.exists() for path in stale_paths))
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
