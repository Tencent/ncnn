# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

"""Harness contract tests: no converter executable or real exporter required.

Use unittest (also discoverable by pytest), including on historical torch
versions without torch.export. All files live in a private temporary directory.
"""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import pnnx_test_utils as utils


class TemporaryTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pnnx-harness-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.name = str(self.root / "model")
        self.prefix = self.name + "_pt2"
        self.model_path = self.prefix + ".pt2"
        self.patch(mock.patch.dict(os.environ, PNNX_TEST_FORMAT="pt2", PNNX_TEST_TIMEOUT="10"))

    def patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def write(self, path, text="stale"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def artifacts(self, prefix):
        paths = [prefix + suffix for suffix in (
            ".pnnx.param", ".pnnx.bin", "_pnnx.py",
            ".ncnn.param", ".ncnn.bin", "_ncnn.py",
        )]
        for source in (prefix + "_pnnx.py", prefix + "_ncnn.py"):
            paths.extend(importlib.util.cache_from_source(source, optimization=level)
                         for level in ("", "1", "2"))
        return paths


class FormatTests(unittest.TestCase):
    def test_default_and_explicit_formats_with_old_and_new_producers(self):
        for export, default in ((None, ("torchscript",)),
                                (SimpleNamespace(save=mock.Mock()), ("torchscript", "pt2"))):
            producer = SimpleNamespace(__version__="mock")
            if export is not None:
                producer.export = export
            with self.subTest(default=default), mock.patch.object(utils, "torch", producer):
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(utils.model_formats(), default)
                with mock.patch.dict(os.environ, PNNX_TEST_FORMAT="torchscript"):
                    self.assertEqual(utils.model_formats(), ("torchscript",))
                with mock.patch.dict(os.environ, PNNX_TEST_FORMAT="pt2"):
                    if export is None:
                        with self.assertRaisesRegex(RuntimeError, "torch.export.save"):
                            utils.model_formats()
                    else:
                        self.assertEqual(utils.model_formats(), ("pt2",))

    def test_unknown_format_and_noncallable_save(self):
        with mock.patch.dict(os.environ, PNNX_TEST_FORMAT="unknown"):
            with self.assertRaisesRegex(RuntimeError, "unknown PNNX_TEST_FORMAT"):
                utils.model_formats()
        with mock.patch.object(utils, "torch", SimpleNamespace(export=SimpleNamespace(save=None))):
            self.assertFalse(utils.has_exported_program())


class ExportTests(TemporaryTestCase):
    def test_torchscript_export_preserves_flags_and_other_format(self):
        path = self.name + "_torchscript.pt"
        self.write(path)
        other = self.write(self.model_path)
        traced = mock.Mock()

        def save(destination):
            self.assertEqual(destination, path)
            self.assertFalse(Path(path).exists())
            self.write(path, "fresh")

        traced.save.side_effect = save
        with mock.patch.object(utils.torch.jit, "trace", return_value=traced) as trace:
            self.assertEqual(utils.export_model("model", ("input",), self.name,
                                                "torchscript", check_trace=False), path)
            trace.assert_called_once_with("model", ("input",), check_trace=False)
        self.assertEqual(other.read_text(), "stale")

    def test_pt2_export_and_failed_export_remove_only_selected_model(self):
        exporter = SimpleNamespace(export=mock.Mock(return_value="program"), save=mock.Mock())
        self.patch(mock.patch.object(utils.torch, "export", exporter, create=True))
        self.write(self.model_path)
        other = self.write(self.name + "_torchscript.pt")

        def save(program, destination):
            self.assertEqual(program, "program")
            self.assertFalse(Path(destination).exists())
            self.write(destination, "fresh")

        exporter.save.side_effect = save
        self.assertEqual(utils.export_model("model", ("input",), self.name, "pt2",
                                            dynamic_shapes="shapes"), self.model_path)
        exporter.export.assert_called_once_with("model", ("input",), dynamic_shapes="shapes")

        def failed_save(program, destination):
            self.write(destination, "partial")
            raise RuntimeError("serializer failed")

        exporter.save.side_effect = failed_save
        with self.assertRaisesRegex(RuntimeError, "serializer failed"):
            utils.export_model("model", (), self.name, "pt2")
        self.assertFalse(Path(self.model_path).exists())
        self.assertEqual(other.read_text(), "stale")

    def test_unknown_export_format(self):
        with self.assertRaisesRegex(ValueError, "unknown model format"):
            utils.export_model(None, (), self.name, "unknown")


class ProcessTests(TemporaryTestCase):
    def setUp(self):
        super().setUp()
        # These tests exercise conversion, not producer capability; do not
        # require the installed torch to have export (see FormatTests instead).
        self.patch(mock.patch.object(utils, "model_formats", return_value=("pt2",)))
        self.patch(mock.patch.object(utils, "find_pnnx", return_value="mock-pnnx"))
        self.process_run = self.patch(mock.patch.object(utils.subprocess, "run"))
        self.process_run.return_value = self.completed()

    def completed(self, code=0, stdout="converter stdout", stderr="converter stderr"):
        return subprocess.CompletedProcess(["mock-pnnx", self.model_path], code, stdout, stderr)

    def expect_unsupported(self, needle="unsupported operator"):
        with mock.patch.object(utils, "export_model", return_value=self.model_path):
            return utils.test_model_formats(None, (), None, self.name,
                                           unsupported_by_pnnx_pt2=needle)

    def test_success_cleans_exact_artifacts_before_running(self):
        selected = self.artifacts(self.prefix)
        preserved = (self.artifacts(self.name + "_torchscript")
                     + self.artifacts(self.prefix + "_neighbor")
                     + [self.model_path, self.prefix + ".unrelated", self.prefix + "_pnnx.py.backup"])
        for path in selected + preserved:
            self.write(path)

        def convert(command, **kwargs):
            self.assertTrue(all(not Path(path).exists() for path in selected))
            self.assertTrue(all(Path(path).exists() for path in preserved))
            self.assertIn("pnnxpy=" + self.prefix + "_pnnx.py", command)
            self.assertIn("inputshape=[1,3]", command)
            self.assertEqual(kwargs["timeout"], 10.0)
            self.assertTrue(kwargs["capture_output"])
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["errors"], "replace")
            self.write(self.prefix + "_pnnx.py", "fresh")
            return self.completed()

        self.process_run.side_effect = convert
        self.assertEqual(utils.convert_model(self.model_path, self.prefix, ("inputshape=[1,3]",)),
                         self.prefix + "_pnnx.py")
        self.assertTrue(all(Path(path).read_text() == "stale" for path in preserved))

    def test_success_without_new_script_cannot_reuse_stale_script(self):
        self.write(self.prefix + "_pnnx.py")
        with self.assertRaisesRegex(RuntimeError, "did not generate") as error:
            utils.convert_model(self.model_path, self.prefix)
        self.assertIn("converter stdout", str(error.exception))
        self.assertIn("converter stderr", str(error.exception))
        self.assertFalse(Path(self.prefix + "_pnnx.py").exists())

    def test_nonzero_exit_returns_output_but_removes_partial_artifacts(self):
        def fail(*args, **kwargs):
            for path in self.artifacts(self.prefix):
                self.write(path, "partial")
            return self.completed(1)

        self.process_run.side_effect = fail
        result = utils.run_pnnx(self.model_path, self.prefix, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "converter stdout")
        self.assertEqual(result.stderr, "converter stderr")
        self.assertTrue(all(not Path(path).exists() for path in self.artifacts(self.prefix)))
        with self.assertRaisesRegex(RuntimeError, "conversion failed") as error:
            utils.convert_model(self.model_path, self.prefix)
        for text in (self.model_path, "return code: 1", "converter stdout", "converter stderr"):
            self.assertIn(text, str(error.exception))

    def test_crashes_never_satisfy_expected_failure(self):
        codes = (-11, -6, -1073741819, -2147483645, 0xc0000005, 0xc0000409,
                 0x80000003, 0x40000015, 129, 134, 137, 139, 192)
        for code in codes:
            with self.subTest(code=code):
                self.process_run.return_value = self.completed(code, "unsupported operator", "crash details")
                with self.assertRaisesRegex(RuntimeError, "pnnx crashed") as error:
                    self.expect_unsupported()
                for text in (str(code), "unsupported operator", "crash details", self.model_path):
                    self.assertIn(text, str(error.exception))

    def test_expected_failure_matches_either_stream_only_on_normal_failure(self):
        for code in (1, 255, 0xffffffff):
            for stdout, stderr in (("unsupported operator", None), (None, "unsupported operator")):
                with self.subTest(code=code, stdout=stdout, stderr=stderr):
                    self.process_run.return_value = self.completed(code, stdout, stderr)
                    self.assertTrue(self.expect_unsupported())

    def test_signed_minus_one_is_a_signal_on_posix_but_application_exit_on_windows(self):
        self.process_run.return_value = self.completed(-1, "unsupported operator", "")
        with mock.patch.object(utils.sys, "platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "pnnx crashed"):
                self.expect_unsupported()
        with mock.patch.object(utils.sys, "platform", "win32"):
            self.assertTrue(self.expect_unsupported())

    def test_unexpected_success_and_unknown_failure_include_both_streams(self):
        for code, reason in ((0, "unexpectedly supports"), (1, "unexpected pnnx PT2 failure")):
            with self.subTest(code=code):
                self.process_run.return_value = self.completed(code)
                with self.assertRaisesRegex(RuntimeError, reason) as error:
                    self.expect_unsupported()
                self.assertIn("converter stdout", str(error.exception))
                self.assertIn("converter stderr", str(error.exception))
        # Even a success that prints the needle is not an expected failure.
        self.process_run.return_value = self.completed(0, "unsupported operator", "")
        with self.assertRaisesRegex(RuntimeError, "unexpectedly supports"):
            self.expect_unsupported()

    def test_timeout_rejected_before_needle_match_with_partial_output(self):
        def hang(*args, **kwargs):
            self.write(self.prefix + "_pnnx.py", "partial")
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"],
                                            output=b"unsupported operator\xff", stderr=b"timeout stderr")

        self.process_run.side_effect = hang
        with self.assertRaisesRegex(RuntimeError, "timed out") as error:
            self.expect_unsupported()
        self.assertIn("unsupported operator", str(error.exception))
        self.assertIn("timeout stderr", str(error.exception))
        self.assertFalse(Path(self.prefix + "_pnnx.py").exists())
        self.process_run.side_effect = subprocess.TimeoutExpired("mock-pnnx", 1)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            utils.run_pnnx(self.model_path, self.prefix)

    def test_timeout_override_and_invalid_timeouts(self):
        utils.run_pnnx(self.model_path, self.prefix, capture_output=True, timeout=2.5)
        self.assertEqual(self.process_run.call_args[1]["timeout"], 2.5)
        for value in ("0", "-1", "nan", "inf", "invalid", ""):
            with self.subTest(value=value), mock.patch.dict(os.environ, PNNX_TEST_TIMEOUT=value):
                self.process_run.reset_mock()
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    utils.run_pnnx(self.model_path, self.prefix)
                self.process_run.assert_not_called()

    def test_launch_failure_is_not_an_expected_unsupported_feature(self):
        self.process_run.side_effect = OSError("unsupported operator: cannot launch")
        with self.assertRaisesRegex(RuntimeError, "could not launch pnnx") as error:
            self.expect_unsupported()
        self.assertIn(self.model_path, str(error.exception))

    def test_capture_flag_controls_echo_not_diagnostic_retention(self):
        for capture in (False, True):
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = utils.run_pnnx(self.model_path, self.prefix, capture_output=capture)
            self.assertEqual(stdout.getvalue(), "" if capture else "converter stdout")
            self.assertEqual(stderr.getvalue(), "" if capture else "converter stderr")
            self.assertEqual(result.stdout, "converter stdout")
            self.assertEqual(result.stderr, "converter stderr")

    def test_output_override_is_rejected_without_deleting_other_files(self):
        other = self.write(self.name + "_other.py")
        with self.assertRaisesRegex(ValueError, "output_prefix"):
            utils.run_pnnx(self.model_path, self.prefix, ("pnnxpy=" + str(other),))
        self.process_run.assert_not_called()
        self.assertEqual(other.read_text(), "stale")


class ExpectationTests(unittest.TestCase):
    def test_exporter_expected_failure_unknown_failure_and_unexpected_success(self):
        with mock.patch.object(utils, "model_formats", return_value=("pt2",)), \
                mock.patch.object(utils, "export_model") as export, \
                mock.patch.object(utils, "run_pnnx") as run:
            export.side_effect = RuntimeError("known export limitation")
            self.assertTrue(utils.test_model_formats(None, (), None, "model",
                                                    unsupported_by_torch_export="known export limitation"))
            export.side_effect = RuntimeError("unknown exporter error")
            with self.assertRaisesRegex(RuntimeError, "unknown exporter error"):
                utils.test_model_formats(None, (), None, "model",
                                         unsupported_by_torch_export="known export limitation")
            export.side_effect = None
            with self.assertRaisesRegex(RuntimeError, "unexpectedly supports"):
                utils.test_model_formats(None, (), None, "model",
                                         unsupported_by_torch_export="known export limitation")
            export.side_effect = SystemExit("known export limitation")
            with self.assertRaises(SystemExit):
                utils.test_model_formats(None, (), None, "model",
                                         unsupported_by_torch_export="known export limitation")
            run.assert_not_called()

    def test_invalid_or_ambiguous_expectations_are_not_silent_passes(self):
        for key in ("unsupported_by_torch_export", "unsupported_by_pnnx_pt2"):
            for value in ("", "   ", True, ["unsupported"]):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, "nonempty diagnostic"):
                        utils.test_model_formats(None, (), None, "model", **{key: value})
        with self.assertRaisesRegex(ValueError, "exactly one failing stage"):
            utils.test_model_formats(None, (), None, "model", unsupported_by_torch_export="one",
                                     unsupported_by_pnnx_pt2="two")


class ComparisonTests(unittest.TestCase):
    def check(self, expected, actual, compare=torch.equal):
        with mock.patch.object(utils, "model_formats", return_value=("torchscript", "pt2")), \
                mock.patch.object(utils, "export_convert_import", return_value=lambda *args: actual):
            return utils.test_model_formats(None, (), expected, "mock-model", compare=compare)

    def test_dtype_and_shape_checked_before_comparator(self):
        expected = torch.tensor([1.0, 2.0])
        for actual in (expected.to(torch.float64), expected.to(torch.int64), expected.reshape(1, 2)):
            with self.subTest(dtype=actual.dtype, shape=actual.shape):
                compare = mock.Mock(return_value=True)
                self.assertFalse(self.check(expected, actual, compare))
                compare.assert_not_called()
        # Broadcasting by torch.allclose must not hide an incompatible shape.
        self.assertFalse(self.check(expected, expected.reshape(1, 2), torch.allclose))
        self.assertFalse(self.check(expected, expected.to(torch.float64)))

    def test_default_exact_comparator_is_not_loosened(self):
        expected = torch.tensor([1.0, 2.0])
        changed = expected + 1e-6
        self.assertTrue(self.check(expected, expected.clone()))
        self.assertFalse(self.check(expected, changed))
        self.assertTrue(self.check(expected, changed, torch.allclose))

    def test_root_tensor_singleton_tuple_compatibility_only(self):
        tensor = torch.tensor([1.0])
        self.assertTrue(self.check((tensor,), tensor))
        self.assertTrue(self.check(tensor, (tensor,)))
        self.assertFalse(self.check([tensor], tensor))
        self.assertFalse(self.check(tensor, [tensor]))
        self.assertFalse(self.check(((tensor,),), tensor))
        self.assertFalse(self.check(((tensor,), tensor), (tensor, tensor)))

    def test_nested_tuple_list_types_lengths_and_leaves_are_preserved(self):
        tensor = torch.tensor([1.0])
        expected = (tensor, [tensor, (tensor, tensor)])
        self.assertTrue(self.check(expected, expected))
        for actual in ((tensor, tensor, tensor, tensor),
                       (tensor, (tensor, (tensor, tensor))),
                       (tensor, [tensor, [tensor, tensor]]),
                       (tensor, [tensor, (tensor,)]),
                       (tensor, [tensor, (tensor, tensor.to(torch.float64))])):
            with self.subTest(actual=actual):
                self.assertFalse(self.check(expected, actual))
        self.assertFalse(self.check((tensor, tensor), [tensor, tensor]))
        self.assertFalse(self.check((tensor, tensor), (tensor,)))

    def test_existing_format_specific_inputs_and_trace_flag(self):
        tensor = torch.tensor([1.0])
        generated = mock.Mock(return_value=tensor)
        with mock.patch.object(utils, "model_formats", return_value=("torchscript", "pt2")), \
                mock.patch.object(utils, "export_convert_import", return_value=generated) as convert:
            self.assertTrue(utils.test_model_formats(
                "model", ("export",), tensor, "name", check_trace=False,
                converted_inputs=("fallback",), torchscript_inputs=("ts",), pt2_inputs=("pt2",)))
            self.assertEqual(generated.call_args_list, [mock.call("ts"), mock.call("pt2")])
            self.assertEqual(convert.call_args_list, [
                mock.call("model", ("export",), "name", "torchscript", check_trace=False),
                mock.call("model", ("export",), "name", "pt2", check_trace=False),
            ])


class ImportTests(TemporaryTestCase):
    def test_same_size_same_mtime_source_does_not_load_stale_bytecode(self):
        path = self.prefix + "_pnnx.py"
        source = "class Model:\n    value = 1\n    def eval(self):\n        return self\n"
        file = self.write(path, source)
        stamp = file.stat()
        self.assertEqual(utils.import_model(path).value, 1)
        self.write(path, source.replace("value = 1", "value = 2"))
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        self.assertEqual(utils.import_model(path).value, 2)


if __name__ == "__main__":
    unittest.main()