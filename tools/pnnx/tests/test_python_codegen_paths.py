#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HELPER = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted_name(node.value) + "." + node.attr
    return ""


class PythonCodegenPathTest(unittest.TestCase):
    def test_path_literals_preserve_exact_values(self):
        paths = (
            r"C:\Users\runner\AppData\Local\Temp\model.pnnx.bin",
            r"C:\new\tensor\runtime.pnnx.bin",
            r"\\server\share\model.pnnx.bin",
            "model's \"quoted\".pnnx.bin",
            "model with spaces.pnnx.bin",
            "trailing\\",
            "line\ncarriage\rtab\tcontrol\x01af\x7f.pnnx.bin",
            "plain.pnnx.bin",
        )
        with tempfile.TemporaryDirectory() as directory:
            # An apostrophe is legal on Windows too. On POSIX also reproduce
            # Windows separators in the actual output filename.
            stem = "model's" if os.name == "nt" else "C:\\Users\\model's\"quoted\""
            output = Path(directory) / (stem + ".py")
            for path in paths:
                with self.subTest(path=path):
                    subprocess.run([str(HELPER), "--python-paths", str(output), path], check=True)
                    source = output.read_text(encoding="utf-8")
                    tree = ast.parse(source, filename=str(output))
                    compile(tree, str(output), "exec")
                    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
                    def arguments(name, index):
                        return [ast.literal_eval(call.args[index]) for call in calls
                                if isinstance(call.func, ast.Attribute)
                                and dotted_name(call.func) == name]
                    self.assertEqual(arguments("zipfile.ZipFile", 0), [path])
                    self.assertEqual(arguments("mod.save", 0), [str(output) + ".pt"])
                    self.assertEqual(arguments("torch.export.save", 1), [str(output.with_suffix(".pt2"))])
                    self.assertEqual(arguments("torch.onnx.export", 2), [str(output) + ".onnx"])
                    self.assertEqual(arguments("pnnx.export", 1), [str(output) + ".pt"])


if __name__ == "__main__":
    if HELPER is None or not HELPER.is_file():
        raise SystemExit("usage: test_python_codegen_paths.py ROUNDTRIP_HELPER [unittest options]")
    unittest.main()
