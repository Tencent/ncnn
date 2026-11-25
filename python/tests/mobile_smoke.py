# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause
"""Dependency-free native binding/inference smoke test for mobile wheels.

This intentionally does not cover NumPy interoperability, model-zoo utilities,
or installation of the wheel's runtime dependencies.
"""

import importlib.util
import sys
import unittest

import ncnn


class MobileCoreSmokeTest(unittest.TestCase):
    def test_import_without_opencv(self):
        self.assertIsNone(importlib.util.find_spec("cv2"))
        self.assertNotIn("cv2", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertTrue(ncnn.__version__)

    def test_cpu_only(self):
        self.assertFalse(hasattr(ncnn, "get_gpu_count"))

    def test_relu_inference(self):
        # no model files, network access, or numpy conversions are needed
        param = """7767517
2 2
Input input 0 1 data
ReLU relu 1 1 data output 0=0
"""
        with ncnn.Net() as net:
            net.opt.num_threads = 1
            net.opt.use_vulkan_compute = False
            self.assertEqual(net.load_param_mem(param), 0)
            self.assertEqual(net.load_model(ncnn.DataReaderFromEmpty()), 0)
            values = (-2.0, -0.5, 0.0, 3.0)
            tensor = ncnn.Mat(len(values))
            for index, value in enumerate(values):
                tensor[index] = value
            with net.create_extractor() as extractor:
                self.assertEqual(extractor.input("data", tensor), 0)
                result, output = extractor.extract("output")
                self.assertEqual(result, 0)
                self.assertEqual(output.w, len(values))
                self.assertEqual([output[i] for i in range(output.w)], [0.0, 0.0, 0.0, 3.0])
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("cv2", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
