# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import os
import runpy
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import Mock

import numpy as np
import pytest
import setuptools

from ncnn import utils
from ncnn.model_zoo.yolact import Yolact
from ncnn.utils._opencv import _require_cv2


def test_import_without_opencv():
    script = textwrap.dedent("""
        import builtins
        original_import = builtins.__import__
        def without_cv2(name, *args, **kwargs):
            if name == 'cv2':
                raise AssertionError('cv2 imported eagerly')
            return original_import(name, *args, **kwargs)
        builtins.__import__ = without_cv2
        import ncnn
        import ncnn.utils as utils
        import ncnn.model_zoo as model_zoo
        from ncnn.model_zoo.yolact import Yolact
        assert ncnn.Net
        assert utils.Rect(0, 0, 2, 3).area() == 6
        assert 'yolact' in model_zoo.get_model_list()
        for name in ('draw_detection_objects', 'draw_faceobjects', 'draw_pose', 'print_topk'):
            assert callable(getattr(utils, name))
        utils.print_topk(utils.np.array([0.1, 0.9]), 1)
    """)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(os.path.abspath(path) for path in sys.path)
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=30)


def block_cv2(monkeypatch, error):
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "cv2":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)


@pytest.mark.parametrize("operation", [
    lambda: utils.draw_detection_objects(None, None, None),
    lambda: utils.draw_faceobjects(None, None),
    lambda: utils.draw_pose(None, None),
    lambda: Yolact.__call__(None, None),
])
def test_missing_opencv(monkeypatch, operation):
    block_cv2(monkeypatch, ModuleNotFoundError("No module named 'cv2'", name="cv2"))
    with pytest.raises(ImportError, match=r'ncnn\[opencv\]'):
        operation()


@pytest.mark.parametrize("error", [
    ModuleNotFoundError("No module named 'dependency'", name="dependency"),
    ImportError("numpy ABI mismatch", name="cv2"),
    OSError("shared library unavailable"),
])
def test_opencv_import_errors_unchanged(monkeypatch, error):
    block_cv2(monkeypatch, error)
    with pytest.raises(type(error)) as caught:
        _require_cv2()
    assert caught.value is error


def test_drawing_with_opencv(monkeypatch):
    cv2 = Mock()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    utils.draw_detection_objects(image, [], [])
    utils.draw_faceobjects(image, [])
    utils.draw_pose(image, [utils.KeyPoint() for _ in range(17)])
    assert cv2.imshow.call_count == 3
    assert cv2.waitKey.call_count == 3


@pytest.mark.parametrize("filename", ["setup.py", "python/setup.py.i"])
def test_opencv_extra_metadata(monkeypatch, filename):
    root = Path(__file__).resolve().parents[2]
    setup = Mock()
    monkeypatch.setattr(setuptools, "setup", setup)
    monkeypatch.chdir(root)
    runpy.run_path(str(root / filename))
    metadata = setup.call_args[1]
    assert "opencv-python" not in metadata["install_requires"]
    assert metadata["extras_require"]["opencv"] == ["opencv-python"]


def test_missing_opencv_python35(monkeypatch):
    filename = Path(__file__).resolve().parents[1] / "ncnn/utils/_opencv.py"
    with monkeypatch.context() as patch:
        patch.delattr(builtins, "ModuleNotFoundError")
        require_cv2 = runpy.run_path(str(filename))["_require_cv2"]
    block_cv2(monkeypatch, ImportError("No module named 'cv2'", name="cv2"))
    with pytest.raises(ImportError, match=r'ncnn\[opencv\]'):
        require_cv2()


def test_yolact_with_opencv(monkeypatch):
    cv2 = Mock()
    cv2.resize.return_value = np.ones((2, 2, 2))
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    model = Yolact.__new__(Yolact)
    model.target_size = 2
    model.mean_vals = [0, 0, 0]
    model.norm_vals = [1, 1, 1]
    model.net = Mock()
    model.net.create_extractor.return_value.extract.return_value = (
        0, np.ones((2, 2, 2))
    )
    model.make_priors = Mock(return_value=[])
    model.detect = Mock(return_value=([], np.ones((2, 2)), [], []))
    boxes, masks, classes, scores = model(np.zeros((2, 2, 3), dtype=np.uint8))
    cv2.resize.assert_called_once()
    assert cv2.resize.call_args.args[1] == (2, 2)
    assert masks.shape == (2, 2, 2)
    assert masks.all()
