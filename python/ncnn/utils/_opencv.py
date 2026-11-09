# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import builtins

_MissingModuleError = getattr(builtins, "ModuleNotFoundError", ImportError)


def _require_cv2():
    try:
        import cv2
    except _MissingModuleError as error:
        if error.name != "cv2":
            raise
        raise ImportError(
            'OpenCV is required for this operation; install "ncnn[opencv]" '
            'or provide a compatible cv2 module'
        ) from error
    return cv2
