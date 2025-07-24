# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import platform
import subprocess

EXEC_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
if platform.system() == 'Linux' or platform.system() == "Darwin":
    EXEC_PATH = EXEC_DIR_PATH + "/pnnx"
elif platform.system() == "Windows":
    EXEC_PATH = EXEC_DIR_PATH + "/pnnx.exe"
else:
    raise Exception("Unsupported platform for pnnx.")

from .utils.export import export
from .utils.convert import convert

try:
    import importlib.metadata
    __version__ = importlib.metadata.version("pnnx")
except:
    pass

def pnnx():
    raise SystemExit(subprocess.call([EXEC_PATH] + sys.argv[1:], close_fds=False))
