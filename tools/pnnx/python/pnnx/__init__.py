# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import os
import platform
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

