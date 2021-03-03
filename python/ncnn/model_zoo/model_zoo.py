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

from .yolov2 import MobileNet_YoloV2
from .yolov3 import MobileNetV2_YoloV3
from .yolov4 import YoloV4_Tiny, YoloV4
from .yolov5 import YoloV5s
from .yolact import Yolact
from .mobilenetssd import MobileNet_SSD
from .squeezenetssd import SqueezeNet_SSD
from .mobilenetv2ssdlite import MobileNetV2_SSDLite
from .mobilenetv3ssdlite import MobileNetV3_SSDLite
from .squeezenet import SqueezeNet
from .fasterrcnn import Faster_RCNN
from .peleenetssd import PeleeNet_SSD
from .retinaface import RetinaFace
from .rfcn import RFCN
from .shufflenetv2 import ShuffleNetV2
from .simplepose import SimplePose
from .nanodet import NanoDet

__all__ = ["get_model", "get_model_list"]

_models = {
    "mobilenet_yolov2": MobileNet_YoloV2,
    "mobilenetv2_yolov3": MobileNetV2_YoloV3,
    "yolov4_tiny": YoloV4_Tiny,
    "yolov4": YoloV4,
    "yolov5s": YoloV5s,
    "yolact": Yolact,
    "mobilenet_ssd": MobileNet_SSD,
    "squeezenet_ssd": SqueezeNet_SSD,
    "mobilenetv2_ssdlite": MobileNetV2_SSDLite,
    "mobilenetv3_ssdlite": MobileNetV3_SSDLite,
    "squeezenet": SqueezeNet,
    "faster_rcnn": Faster_RCNN,
    "peleenet_ssd": PeleeNet_SSD,
    "retinaface": RetinaFace,
    "rfcn": RFCN,
    "shufflenetv2": ShuffleNetV2,
    "simplepose": SimplePose,
    "nanodet": NanoDet,
}


def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += "%s" % ("\n\t".join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    return list(_models.keys())
