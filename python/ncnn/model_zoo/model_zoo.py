from .yolov2 import MobileNet_YoloV2
from .yolov3 import MobileNetV2_YoloV3
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

__all__ = ['get_model', 'get_model_list']

_models = { 
            'mobilenet_yolov2': MobileNet_YoloV2,
            'mobilenetv2_yolov3': MobileNetV2_YoloV3, 
            'mobilenet_ssd': MobileNet_SSD, 
            'squeezenet_ssd': SqueezeNet_SSD, 
            'mobilenetv2_ssdlite': MobileNetV2_SSDLite, 
            'mobilenetv3_ssdlite': MobileNetV3_SSDLite, 
            'squeezenet' : SqueezeNet,
            'faster_rcnn' : Faster_RCNN,
            "peleenet_ssd" : PeleeNet_SSD,
            "retinaface" : RetinaFace,
            "rfcn" : RFCN,
            "shufflenetv2" : ShuffleNetV2,
            "simplepose" : SimplePose,
        }

def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net

def get_model_list():
    return list(_models.keys())