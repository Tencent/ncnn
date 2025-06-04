# Kenny Bradley 2023
# Ported yolov7-tiny to python based on:
#   - https://github.com/Qengineering/YoloV7-ncnn-Raspberry-Pi-4/blob/main/yolo.cpp
#
# Format based on the ncnn yolov4 implementation by THL A29 Limited, a Tencent company
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

import ncnn
from .model_store import get_model_file
from ..utils.objects import Detect_Object
import numpy as np


#def sigmoid_binned(val)
#   this could use a much faster binned lookup table instead of np.exp and floating division

def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))

#IOU functions:
#find the overlap width given ([x1,x2], [x3,x4]) or ([y1,y2], [y3,y4])
def calcOverlap(r1, r2):
    #r1 contains r2
    if r1[0] <= r2[0] and r1[1] >= r2[1]:
        return r2[1] - r2[0]
    #r2 contains r1
    elif r1[0] >= r2[0] and r1[1] <= r2[1]:
        return r1[1] - r1[0]
    #r1.1 is between r2.0 and r2.1
    elif r1[0] <= r2[0] and r1[1] >= r2[0]: # r1[1] <= r2[1] is true since the first if failed
        return r1[1] - r2[0]
    #r1.0 is between r2.0 and r2.1
    elif r1[0] >= r2[0] and r1[0] <= r2[1]: # r1[1] >= r2[1] is true since the second if failed
        return r2[1] - r1[0]
    else:
        return 0

#find X and Y overlaps and return intersection area
def calcIntersection(r1 : Detect_Object, r2 : Detect_Object):
    xOverlap = calcOverlap([r1.rect.x, r1.rect.x+r1.rect.w], [r2.rect.x, r2.rect.x+r2.rect.w])
    yOverlap = calcOverlap([r1.rect.y, r1.rect.y+r1.rect.h], [r2.rect.y, r2.rect.y+r2.rect.h])
    return xOverlap*yOverlap


#with r = [X1,X2,Y1,Y2] as the format return the IOU
def IOU(r1 : Detect_Object, r2 : Detect_Object):
    intersection = calcIntersection(r1,r2)
    #union =        r1 area       +        r2 area        - duplicate area
    union = (r1.rect.w*r1.rect.h) + (r2.rect.w*r2.rect.h) - intersection
    if union == 0:
        return 0
    else:
        return intersection/union

#NMS
#detections are pre-sorted in ascending confidence order
#detections are a list of Detect_Objects with : label, prob, rect
def NMS(detections, iou_thresh=0.45):
    cleanDetections = []
    detByClasses = {}
    #group by class
    for det in detections:
        #det.label is the class
        if det.label not in detByClasses.keys():
            detByClasses[det.label] = []
        detByClasses[det.label].append(det)

    #for each class find the values to keep
    for key, dets in detByClasses.items():
        for i in range(len(dets)):
            keep = 1
            #keep unless a higher priority det has IOU > thresh
            for j in range(i+1,len(dets)):
                iou = IOU(dets[i], dets[j])
                if iou > iou_thresh:
                    keep = 0
                    break
            if keep:
                cleanDetections.append(dets[i])

    #return cleaner list of Detect_Object values
    return cleanDetections
    
class YoloV7_Base:
    def __init__(self, target_size, num_threads=1, use_gpu=False, use_strides=[8,16,32]):
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu
        self.use_strides = use_strides

        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = self.num_threads

        # original pretrained model from https://github.com/AlexeyAB/darknet
        # the ncnn model https://drive.google.com/drive/folders/1YzILvh0SKQPS_lrb33dmGNq7aVTKPWS0?usp=sharing
        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param(get_model_file("yolov7-tiny.param"))
        self.net.load_model(get_model_file("yolov7-tiny.bin"))

        self.class_names = [
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"
        ]

    def __del__(self):
        self.net = None

    def __call__(self, img):

        img_h = img.shape[0]
        img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(
            img,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            img.shape[1],
            img.shape[0],
            self.target_size,
            self.target_size,
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.input("images", mat_in)

        outValues = []
        if 8 in self.use_strides:
            ret8, out8 = ex.extract("output");
            outValues.append(out8)
        else:
            outValues.append(None)

        if 16 in self.use_strides:
            ret16, out16 = ex.extract("288");
            outValues.append(out16)
        else:
            outValues.append(None)

        if 32 in self.use_strides:
            ret32, out32 = ex.extract("302");
            outValues.append(out32)
        else:
            outValues.append(None)

        #           P3/8,                  P4/16,                  P5/32
        anchors = [[12,16, 19,36, 40,28], [36,75, 76,55, 72,146], [142,110, 192,243, 459,401]]
        strides = [8,16,32]

        objects = []
        #this threshold is the value for which sigmoid gives 0.25 which is the threshold
        threshNonSigmoid = -1.098612
        for strideCount, mat_out in enumerate(outValues):
            if mat_out is None:
                continue

            stride = strides[strideCount]
            for c in range(3):
              mat = mat_out.channel(c)

              #yolo should always be square, it is expected to be 52x52
              #    but sqrt() guarantees the correct size for side
              side = int(np.sqrt(mat.h))

              anchorW = anchors[strideCount][c*2]
              anchorH = anchors[strideCount][c*2+1]
              index = 0
              for i in range(side):
                  for j in range(side):

                      #values 5-84 are class data
                      classData=mat.row(index)[5:]
                      maxLabel = max(classData)
                      
                      #optimization
                      #if either the objectness or max class score resolve to < 0.25 we can skip this
                      #  but the values are pre-sigmoid so compare to threshNonSigmoid.
                      #  1 / (1+e^(-1.098612)) = 0.25 so just compare to the -1.098612 threshold
                      if mat.row(index)[4] < threshNonSigmoid or maxLabel < threshNonSigmoid:
                          index += 1
                          continue

                      #values 0-3 are coordinate data
                      locData = mat.row(index)[0:4]
                      #value 4 is the box confidence score
                      box_score = sigmoid(mat.row(index)[4])
                      #get the highest scoring class for this detection to multiply by the box_score
                      label = np.argmax(classData)
                      class_score = sigmoid(mat.row(index)[label+5])

                      conf = box_score * class_score
                      if conf > 0.25:
                          obj = Detect_Object()
                          obj.label = self.class_names[label]
                          obj.prob = conf
                          #convert from raw yolo output to W,H and X,Y
                          obj.rect.w = ((sigmoid(locData[2]) *2) ** 2) * anchorW
                          obj.rect.h = ((sigmoid(locData[3]) *2) ** 2) * anchorH
                          obj.rect.x = ((sigmoid(locData[0]) * 2) - 0.5 + j) * stride - (obj.rect.w/2)
                          obj.rect.y = ((sigmoid(locData[1]) * 2) - 0.5 + i) * stride - (obj.rect.h/2)
                          objects.append(obj)

                      index +=1

        #sort based on probability in ascending order
        objects.sort(key = lambda x: x.prob)
        filtered_objects = NMS(objects)

        #rescale to input image size
        XscaleAdj = img_w / self.target_size
        YscaleAdj = img_h / self.target_size
        for count in range(len(filtered_objects)):
            filtered_objects[count].rect.x *= XscaleAdj
            filtered_objects[count].rect.w *= XscaleAdj
            filtered_objects[count].rect.y *= YscaleAdj
            filtered_objects[count].rect.h *= YscaleAdj

        return filtered_objects



class YoloV7_Tiny(YoloV7_Base):
    def __init__(self, **kwargs):
        super(YoloV7_Tiny, self).__init__(416, **kwargs)
