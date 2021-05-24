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

import ncnn
from .model_store import get_model_file
from ..utils.objects import Detect_Object


class PeleeNet_SSD:
    def __init__(self, target_size=304, num_threads=1, use_gpu=False):
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = [103.9, 116.7, 123.6]
        self.norm_vals = [0.017, 0.017, 0.017]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu

        # model is converted from https://github.com/eric612/MobileNet-YOLO
        # and can be downloaded from https://drive.google.com/open?id=1Wt6jKv13sBRMHgrGAJYlOlRF-o80pC0g
        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param(get_model_file("pelee.param"))
        self.net.load_model(get_model_file("pelee.bin"))

        self.class_names = [
            "background",
            "person",
            "rider",
            "car",
            "bus",
            "truck",
            "bike",
            "motor",
            "traffic light",
            "traffic sign",
            "train",
        ]

    def __del__(self):
        self.net = None

    def __call__(self, img):
        img_h = img.shape[0]
        img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(
            img,
            ncnn.Mat.PixelType.PIXEL_BGR,
            img.shape[1],
            img.shape[0],
            self.target_size,
            self.target_size,
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)

        ex.input("data", mat_in)

        ret, mat_out = ex.extract("detection_out")

        objects = []

        # printf("%d %d %d\n", mat_out.w, mat_out.h, mat_out.c)

        # method 1, use ncnn.Mat.row to get the result, no memory copy
        for i in range(mat_out.h):
            values = mat_out.row(i)

            obj = Detect_Object()
            obj.label = values[0]
            obj.prob = values[1]
            obj.rect.x = values[2] * img_w
            obj.rect.y = values[3] * img_h
            obj.rect.w = values[4] * img_w - obj.rect.x
            obj.rect.h = values[5] * img_h - obj.rect.y

            objects.append(obj)

        """
        #method 2, use ncnn.Mat->numpy.array to get the result, no memory copy too
        out = np.array(mat_out)
        for i in range(len(out)):
            values = out[i]
            obj = Detect_Object()
            obj.label = values[0]
            obj.prob = values[1]
            obj.rect.x = values[2] * img_w
            obj.rect.y = values[3] * img_h
            obj.rect.w = values[4] * img_w - obj.rect.x
            obj.rect.h = values[5] * img_h - obj.rect.y
            objects.append(obj)
        """

        ret, seg_out = ex.extract("sigmoid")

        resized = ncnn.Mat()
        ncnn.resize_bilinear(seg_out, resized, img_w, img_h)

        return objects, resized
