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

import numpy as np
import ncnn
from .model_store import get_model_file


class ShuffleNetV2:
    def __init__(self, target_size=224, num_threads=1, use_gpu=False):
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu

        # https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe
        # models can be downloaded from https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe/releases
        # the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
        self.net.load_param(get_model_file("shufflenet_v2_x0.5.param"))
        self.net.load_model(get_model_file("shufflenet_v2_x0.5.bin"))

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

        ret, mat_out = ex.extract("fc")

        # manually call softmax on the fc output
        # convert result into probability
        # skip if your model already has softmax operation
        softmax = ncnn.create_layer("Softmax")

        pd = ncnn.ParamDict()
        softmax.load_param(pd)

        softmax.forward_inplace(mat_out, self.net.opt)

        mat_out = mat_out.reshape(mat_out.w * mat_out.h * mat_out.c)

        cls_scores = np.array(mat_out)
        return cls_scores
