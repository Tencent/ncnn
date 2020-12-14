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

import time
import ncnn

dr = ncnn.DataReaderFromEmpty()

net = ncnn.Net()
net.load_param("test.param")
net.load_model(dr)

in_mat = ncnn.Mat((227, 227, 3))

start = time.time()

ex = net.create_extractor()
ex.input("data", in_mat)
ret, out_mat = ex.extract("output")

end = time.time()
print("timespan = ", end - start)
