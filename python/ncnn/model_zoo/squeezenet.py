import numpy as np
import ncnn
from .model_store import get_model_file

class SqueezeNet:
    def __init__(self, target_size=227, num_threads=1, use_gpu=False):
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = [104.0, 117.0, 123.0]
        self.norm_vals = []

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        
        # the ncnn model https://github.com/caishanli/pyncnn-assets/tree/master/models
        self.net.load_param(get_model_file("squeezenet_v1.1.param"))
        self.net.load_model(get_model_file("squeezenet_v1.1.bin"))
            
    def __del__(self):
        self.net = None

    def __call__(self, img):
        img_h = img.shape[0]
        img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR, img.shape[1], img.shape[0], self.target_size, self.target_size)
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)

        ex.input("data", mat_in)

        mat_out = ncnn.Mat()
        ex.extract("prob", mat_out)

        #printf("%d %d %d\n", mat_out.w, mat_out.h, mat_out.c)
        
        out = np.array(mat_out)
        return out