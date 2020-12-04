import ncnn
from .model_store import get_model_file
from ..utils.objects import KeyPoint

class SimplePose:
    def __init__(self, target_width=192, target_height=256, num_threads=1, use_gpu=False):
        self.target_width = target_width
        self.target_height = target_height
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = [0.485*255., 0.456*255., 0.406*255.]
        self.norm_vals = [1/0.229/255., 1/0.224/255., 1/0.225/255.]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu

        # the simple baseline human pose estimation from gluon-cv
        # https://gluon-cv.mxnet.io/build/examples_pose/demo_simple_pose.html
        # mxnet model exported via
        #      pose_net.hybridize()
        #      pose_net.export('pose')
        # then mxnet2ncnn
        # the ncnn model https://github.com/caishanli/pyncnn-assets/tree/master/models
        self.net.load_param(get_model_file("pose.param"))
        self.net.load_model(get_model_file("pose.bin"))

    def __del__(self):
        self.net = None

    def __call__(self, img):
        h = img.shape[0]
        w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img.shape[1], img.shape[0], self.target_width, self.target_height)
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)
        ex.input("data", mat_in)

        mat_out = ncnn.Mat()
        ex.extract("conv3_fwd", mat_out)

        keypoints = []

        for p in range(mat_out.c):
            m = mat_out.channel(p)

            max_prob = 0.
            max_x = 0
            max_y = 0
            for y in range(mat_out.h):
                ptr = m.row(y)
                for x in range(mat_out.w):
                    prob = ptr[x]
                    if prob > max_prob:
                        max_prob = prob
                        max_x = x
                        max_y = y

            keypoint = KeyPoint()
            keypoint.p.x = max_x * w / float(mat_out.w)
            keypoint.p.y = max_y * h / float(mat_out.h)
            keypoint.prob = max_prob

            keypoints.append(keypoint)

        return keypoints