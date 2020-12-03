import time
import ncnn

dr = ncnn.DataReaderFromEmpty()

net = ncnn.Net()
net.load_param("params/test.param")
net.load_model(dr)

#must use named param w, h, c for python has no size_t(unsigned int) to call the correct ncnn.Mat
in_mat = ncnn.Mat(w=227, h=227, c=3)
out_mat = ncnn.Mat()

start = time.time()

ex = net.create_extractor()
ex.input("data", in_mat)
ex.extract("output", out_mat)

end = time.time()
print("timespan = ", end - start)
